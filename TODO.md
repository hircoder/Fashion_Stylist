# TODO: the path from working demo to production-grade service

An honest audit of what is missing, wrong, or undone, with a solution and a plan
per item. Ordered by priority. Effort estimates assume one engineer.

What this list is NOT: the demo working. The service is live, measured (57 ms
steady p50 from Japan, 11,500-request battery, zero unexplained 5xx) and every
degradation is visible in the response. The gaps below are what separates that
from something an SRE team would sign off on.

## P0: production blockers

### 1. No authentication on the API
* Problem: `/recommend` is open. Anyone with the URL can burn Bedrock tokens and
  CPU. The per-IP limiter slows abuse, it does not stop it.
* Solution: an API key checked at the edge (CloudFront function verifying a
  signed header) or, once an ALB exists, OIDC / Cognito. Per-key quotas in the
  limiter, keys in Secrets Manager.
* Plan: CloudFront function + key table (1 day). Move to ALB auth with item 2.

### 2. Single origin: no failover, deploys drop traffic
* Problem: one EC2 instance. A deploy costs a measured 6.7 s of 502/504; an
  instance failure is a full outage. No autoscaling, no multi-AZ.
* Solution: ALB in front of an autoscaling group (min 2, two AZs) or an ECS
  service (item 3), health checks on `/ready`, connection draining, rolling
  deploys. CloudFront origin becomes the ALB with an ACM certificate, which also
  closes the http-only origin hop.
* Plan: 2-3 days together with item 3. The deploy window goes from 6.7 s to zero.

### 3. Two deployment paths: the container is not what AWS runs
* Problem: a Dockerfile exists (Railway/CI path) but the AWS box runs bare
  systemd from a source tarball. Two paths drift; the battle-tested one is not
  the containerised one.
* Solution: one artifact. Build the image in CI, push to ECR, run it on ECS
  Fargate (2 tasks, 2 vCPU / 4 GB each fits the measured 1 GB index footprint).
  The systemd path stays as the cheap dev fallback.
* Plan: ECR + task definition + service behind the ALB from item 2 (2 days).
  Add trivy image scanning to CI (half a day).

### 4. No alarms, no dashboard
* Problem: the request log line carries fallback rate, stage timings and token
  counts, and nothing watches it. A Bedrock outage at 3am is silent.
* Solution: CloudWatch alarms on p95, 5xx rate, planner fallback rate, Bedrock
  throttles and a monthly budget; one dashboard with the request rate, stage
  latencies, cache hit rates and token spend.
* Plan: 1 day. The metrics already exist in the logs; this is wiring.

### 5. Logs die with the instance
* Problem: journald only. Terminate the box, lose the history.
* Solution: structured JSON logs (request id, stages, usage, warnings as fields)
  shipped by the CloudWatch agent; 30 day retention; queries via Logs Insights.
* Plan: JSON formatter + agent config (1 day). OTel traces later (P2).

## P1: hardening

### 6. Shared cache
* Problem: each worker warms its own plan / semantic / response caches. Cold
  queries flip answers for about a second; more replicas make it worse.
* Solution: ElastiCache Redis for the exact-plan and response caches and the
  rate limiter; the semantic cache stays in-process (it needs the vectors) with
  Redis-backed plan bodies. Cache keys already carry prompt version.
* Plan: 1-2 days. Also makes the limiter global (today burst doubles per worker).

### 7. Edge protections
* Problem: rate limiting is in-process; no WAF; origin reachable over plain http
  from CloudFront (SG-locked to the prefix list, but plaintext).
* Solution: WAF rate rules + IP reputation on the distribution; the ALB + ACM
  from item 2 makes the origin hop TLS; a custom origin secret header so only
  CloudFront reaches the ALB.
* Plan: half a day once the ALB exists.

### 8. CI/CD to AWS
* Problem: deploys are hand-run SSM commands. Honest for a take-home, not
  repeatable engineering.
* Solution: GitHub Actions: test -> build -> push ECR -> ECS rolling deploy ->
  smoke probes -> auto-rollback on alarm. The battery script becomes the smoke
  stage.
* Plan: 2 days after item 3.

### 9. Index and catalog refresh
* Problem: the catalog is a static snapshot; the index is rebuilt by hand.
* Solution: a scheduled job (EventBridge -> ECS task) that ingests the latest
  dump, builds, uploads with a new sha, and flips INDEX_URL; rollback is
  flipping it back. The loader already refuses mismatched artifacts.
* Plan: 2 days. Data versioning is already in meta.json.

### 10. Provider circuit breaker
* Problem: failures negative-cache for 30 s, but a hard outage still sends every
  30th request into a doomed call.
* Solution: a real breaker (closed / open / half-open) around each provider with
  failure-rate thresholds; fallback stays the regex planner.
* Plan: 1 day, mostly tests.

## P2: quality and scale

### 11. Human relevance labels
* Problem: the eval is a type-correctness floor; it cannot see style or fit.
* Solution: 200 labelled queries (3 graders, majority vote), nDCG@4 and slot
  recall; an LLM judge calibrated against those labels for the nightly run.
* Plan: 1-2 weeks including the grading round.

### 12. Sub-second full-rerank mode
* Problem: full rerank is 1.4-1.8 s (Nova Micro). Good, not great.
* Solution: a cross-encoder (bge-reranker-v2-m3) on the origin's idle vCPUs:
  50 pairs in 60-120 ms; reasons from the template or streamed after. The LLM
  stays as an async refinement.
* Plan: 3 days behind a flag, gated on the eval set.

### 13. Streaming
* Problem: the answer arrives all at once.
* Solution: SSE per slot: retrieval-order items render in ~200 ms, reranked
  order swaps in as calls land.
* Plan: 2-3 days, UI included.

### 14. Plan determinism
* Problem: identical queries can get different plans across calls even at
  temperature 0 (provider-side nondeterminism), so cold answers vary.
* Solution: short term, plan pinning by query hash in the shared cache. Long
  term, the distilled 1-3B planner (10-20K pairs from the current planner),
  which is deterministic, ~200 ms, and cheap.
* Plan: pinning is free with item 6; distillation 1-2 weeks.

### 15. Catalog scale
* Problem: exact search holds to about 1M rows x 384; memory and p95 grow
  linearly after.
* Solution: HNSW (faiss or usearch) with the masks applied post-search plus
  over-fetch, int8 embeddings (4x memory cut), row-group serving.
* Plan: when the catalog demands it, 1 week.

### 16. Cost hygiene
* Problem: no budget alarm; the demo box is oversized for steady load.
* Solution: AWS Budgets alert at $50/mo; c7i.large (half cost) until unique-query
  load says otherwise (measured ceiling 10-13 req/s on xlarge, halve it);
  Savings Plan when the shape settles; S3 lifecycle on old artifacts. Token
  spend is already visible per request.
* Plan: half a day.

### 17. DR and multi-region
* Problem: one region. The scripts are region generic, but nothing rehearses it.
* Solution: document and drill the second-region bring-up (the us-east-1 to
  Tokyo move WAS one: about 30 minutes end to end). Active-passive with a
  Route 53 health check when a real SLO demands it.
* Plan: a written runbook now (the deploy README is most of it); drill quarterly.

## Small known items

* `.env.example` line 7 still names an older model; add `LLM_RERANK_MODEL` there
  too (file was locked to tooling edits).
* Haiku on Bedrock via the instance role needs the one-time console use-case
  form (documented in docs/aws-latency.md).
* 12 h soak and chaos tests (LLM garbage / hang, index corruption on boot) are
  specified in docs/production.md but not yet run; the 5 min soak and the
  battery are.
