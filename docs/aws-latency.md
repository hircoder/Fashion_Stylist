# AWS deployment: getting under half a second

Branch `AWS_deployment`. Goal: p50 prompt-to-result under 0.5 s on an AWS stack with
Bedrock doing the LLM work and CloudFront in front. Everything below was measured on a
live deployment on 2026-08-24; raw run data in `deploy/aws/experiments/`.

## The shape

* EC2 c7i.xlarge (us-east-1, Amazon Linux 2023), bootstrapped by user-data from two S3
  tarballs (source + prebuilt 100K index), systemd runs uvicorn with 2 workers. No
  docker anywhere: the build machine doesnt have it and it turned out not to be missed.
* CloudFront in front of the instance (http origin on :8000, security group only admits
  the cloudfront origin-facing prefix list). `/assets/*` cached at the edge (verified
  Miss then Hit), everything else passes through. The UI is served through the edge.
* Bedrock via the instance role, `us.amazon.nova-micro-v1:0` as the planner, no keys
  anywhere on the box.
* The fast profile is configuration, not new architecture: PLANNER_BUDGET_S=0.35 (a
  request waits at most 350 ms for a plan), PLANNER_CALL_TIMEOUT_S=20 (the shared call
  finishes in the background and lands in the cache), RERANK_DEFAULT=0 (deterministic
  reasons; a client can still ask for the LLM rerank per request), SEMANTIC_PLAN_CACHE=1
  (near-duplicate queries reuse the nearest plan, guarded so a different budget or
  audience never crosses over).

## The numbers (measured from Japan, so ~200 ms of every wall number is the Pacific)

| experiment | p50 wall | p95 wall | notes |
|---|---|---|---|
| retrieval only (use_llm=false) | 280 ms | 304 ms | server ~75 ms |
| fast profile, cold queries | 639 ms | 831 ms | 350 ms bounded plan wait + work |
| fast profile, steady state | 372 ms | 646 ms | 30/36 requests on LLM plans |
| steady + connection kept alive | 344 ms | 550 ms | what a browser actually does |
| individual warm requests | 213 to 300 ms | | server 65 to 130 ms, calls 0 |
| LLM rerank switched on | 1.33 s | 1.48 s | why the profile leaves it off |

The served path is under 0.5 s at p50 from the far side of the planet; a client in the
US would lose most of the remaining ~200 ms of RTT. Server-side, a warm request is 65 to
130 ms end to end: retrieval over 100K rows plus plan lookup.

## How the plan gets fast

A cold query is answered in ~440 ms with the regex plan (the request only waits 350 ms
for the LLM). The Nova call keeps running in the background, its plan lands in both the
exact and the semantic cache, and every later request for that query, or anything close
to it, gets the full LLM plan at cache speed. Verified live: the beach query returns
["swimsuit", "cover-up", "sandals", "sun hat", "sunglasses"] from Nova Micro, and a
paraphrase ("i need an outfit for going to the beach in summer") comes back with the
same 5 slots, `plan_cache_hit: true`, zero LLM calls, 215 ms wall.

Two bugs had to die to make this true, both found by measuring the deployed thing:
1. The shared planner call inherited the 0.35 s waiter budget as its own timeout, so it
   never finished in the background. It now runs on its own clock.
2. The MIN_PLAN_SECONDS floor treated the configured wait as a mistake and skipped the
   planner entirely below 0.5 s. The floor now guards the remaining request deadline,
   not the configured wait.
Also: a background failure used to vanish (no waiter left to observe it). It now logs
and negative-caches, wich is exactly how the next finding surfaced.

## Model findings

* `us.amazon.nova-micro-v1:0`: works via the Converse API with one forced tool, gives
  the full 5-slot outfit decomposition, plan fills in ~1.5 to 2.5 s in the background.
  The default.
* `us.amazon.nova-lite-v1:0`: same quality on the probe queries, slower fill (p95 of a
  mixed run 2.7 s). Not worth it here.
* `us.anthropic.claude-haiku-4-5-20251001-v1:0`: invocable with the operator's user
  credentials but NOT with the instance role: Bedrock answers ResourceNotFoundException
  "model use case details have not been submitted". One console form away; a finding,
  not a fix, for now.
* Rerank stays off in the profile; the measured LLM rerank adds a second. The
  cross-encoder from docs/production.md remains the right next step for reranked
  sub-second responses.

## Costs while it runs

c7i.xlarge on demand about $0.17/h (~$4/day), CloudFront + S3 pennies at test volume,
Nova Micro tokens fractions of a cent per plan. `deploy/aws/99_teardown.sh` removes the
distribution, instance, EIP, security group, role and bucket.

## What i would do next on this branch

Shared plan cache (Redis/ElastiCache) so workers and replicas stop warming separately;
the Haiku use-case form; the cross-encoder rerank stage on the instance (it has 4 idle
vCPUs at this load); a us-east client for honest non-Pacific numbers; CloudWatch alarms
on the fallback rate now that it is on the request log line.
