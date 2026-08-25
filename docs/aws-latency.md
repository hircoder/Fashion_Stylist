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

* Cold query: ~440 ms with the regex plan (waits only 350 ms for the LLM).
* The Nova call finishes in the background; the plan lands in the exact + semantic
  caches; later requests and paraphrases get it at cache speed.
* Verified live: beach -> 5 slots from Nova Micro; a paraphrase -> same 5 slots,
  `plan_cache_hit: true`, zero calls, 215 ms wall.

Two bugs died here, both found on the deployed thing:

1. The shared planner call inherited the 0.35 s waiter budget as its own timeout, so it
   never finished in the background. Now it runs on its own clock.
2. MIN_PLAN_SECONDS skipped the planner entirely below 0.5 s. The floor now guards the
   remaining request deadline, not the configured wait.

Also: background failures used to vanish (no waiter left). They now log and
negative-cache, wich is how the next finding surfaced.

## Model findings

* `us.amazon.nova-micro-v1:0`: works via the Converse API with one forced tool, gives
  the full 5-slot outfit decomposition, plan fills in ~1.5 to 2.5 s in the background.
  The default.
* `us.amazon.nova-lite-v1:0`: same quality on the probe queries, slower fill (p95 of a
  mixed run 2.7 s). Not worth it here. (Round two overturned this with a real quality
  probe; see below.)
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


## Round two: Tokyo, measured budgets, and two cache bugs

Review-panel consensus: round one's ordering was backwards. ~200 ms of the steady
number was Pacific RTT; no code touches that. So:

* Origin moved to ap-northeast-1 first. Same scripts, now region generic (public DNS
  from describe-instances, per-region state files, SSM policy in 02_iam.sh).
* apac Nova profiles exist in Tokyo and invoke from the instance role (settled a
  doc-vs-reality dispute). Artifacts boot from a Tokyo bucket, no cross-region pull.
* Knobs measured before moving: `planner_cdf.py` on the box. Micro p50 999 ms, Lite
  1154 ms, 0/12 plans inside 350 ms. The 0.35 s wait was pure cost -> 0.10 s. Cold
  queries answer ~250 ms sooner; the LLM plan still lands ~1 s later for everyone
  after.

Two cache bugs found by the panel, both real, both fixed with tests:

1. Background plan completion only fed the exact-string cache. A paraphrase missed it
   and paid for a fresh Bedrock call, forever. The completion callback now stores
   into the semantic cache too, and the live ladder proves it: a paraphrase of a
   query planned 5 s earlier serves the LLM plan in 116 ms with zero new calls.
2. The new response cache could freeze a degraded answer. First request waits 100 ms,
   gets the regex fallback, the fallback gets cached for the ttl while the good plan
   lands 1 s later, unseen. Now only warning-free responses are cached (every
   fallback path writes a warning), and a hit says so: `served_from_cache: true`,
   usage counters zeroed, the timing field holds the real serve time instead of a
   flattering 0.0.

Also new: a per-service cap of 8 concurrent background planner calls, so a flood of
unique cold queries cannot build an unbounded Bedrock queue behind clients who
already got their answer. And the slot-query embedding LRU, which is why a warm
5-slot request now spends under 1 ms of server time.

### The numbers, from a client in Japan

| path | round one (us-east-1) | round two (Tokyo) |
|---|---|---|
| steady keep-alive p50 | 344-372 ms | 57 ms |
| warm repeat (response cache) | n/a | 13-17 ms wall, ~0.3 ms server |
| cold unique query | ~640 ms | 210 ms |
| plan-cache hit after background fill | n/a | 112 ms |
| paraphrase via semantic cache | 215 ms | 116 ms |
| direct origin warm hit | n/a | 22 ms |

* Ramp (repeat mode, fresh TLS/request): p50 43-47 ms at c=1..8, 40 rps, zero errors.
* CloudFront costs 10-15 ms over the bare origin; buys edge TLS, HTTP/3 (credited with
  nothing), /assets cache. Every row under 0.5 s, cold included.
* Workers stay at 2: p50s within noise, better p95 at c=4/c=16, second failure domain.
  First comparison attempt accidently benchmarked the per-process rate limiter (a wall
  of 429s): load-test past your own guardrails, then put them back.

### Quality, measured against the live endpoint

28 queries POSTed at production, offline rules, clean restart per model:

* Nova Micro 0.705 match@4 micro, success 0.39. Nova Lite 0.772, success 0.50, 28/28
  plans landed. Planner = Lite; background work, zero user latency cost.
* Round one's "Lite not worth it" did not survive a real quality probe.
* Gap to Sonnet-planned local eval (0.885): planner model quality. Fix = Haiku
  use-case form or a bigger Nova, not pipeline work.

Left open, deliberately: per-worker cache split (~1 s of flip-flop on cold queries),
the shared cache that fixes it, CloudWatch alarms on fallback rate. ~$4/day;
`99_teardown.sh` removes everything, both regions.


## Round three: the acceptance battery, and what it caught

Latency probes prove the fast path. The battery asks the rest: contract, guardrail
behavior, deploy cost, saturation point. `deploy/aws/test_battery.py`, ten sections,
raw results exp20 to exp22.

The headline is a bug:

* Symptom: server p50 2.44 s on one worker, 0.4 ms on the other, idle box.
  retrieve_ms was the whole number.
* Trigger: a Nova Lite beach plan, 5 slots, $135 total budget. Tight price masks +
  "women's summer X" queries starved the type gate; the ranker widened its window x4
  per pass; every pass hydrated every window row via per-row pandas iloc.
* Profile: 19,783 row hydrations in one request, 6.5 of 7.4 s inside pandas. 10.4 s
  under 12 parallel connections. The response cache refused to mask it (the plan
  carried warnings; degraded answers are never cached).
* Fix: one vectorized column gather per window for scoring fields (title, rating,
  count, audience, group key); the row walk only for returned rows.
* Proof: captured plan replay 2,831 -> 985 ms cold; live steady p50 2,442 -> 12.8 ms;
  a regression test counts hydrations per request.
* Never showed on us-east-1 (Micro rarely emits budgets) or the local eval (Sonnet
  plans). Only live Lite traffic walked it. Load-test the deployed thing, not a model
  of it.

The rest of the battery:

* Contract: 20/20. Every malformed input -> 422/413/405 with the documented body;
  every response carries its full schema.
* Guardrails: 70 rapid requests, first 429 at #65 (two workers = two token buckets,
  effective burst doubles; noted), throttles in 12 ms, clean recovery in 20 s.
* Steady keep-alive n=100: p50 12.8 ms, p99 16.9 ms, max 17.5 ms. Fresh TLS n=50:
  p50 34.9 ms.
* Ramp, 21 cells, c=1..32, 672 requests, zero errors. Repeat: 36-52 ms p50, 100 req/s
  peak. Unique cold: 4 vCPUs saturate at 10-13 req/s, p50 2.7 s at c=32, zero 5xx.
* Soak, 5 min at c=6 mixed: 5,010 requests, 16.7 req/s, zero errors, per-minute p95
  flat 870-1,010 ms, host memory flat 1.8-2.1 GB.
* Restart under traffic: 6.7 s of 502/504, full recovery. First item on the gap list
  (origin failover removes it).
* Cache ladder: passes with assertions on one keep-alive connection (fresh connections
  round-robin workers and read the split as failure). Consistency x5: identical.
* Quality rerun after the fix: match@4 micro 0.800, success 0.50, 28/28 plans. The fix
  moved where fields are read, not how rows rank.


## Round four: all 826,108 rows

The 100K index was always the pragmatic subset, and the eval had already priced the
difference (0.885 vs 0.935 on the same plans, every brand query on-brand only on the
full catalog). Round four moves production onto the full index and re-measures.

The switch, on the same box:

* The full index (818 MB on disk, 3.3 GB in memory per process) went up as its own
  tarball; the 100K copy stays on the box, so rollback is one rename and a restart.
* Workers went from 2 to 1: two full-index copies do not fit in 8 GB. Memory after
  the switch: 3.1 GB used, 4.3 GB spare, load average 0.72 under the probes.
* One immediate lesson: with one worker there is one token bucket, and the first
  probe run throttled itself (41 of 50 fresh requests hit 429). The limiter came off
  for the measurement window, went back after, and the docs now say limits are
  per process rather than per worker.

The numbers, from the same Japan client through CloudFront (exp23 to exp25):

| path | 100K, 2 workers | full 826K, 1 worker |
|---|---|---|
| steady keep-alive p50 | 12.8 ms | 13.0 ms |
| fresh TLS p50 | 34.9 ms | 36.5 ms |
| response-cache hit | 13-17 ms | 15 ms |
| cold unique query | 210 ms | 464 ms |
| uncached answer on an LLM plan | ~0.4 s | 0.7-1.0 s server-side |
| unique cold capacity | 10-13 req/s | ~3.8 req/s |

Cached paths did not move: they never touch the matmul, and the response and plan
caches do the answering. The bill lands entirely on uncached work, roughly the 8.3x
row count divided by the lost worker. Capacity for unique cold traffic is now ~3.8
req/s per box, wich replaces the old 10-13 as the replica-sizing number.

Quality moved the right way on every measure (exp25, original 28 queries, Nova Lite
plans): match@4 micro 0.800 -> 0.819, macro 0.872, success 0.50 -> 0.571, zero empty
slots, 28/28 plans landed.

exp26 runs the extended 78-query set against the same live service: 0.666 micro /
0.749 macro / 0.628 success overall, with the original 28 averaging 0.861 inside it
and the new 50 averaging 0.685. The per-class split names the weak spot: open-ended
outfit decomposition (0.41) and loose conversational asks (0.48) under a small
planner, while materials, budgets and the look-alike traps hold 0.81-0.88.

Left open, stated: the cold path now peeks over the 0.5 s target (464 ms is fine;
0.7-1.0 s on an uncached model plan is not), so the next latency work is retrieval
(int8 vectors, pre-tokenised bm25) or a 16 GB box that restores the second worker.
