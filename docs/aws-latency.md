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

The second pass started from a review-panel consensus that the ordering in round one
was backwards. About 200 ms of the steady number was trans-Pacific RTT, and no code
change touches that. So the origin moved to ap-northeast-1 first, on an identical
c7i.xlarge built by the same scripts (now region generic: the public DNS comes from
describe-instances instead of string surgery, state files carry the region, the SSM
policy is in 02_iam.sh instead of my shell history). The apac Nova profiles exist in
Tokyo and invoke fine from the instance role, which settled a doc-vs-reality argument
one reviewer raised. Artifacts boot from a Tokyo bucket now, no cross-region pull.

Before touching any knob we measured the thing the knob controls. `planner_cdf.py`
times real planner calls from the box: Nova Micro p50 999 ms, Nova Lite p50 1154 ms,
and neither lands a single plan inside 350 ms out of 12. The old 0.35 s bounded wait
was pure cost. It is 0.10 s now. Cold queries answer with the regex plan about 250 ms
sooner and the LLM plan still arrives a second later, into the caches, for every
request after.

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

Repeat-mode ramp through CloudFront, fresh TLS connection per request: p50 43-47 ms
at c=1 through c=8, 40 rps at c=8, zero errors. CloudFront costs 10-15 ms on a warm
hit versus the bare origin and pays for it with edge TLS, HTTP/3 (enabled, credited
with nothing) and the /assets cache. Every row sits under the 0.5 s target, cold
included.

Worker count stayed at 2. The panel was unanimous that dropping to 1 for cache
warmth trades a failure domain for noise, and the loopback ramp agreed: p50s within
run noise, 2 workers held the better p95 at c=4 and c=16. The first attempt at that
comparison accidently benchmarked the per-process rate limiter (a wall of 429s at
c=8), which is its own small lesson about load-testing through your own guardrails.

### Quality, measured against the live endpoint

The 28 evaluation queries ran against production through CloudFront, scored by the
same rules as the local harness, after a clean restart per model. Nova Micro planned
0.705 match@4 micro; Nova Lite 0.772, query success 0.50 versus 0.39, and 28 of 28
plans landed. The planner is Nova Lite now. It costs the user nothing, because
planning is background work by design, and the round-one claim that Lite was not
worth it did not survive contact with an actual quality probe. The remaining gap to
the Sonnet-planned local eval (0.885 on this index) is planner model quality, and
the honest fix is the Haiku use-case form or a bigger Nova, not pipeline work.

Left open, deliberately: the per-worker cache split (each worker warms seperately, so a
cold query can flip between LLM and regex answers for about a second), the shared
cache that would fix it, and CloudWatch alarms on the fallback rate. The instance
runs about $4/day; `99_teardown.sh` still removes everything it made, in both
regions.


## Round three: the acceptance battery, and what it caught

Latency probes tell you the fast paths are fast. An acceptance battery asks harder
questions: does the contract hold, do the guardrails fail the right way, what does a
deploy cost, and where does the machine actually saturate. `deploy/aws/test_battery.py`
runs ten sections against the live endpoint; the raw results are exp20 to exp22.

The headline is a bug. The first steady run came back with a server-side p50 of
2.44 seconds on one worker while the other answered the same query in half a
millisecond. Not rerank, not Bedrock: retrieve_ms was the whole number, on an idle
box. The trigger turned out to be plan shape. Nova Lite had given one worker a beach
plan with five slots and a total budget of $135; tight price masks plus queries like
"women's summer cover-up" left the type gate short of matches, so the ranker widened
its window four-fold per pass, and every widening pass fully hydrated every window row
through per-row pandas iloc. A cProfile on the box showed 19,783 row hydrations in one
request, 6.5 of 7.4 seconds inside pandas. Under 12 parallel connections the same plan
hit 10.4 seconds. And becuase the plan carried warnings, the response cache refused it
each time, so the worker re-paid the full price on every request.

The fix reads scoring fields (title, rating, count, audience, group key) with one
vectorized column gather per window and leaves the full row walk to the handful of
rows a slot actually returns. Replaying the captured plan on the box: 2,831 ms before,
985 ms after in a cold process. The live steady p50 went from 2,442 ms to 12.8 ms. A
regression test now counts full hydrations per request and fails if a window ever gets
row-walked again. Worth saying plainly: this path never showed in us-east-1 (Micro's
plans rarely carry budgets) or in the local eval (Sonnet plans). Only live Lite
traffic walked it, wich is exactly why you load-test the deployed thing and not a
model of it.

The rest of the battery, briefly. Contract: 20 of 20, every malformed input gets its
422/413/405 with the documented error body, every response carries its full schema.
Guardrails: 70 rapid requests met their first 429 at request 65 (two workers means two
token buckets, so the effective burst doubles: noted, accepted for a demo), throttles
answered in 12 ms with a well-formed body, and service resumed clean after 20 s.
Steady keep-alive n=100: p50 12.8 ms, p99 16.9 ms, max 17.5 ms. Fresh TLS per request
n=50: p50 34.9 ms. The 21-cell ramp (repeat, unique and mixed at c=1 through 32, 672
requests) finished with zero errors: repeat traffic holds 36 to 52 ms p50 all the way
up (100 req/s peak), unique cold traffic saturates the four vCPUs at 10 to 13 req/s
with p50 climbing to 2.7 s at c=32 and not a single 5xx, just honest queuing. The five
minute soak at c=6 mixed: 5,010 requests, 16.7 req/s, zero errors, per-minute p95 flat
between 870 and 1,010 ms with no drift. Host memory held at 1.8 to 2.1 GB the whole
time. Restart under traffic: one systemd restart costs 6.7 s of 502/504 through
CloudFront, then full recovery; a second origin behind failover removes that, and it
sits first on the production gap list. The cache ladder passed with assertions once
the probe rode a single keep-alive connection like a browser would (a fresh connection
per rung round-robins across workers and reads the split as a failure). Five identical
warm requests returned identical items. And the 28-query quality probe, rerun after
the fix at production settings: match@4 micro 0.800 on the planned pass, success 0.50,
28 of 28 plans landed. The fix moved where fields are read, not how rows rank.
