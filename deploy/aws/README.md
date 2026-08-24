# deploy/aws: everything in this directory, on one page

The AWS deployment kit: five build scripts, one teardown, six probes, and 22
experiment records. This file is the map and the context; `docs/aws-latency.md`
tells the story in order.

## What gets built

Browser -> CloudFront (edge TLS, HTTP/3, `/assets` cached hard, price class All)
-> EC2 origin (c7i.xlarge, systemd, 2 uvicorn workers, no docker, boots from S3
tarballs) -> Bedrock in the same region (Nova Lite plans in the background, Nova
Micro reranks on request).

Security posture, in one breath: port 8000 admits only the CloudFront
origin-facing prefix list, no ssh keys exist, ops go through SSM, the instance
role is the only credential, keys never touch the box.

## The scripts, in run order

| script | what it does | the detail that matters |
|---|---|---|
| `00_env.sh` | names, region, per-region model + bucket picks | `STYLIST_REGION=ap-northeast-1` flips everything regional; lite plans + micro reranks in apac |
| `01_artifacts.sh` | source + index tarballs into S3 | `COPYFILE_DISABLE=1` and `--exclude '._*'`: macOS AppleDouble files once broke the loader on EC2 |
| `02_iam.sh` | instance role | s3 read (both buckets via wildcard), bedrock invoke, SSM core. Nothing else |
| `03_ec2.sh` | SG, instance, elastic IP | AMI via SSM parameter (region safe); public DNS read from describe-instances, never string-built; state written to `.instance.<region>` |
| `04_cloudfront.sh` | one distribution | CachingDisabled + AllViewerExceptHostHeader for the app, CachingOptimized for `/assets` |
| `05_test.sh` | smoke probes through the distribution | |
| `99_teardown.sh` | removes everything, every region with a state file | ~$4/day while it runs, so run this when done |
| `user-data.sh` | the boot script template | dnf + uv + tarballs + model pre-download + systemd unit + warmup (two passes over the six UI examples) |

## The knobs the unit sets (fast profile)

* `PLANNER_BUDGET_S=0.10`: a request waits 100 ms for the plan, then answers with
  the regex plan. Measured, not taste: `planner_cdf.py` showed 0 of 12 plans land
  inside 350 ms (Micro p50 999 ms, Lite 1154 ms).
* `PLANNER_CALL_TIMEOUT_S=20`: the shared background call runs on its own clock
  and lands in the exact AND semantic plan caches. At most 8 run at once.
* `RERANK_DEFAULT=0`: deterministic reasons by default; `rerank: true` per request
  brings the LLM rerank (~1.4-1.8 s with Micro).
* `SEMANTIC_PLAN_CACHE=1`: paraphrases reuse the nearest plan at cosine 0.92,
  budget and audience guarded.
* `RESPONSE_CACHE_TTL_S=300`: identical bodies served from memory, warning-free
  answers only, hits say `served_from_cache: true`.
* `RATE_LIMIT_PER_MINUTE=240`, `TRUST_PROXY_HEADERS=1` (client ip from CloudFront).

## The probes

| probe | what it measures |
|---|---|
| `latency_probe.py` | percentiles per configuration, keep-alive or fresh connections, server timings split out |
| `load_probe.py` | concurrency ramp, repeat / unique / mixed workloads, p50/p95/p99 + rps |
| `planner_cdf.py` | real planner completion times from the box: the evidence behind the 0.10 s wait |
| `quality_probe.py` | the 28 eval queries POSTed at production, scored by the offline rules, cold and planned passes |
| `test_battery.py` | ten acceptance sections: contract, headers, guardrails, steady, fresh TLS, ramp, soak, restart under traffic, cache ladder, consistency |
| `profile_retrieval.py` pattern | (scratch, not committed) cProfile replay of a captured plan on the box; found the hydration hot path |

## The experiments (exp01 to exp22)

Round one, us-east-1, Nova Micro planning:

* exp01 retrieval only 280 ms p50; exp02-07 fast profile cold/warm through two bug
  fixes (shared-call timeout, MIN_PLAN floor): cold 639 -> steady 344-389 ms.
* exp08 rerank on: 1.33 s (why the profile leaves it off).
* exp09 steady keep-alive 344 ms. exp10-11 Haiku attempt (role needs a console
  use-case form; documented). exp12-13 Lite vs Micro steady: 388 / 372 ms.

Round two, Tokyo:

* exp14 steady 57.3 ms (warm hits 13-17 ms). exp15 repeat ramp: 43-47 ms p50, 40
  rps at c=8, zero errors.
* exp16 planner CDF (the 0.10 s evidence). exp17 worker ramp: kept 2 workers.
* exp18 live quality: Lite 0.772 match@4 vs Micro 0.705; the planner is Lite.
* exp19 cold ladder: cold 210 / plan-cache 112 / response-cache 34 / paraphrase
  116 ms, all from a Japan client.

Round three, the battery:

* exp20: ~11,500 requests. Contract 20/20, guardrail envelope clean, 21 ramp cells
  zero errors, 5 min soak 5,010 requests zero errors and flat p95, restart costs
  6.7 s of 5xx, cache ladder asserted, consistency x5 identical.
* exp21: the bug the battery caught. A budget-carrying plan starved the type gate,
  widened the ranking window, and hydrated every window row through per-row pandas
  iloc: 19,783 hydrations per request, 2.44 s server p50, 10.4 s under 12-way
  load. Fixed with vectorized column gathers; 985 ms cold replay, 12.8 ms live
  steady after. Load peaked 7.51 on 4 vCPUs during the unique cells; host memory
  flat 1.8-2.1 GB.
* exp22: quality after the fix, match@4 micro 0.800, success 0.50: ranking did not
  move.

`SUMMARY.json` holds one row per experiment with the headline numbers.

## Capacity and limits, measured

* Repeat traffic: 36-52 ms p50 from c=1 to c=32, 100 req/s peak per box.
* Unique cold traffic: 4 vCPUs saturate at 10-13 req/s; size replicas from THIS.
* Known limits, stated: single origin (a deploy costs ~7 s of 5xx; failover fixes
  it), per-worker rate buckets (effective burst doubles), per-worker caches (~1 s
  of answer flip on cold queries; a shared cache fixes it).

## Runbook

```bash
# deploy code: re-tar, upload, extract, restart (warmup refires)
git archive --format=tar.gz -o /tmp/src.tar.gz HEAD
aws s3 cp /tmp/src.tar.gz s3://<bucket>/src/stylist-src.tar.gz
aws ssm send-command --document-name AWS-RunShellScript --instance-ids <id> \
  --parameters 'commands=["cd /opt/stylist && aws s3 cp s3://<bucket>/src/stylist-src.tar.gz src.tar.gz && tar -xzf src.tar.gz && rm src.tar.gz && systemctl restart stylist"]'

# change a knob: edit the systemd drop-in, restart
#   /etc/systemd/system/stylist.service.d/override.conf

# logs
journalctl -u stylist --since "-10 min"
cat /var/log/stylist-boot.log /var/log/stylist-warmup.log

# everything down
./99_teardown.sh
```

## In one line

Five scripts build it in any region, six probes measure it, 22 experiment files
prove every number in the docs, and the teardown removes all of it.
