# Evaluation

What this measures, honestly: whether the service returns the *right kind of product*
(and, for brand queries, the right brand) for each slot of 28 human style queries. It does
not measure taste and it has no human relevance labels. Treat every number below as a
regression diagnostic, not as accuracy.

## Setup

* Queries: `scripts/eval_queries.json`. 20 requests written the way people type ("what
  should my husband wear to an outdoor wedding in june", "something to keep my ears warm
  in winter", "white sneakers under $40") plus 8 brand requests ("nike running shoes for
  men", "levi's jeans for men"). For each query the expected slots and a rule per slot were
  written before running anything: `any` = at least one of these words must appear in the
  title (whole words, plural tolerant), `none` = none of these may, `all` = one of these
  must (the brand, so a branded sock never passes a jeans slot).
* Metrics, all at k=4 items per slot:
  * `match@k`: share of returned items whose title passes the rule of the slot they came
    back in. A returned slot is mapped to a rule by exact name, then by the best word
    overlap; a slot the planner invented that has no rule ("beach bag" in the beach
    query) is scored against the union of the query's rules, which keeps the forbidden
    words and the brand requirement. Those slots are counted in `unmapped`, about 20 of
    65 slots on every index, and they make the LLM configs look a little worse than they
    are.
  * `macro`: the same rate averaged per query, with a percentile bootstrap 95% interval
    over queries. A failed request or an all-empty answer counts as zero.
  * `mapped precision`: `match@k` restricted to slots that mapped to a rule.
  * `slot recall`: share of the expected slots (rule names) that some returned slot mapped
    to. The regex planner returns one slot called "items", so its recall is zero by
    construction; compare it on `match@k` only.
  * `query success`: every expected slot came back, none is empty, and every returned
    slot has at least one passing item. Strict on purpose.
  * `empty slots`, `price violations` (a known price outside an explicit min/max), wall
    clock p50/p95 per request.
  * Paired deltas: the mean of per-query differences between two configurations on the
    same queries, with a bootstrap interval. An interval that excludes zero is a real
    difference on this set; most are not.
* Configurations: `bm25` / `dense` / `hybrid` use the regex planner (one slot = the raw
  sentence); `hybrid_nokw` and `hybrid_noquality` switch off the keyword boost and the
  rating prior one at a time; `llm_plan` is the LLM planner with hybrid retrieval and no
  rerank; `llm_plan_dense` / `llm_plan_bm25` the LLM planner with one channel; and
  `llm_plan_rerank` the full pipeline. Every llm configuration on every index uses the
  same 28 plans (`docs/eval_plans.json`, written by the first run and reloaded by the
  others), so the comparison is paired.
* LLM: `claude-sonnet-4-6` through an Azure endpoint, prompt version 2, evaluation
  budgets of 45 s per stage (so a slow provider hour measures quality, not timeouts; the
  production defaults are 15 s and 20 s). Latencies of the llm configs on the random and
  full indexes are with the plan cache warm; the cold numbers, 4 to 7 s for the plan, are
  the popular index's `llm_plan` row and `docs/live_run_sonnet.json`.
* Three indexes: the default popular-100K, a seeded random 100K, and the full 826,108
  rows, all built with pipeline version 3 and the pinned model revision.

Re-run with `make eval` (or `python scripts/evaluate.py --index-dir ... --configs ...
--plan-cache docs/eval_plans.json`); the tables come from `scripts/eval_report.py`.

## Results

### popular 100K (the default index)

index: `index` (100,000 rows, sampling=popular), llm: claude-sonnet-4-6, prompt v2, 28 queries, code b4be12b

| config | match@k | macro (95% CI) | mapped precision | slot recall | query success | empty slots | price viol. | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.500 | 0.500 (0.35 to 0.66) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 8 | 10 |
| dense | 0.696 | 0.696 (0.55 to 0.82) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 28 | 35 |
| hybrid | 0.625 | 0.625 (0.48 to 0.76) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 32 | 40 |
| hybrid_nokw | 0.625 | 0.625 (0.49 to 0.77) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 31 | 42 |
| hybrid_noquality | 0.625 | 0.625 (0.48 to 0.76) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 31 | 39 |
| llm_plan | 0.881 | 0.848 (0.74 to 0.94) | 0.900 | 0.88 | 0.61 | 0/65 | 0 | 4357 | 7110 |
| llm_plan_dense | 0.862 | 0.840 (0.73 to 0.93) | 0.906 | 0.88 | 0.64 | 0/65 | 0 | 59 | 100 |
| llm_plan_bm25 | 0.865 | 0.826 (0.70 to 0.93) | 0.889 | 0.88 | 0.61 | 0/65 | 0 | 33 | 52 |
| llm_plan_rerank | 0.885 | 0.857 (0.76 to 0.94) | 0.911 | 0.88 | 0.61 | 0/65 | 0 | 5501 | 6766 |

paired differences on the same queries (mean of per-query match rate, 95% bootstrap interval):

| comparison | mean delta | 95% CI | n |
|---|---|---|---|
| bm25 - hybrid | -0.125 | -0.23 to -0.02 | 28 |
| dense - hybrid | +0.071 | +0.01 to +0.13 | 28 |
| hybrid_nokw - hybrid | +0.000 | -0.05 to +0.05 | 28 |
| hybrid_noquality - hybrid | +0.000 | +0.00 to +0.00 | 28 |
| llm_plan_dense - llm_plan | -0.008 | -0.04 to +0.02 | 28 |
| llm_plan_bm25 - llm_plan | -0.022 | -0.05 to -0.00 | 28 |
| llm_plan_rerank - llm_plan | +0.009 | -0.00 to +0.03 | 28 |

### random 100K

index: `index_random100k` (100,000 rows, sampling=random), llm: claude-sonnet-4-6, prompt v2, 28 queries, code 6d28732

| config | match@k | macro (95% CI) | mapped precision | slot recall | query success | empty slots | price viol. | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.562 | 0.562 (0.40 to 0.73) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 7 | 10 |
| dense | 0.696 | 0.696 (0.55 to 0.82) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 29 | 39 |
| hybrid | 0.679 | 0.679 (0.54 to 0.81) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 36 | 45 |
| hybrid_nokw | 0.705 | 0.705 (0.57 to 0.83) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 33 | 45 |
| hybrid_noquality | 0.679 | 0.679 (0.54 to 0.81) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 31 | 42 |
| llm_plan | 0.881 | 0.849 (0.76 to 0.93) | 0.906 | 0.88 | 0.68 | 0/65 | 0 | 59 | 123 |
| llm_plan_dense | 0.877 | 0.861 (0.77 to 0.94) | 0.911 | 0.88 | 0.68 | 0/65 | 0 | 59 | 92 |
| llm_plan_bm25 | 0.877 | 0.839 (0.75 to 0.92) | 0.900 | 0.88 | 0.64 | 0/65 | 0 | 31 | 45 |
| llm_plan_rerank | 0.885 | 0.857 (0.75 to 0.95) | 0.911 | 0.88 | 0.61 | 0/65 | 0 | 5360 | 6367 |

paired differences on the same queries (mean of per-query match rate, 95% bootstrap interval):

| comparison | mean delta | 95% CI | n |
|---|---|---|---|
| bm25 - hybrid | -0.116 | -0.25 to +0.01 | 28 |
| dense - hybrid | +0.018 | -0.07 to +0.11 | 28 |
| hybrid_nokw - hybrid | +0.027 | +0.00 to +0.07 | 28 |
| hybrid_noquality - hybrid | +0.000 | +0.00 to +0.00 | 28 |
| llm_plan_dense - llm_plan | +0.012 | -0.01 to +0.04 | 28 |
| llm_plan_bm25 - llm_plan | -0.009 | -0.03 to +0.01 | 28 |
| llm_plan_rerank - llm_plan | +0.008 | -0.04 to +0.05 | 28 |

### full catalog, 826,108 rows

index: `index_full` (826,108 rows, sampling=all), llm: claude-sonnet-4-6, prompt v2, 28 queries, code 6d28732

| config | match@k | macro (95% CI) | mapped precision | slot recall | query success | empty slots | price viol. | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.652 | 0.652 (0.48 to 0.80) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 12 | 37 |
| dense | 0.786 | 0.786 (0.67 to 0.89) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 106 | 128 |
| hybrid | 0.750 | 0.750 (0.62 to 0.86) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 114 | 124 |
| hybrid_nokw | 0.750 | 0.750 (0.62 to 0.87) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 110 | 127 |
| hybrid_noquality | 0.750 | 0.750 (0.62 to 0.86) | 0.000 | 0.00 | 0.00 | 0/28 | 0 | 111 | 126 |
| llm_plan | 0.935 | 0.959 (0.92 to 0.99) | 0.972 | 0.88 | 0.71 | 0/65 | 0 | 244 | 463 |
| llm_plan_dense | 0.923 | 0.953 (0.92 to 0.99) | 0.972 | 0.88 | 0.71 | 0/65 | 0 | 234 | 419 |
| llm_plan_bm25 | 0.915 | 0.935 (0.87 to 0.98) | 0.961 | 0.88 | 0.68 | 0/65 | 0 | 87 | 200 |
| llm_plan_rerank | 0.935 | 0.967 (0.94 to 0.99) | 0.978 | 0.88 | 0.71 | 0/65 | 0 | 5529 | 6472 |

paired differences on the same queries (mean of per-query match rate, 95% bootstrap interval):

| comparison | mean delta | 95% CI | n |
|---|---|---|---|
| bm25 - hybrid | -0.098 | -0.18 to -0.02 | 28 |
| dense - hybrid | +0.036 | -0.06 to +0.12 | 28 |
| hybrid_nokw - hybrid | +0.000 | -0.03 to +0.03 | 28 |
| hybrid_noquality - hybrid | +0.000 | +0.00 to +0.00 | 28 |
| llm_plan_dense - llm_plan | -0.006 | -0.01 to +0.00 | 28 |
| llm_plan_bm25 - llm_plan | -0.024 | -0.06 to +0.00 | 28 |
| llm_plan_rerank - llm_plan | +0.008 | -0.01 to +0.03 | 28 |

## What the numbers say

* The planner is the feature. The raw sentence gets 0.63 to 0.75 on hybrid retrieval;
  the planner's listing-style queries lift every index to 0.88 to 0.94 and give the
  outfit queries their slots (recall 0.88: the planner skips one expected slot in about
  one query in eight, typically "sunglasses" for "beach bag").
* Dense beats bm25 on sentences on every index (bm25 - hybrid is -0.12 to -0.13 with the
  regex planner). Under the LLM planner the two channels are within 2 points and the
  intervals include zero; hybrid is kept for brand and exact-phrase queries, where bm25
  is the channel that carries the brand token.
* The reranker adds about 1 point of match@k on the same plans (+0.009, interval
  -0.00 to +0.03). Its value is elsewhere: it reads the constraints (audience, budget,
  occasion), writes the reasons, and it is what keeps an off-type item out of a slot
  when retrieval order would have shown it (the admissibility rule, ADR-0015).
* The keyword boost and the rating prior do nothing measurable on this metric (deltas of
  0.00 to +0.03 with intervals through zero). They stay for the ordering inside a slot,
  which the metric cannot see.
* The full catalog is better than any 100K subset on the same plans: 0.935 against 0.885,
  and all eight brand queries come back fully on-brand, where the popular subset has no
  Levi's jeans at all and three Columbia fleeces for a rain jacket request. The popular
  subset is not losing product type; it loses coverage of specific brands and the long
  tail, which is the trade the default makes for a 3 minute build and 1 GB of memory.
* Zero empty slots and zero price violations in every configuration on every index.
  `query success` is 0.61 to 0.71 for the full pipeline because it demands every
  expected slot plus a passing item in every returned slot, including the invented ones
  scored against the union.

## Brand queries, by index

| query | popular 100K | random 100K | full |
|---|---|---|---|
| nike running shoes for men | 4/4 | 4/4 | 4/4 |
| adidas track pants | 4/4 | 3/4 | 4/4 |
| the north face jacket for women | 4/4 | 4/4 | 4/4 |
| champion hoodie | 4/4 | 4/4 | 4/4 |
| fruit of the loom underwear for men | 4/4 | 2/4 | 4/4 |
| columbia rain jacket for women | 0/4 | 0/4 | 4/4 |
| levi's jeans for men | 0/4 | 0/4 | 4/4 |
| calvin klein underwear for men | 2/4 | 3/4 | 4/4 |

Items passing the brand-and-type rule out of 4, full pipeline. The zeros are catalog
gaps on the subsets (one Levi's jeans row, no Columbia rain jacket); in those cases the
response says so in a warning and shows the product type from other brands.

## Caveats, so nobody over-reads this

* 28 queries is enough to catch a regression of 5 points or more and not enough to
  rank two configurations that differ by 2; the intervals say so.
* The rules check titles. A correct product whose title never names its type (model
  names only) fails the rule, and a wrong product that mentions the type word passes.
  `mapped precision` of 0.91 to 0.98 is therefore an upper bound on precision and a
  lower bound at the same time.
* The LLM is not deterministic. Run to run, `llm_plan_rerank` moved by up to 3 points
  on the popular index in this project's history; the paired plan cache removes the
  planner's share of that noise, not the reranker's.
* Everything was measured on one machine (M4 laptop) against one provider endpoint;
  latency columns describe that setup only. `docs/production.md` has the throughput and
  cost measurements.
* There is no human judgement anywhere in this. The next step in `docs/production.md`
  (200 labelled queries, nDCG, a calibrated LLM judge) is the step that turns these into
  relevance numbers.
