# Evaluation

What this measures, honestly: whether the service returns the *right kind of product*
for each slot of 20 human style queries. It does not measure taste, and it has no human
relevance labels. Treat every number below as a regression diagnostic, not as accuracy.

## Setup

* Queries: `scripts/eval_queries.json`, 20 requests written the way people type
  ("what should my husband wear to an outdoor wedding in june", "something to keep my
  ears warm in winter", "white sneakers under $40"). For each query i wrote, before
  running anything, the slots i expected and a rule per slot: a list of words of which at
  least one must appear in the title (`any`) and words that must not (`none`).
* `keyword_match@k`: share of returned items (k=4 per slot) whose title passes the rule of
  the slot they came back in. Returned slots are matched to my rules by name overlap;
  a slot the planner invented that i had no rule for (the planner added "gym bag" to the
  yoga query, "cummerbund" to black tie) falls back to the union of that query's rules,
  which is stricter than it should be. So the LLM configs are under-counted a little.
* `empty slots`, `price violations` (items with a known price above an explicit
  max_price) and wall clock p50/p95 per request, measured on a laptop (M4, 24 GB) with
  `claude-sonnet-4-6` through an Azure endpoint, so the LLM latencies are one provider's.
* Configs: `bm25` / `dense` / `hybrid` use the regex planner (one slot = the raw query),
  `hybrid_noboost` switches off the keyword boost and the rating prior, `llm_plan` uses
  the LLM planner with hybrid retrieval and no rerank, `llm_plan_rerank` is the full
  pipeline. `llm_plan_dense` / `llm_plan_bm25` are the LLM planner with one channel only.
* Three indexes: the default popular-100K, a seeded random 100K, and the full 826,108 rows.

Re-run with `make eval` (or `uv run python scripts/evaluate.py --index-dir ... --configs ...`),
tables come from `scripts/eval_report.py`.

## Results

### eval_popular100k

index: `index` (100,000 rows, sampling=popular), llm: claude-sonnet-4-6

| config | keyword_match@k | empty slots | price violations | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.463 | 0/20 | 0 | 8 | 11 |
| dense | 0.787 | 0/20 | 0 | 48 | 96 |
| hybrid | 0.725 | 0/20 | 0 | 53 | 70 |
| hybrid_noboost | 0.725 | 0/20 | 0 | 55 | 71 |
| llm_plan | 0.882 | 0/57 | 0 | 4496 | 6579 |
| llm_plan_rerank | 0.915 | 0/56 | 0 | 9841 | 12024 |

### eval_random100k

index: `index_random100k` (100,000 rows, sampling=random), llm: claude-sonnet-4-6

| config | keyword_match@k | empty slots | price violations | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.575 | 0/20 | 0 | 8 | 14 |
| dense | 0.812 | 0/20 | 0 | 28 | 38 |
| hybrid | 0.762 | 0/20 | 0 | 32 | 34 |
| hybrid_noboost | 0.775 | 0/20 | 0 | 33 | 36 |
| llm_plan | 0.873 | 0/55 | 0 | 4332 | 6555 |
| llm_plan_rerank | 0.871 | 0/58 | 0 | 10610 | 12379 |

### eval_full

index: `index_full` (826,108 rows, sampling=all), llm: claude-sonnet-4-6

| config | keyword_match@k | empty slots | price violations | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.588 | 0/20 | 0 | 11 | 41 |
| dense | 0.838 | 0/20 | 0 | 99 | 105 |
| hybrid | 0.738 | 0/20 | 0 | 109 | 113 |
| hybrid_noboost | 0.750 | 0/20 | 0 | 110 | 114 |
| llm_plan | 0.895 | 0/55 | 0 | 4667 | 6400 |
| llm_plan_rerank | 0.893 | 0/56 | 0 | 10294 | 12200 |

### eval_popular100k_channels

index: `index` (100,000 rows, sampling=popular), llm: claude-sonnet-4-6

| config | keyword_match@k | empty slots | price violations | p50 ms | p95 ms |
|---|---|---|---|---|---|
| llm_plan_dense | 0.886 | 0/57 | 0 | 4462 | 6088 |
| llm_plan_bm25 | 0.873 | 0/55 | 0 | 4465 | 5749 |

### eval_full_channels

index: `index_full` (826,108 rows, sampling=all), llm: claude-sonnet-4-6

| config | keyword_match@k | empty slots | price violations | p50 ms | p95 ms |
|---|---|---|---|---|---|
| llm_plan_dense | 0.917 | 0/54 | 0 | 4686 | 6529 |
| llm_plan_bm25 | 0.895 | 0/55 | 0 | 4553 | 6019 |



Brand queries are not in the 20, so i checked them separately on the full index with the
regex planner: for 8 brand queries ("levi's 501 original fit jeans", "adidas samba
sneakers", "birkenstock arizona sandals", ...) the brand name appeared in the top 4 for
25/32 results with dense only, 31/32 with hybrid, 32/32 with bm25 only.

## What i take from it

1. **The LLM planner is the feature.** With the raw sentence, even the best single
   channel stays under 0.84; product-listing style queries from the planner lift every
   index to 0.87-0.92, and they are what makes multi-slot requests possible at all (the
   regex planner can't split "outfit for the beach" into five product types).
2. **Dense beats bm25 on conversational text, bm25 beats dense on brands.** Raw sentences
   contain words like "outfit", "something", "summer" that bm25 happily matches in the
   wrong listings. Once the planner rewrites the query, the two channels are within one
   or two points of each other and hybrid sits in between. I kept hybrid as the default
   because of the brand check (31/32 vs 25/32), and because brand and model-number queries
   are common in real shops even if my 20 queries have none.
3. **The reranker helps on the popular index (+3.3 points) and is flat on the other two.**
   Its value on this metric is small because the planner already fixed product type; what
   it adds is constraint handling ("under 200 total", "for my 6 year old") and the reasons,
   which this metric cannot see. It costs ~5 s of latency (one call per slot, in parallel).
4. **Popular-100K vs full catalog.** With the full pipeline the popular subset scores
   0.915 against 0.893 for the full index, so the default subset is not hurting product
   type. The full index wins on raw dense retrieval (0.838 vs 0.787) and it has more priced
   items, which matters for strict price queries ("white sneakers under $40" found kids
   sneakers and shoelaces among the priced items of the 100K index). Serving cost: 1.0 GB
   RSS and p50 32 ms retrieval for 100K, 3.3 GB and p50 113 ms for the full index
   (`scripts/benchmark.py`, concurrency 1).
5. **Boosts are neutral here.** `hybrid_noboost` is within a point of `hybrid` on all
   three indexes. The keyword boost only does something when keywords are selective
   (planner keywords are type synonyms; the regex planner's keywords are just the query's
   content words, which nearly every candidate matches).
6. **No empty slots and no price violations** in 376 slot results, which is what the
   mask-before-top-N design and the strict explicit bound were for.

## Known gaps

* No relevance labels. Next step: label the pooled top results of these 20 queries by
  hand, then report nDCG against that instead of keyword rules.
* The keyword rules are mine and written once; they are blind to style, fit, colour and
  to whether an outfit hangs together.
* One LLM, one run per config. Planner output varies between runs (slot names and counts
  change), so differences under ~2 points are noise.
* Latency numbers include a shared machine and a remote endpoint; they are indicative.
