# Exploration: notebooks, scripts, prompt experiments

What i used to understand the data, pick the models and tune the prompts, and what each
one found. Everything here is reproducible from the repo.

## notebooks/01_explore_data.ipynb (executed, outputs kept)

Written for two audiences at once: each section opens with the question in plain
words and closes with the decision it forced. Two parts, twelve sections, ten charts.

Part 1, before any code was written:

1. Field coverage over all 826,108 rows: `categories` and `bought_together` empty on every
   row, price 6.1%, description 7.2%, features 56%, department 14.5%, images and titles
   100%. Title length median 89 chars. Rating count median 4, 16,483 rows with 100+.
2. Coverage by popularity bucket (from the ingest stats): price 4.9% -> 26%, features 50%
   -> 79%, department 12% -> 49% going from 0-4 ratings to 100+. This is the evidence for
   the popular-first demo index (ADR-005).
3. Audience split (49% women, 24% unguessable and treated as a wildcard) and the
   heuristics checked on a sample of real titles before trusting them.
4. Three embedding models on 40K listings with 8 conversational queries, side by side
   with bm25: bge-small-en-v1.5 761 docs/s, all-MiniLM-L6-v2 1,411 docs/s,
   arctic-embed-xs 1,458 docs/s. bge-small gave the most sensible lists; bm25 failed on
   "outfit for the beach" and "keep my ears warm" and won on "wedding guest". Also showed
   gender leaking through dense retrieval ("men's chinos" returned a women's office suit),
   which is why audience is a mask, not a hope.
5. Planner examples: the beach query, a men's wedding outfit with a total budget (shows the
   split across slots), and a French query (shows translation into English listing
   queries).

Part 2, the analytics behind the production decisions:

6. Prices in depth: the 50,249 known prices (median $19.89, 53% under $20), coverage by
   audience (women 4.2%, unisex 12.3%), and the strict-explicit / flagged-inferred policy
   they forced (ADR-008).
7. Ratings: star averages cluster high (mean 3.91) on tiny samples, and the Bayesian
   adjustment with a worked example (a lone 5.0 becomes 3.96; a 4.8 from 500 barely moves).
8. Variants: 83,621 groups cover 262,880 rows, a third of the catalog; 56,720 rows repeat
   another row's title (case-insensitive); the biggest "group" is 151 dead listings titled
   "marked for archive". Query-time collapse, ADR-009.
9. Brands: rows mentioning each evaluation brand, full catalog vs the popular 100K
   (levi's: 50 vs 3). The measured reason brand queries need the full catalog.
10. The evaluation evidence drawn from `docs/eval_*.json`: match@4 by configuration and
    index, the planner as the big jump, the full catalog on top, and what it costs
    (18.4 min build, 3.3 GB, 110 ms retrieval).
11. Where the seconds go: stage timings from the recorded live run; retrieval is ~1% of
    an 11.5 s request, which is what justified background planning on the AWS branch.
12. A closing map from each chart to the decision it settled.

## scripts/

* `evaluate.py` + `eval_queries.json`: 28 queries (20 conversational, 8 brand) with hand written slot rules, nine
  retrieval/LLM configurations, three indexes. Produces the json in `docs/eval_*.json`;
  `eval_report.py` turns them into the tables in `docs/evaluation.md`.
* `benchmark.py`: serving RSS after load and retrieval latency at concurrency 1, 2, 4.
  Numbers in `docs/evaluation.md` and the README.
* `make_fixture.py`: the 486 row stratified sample that ships in the repo for tests and
  the no-download demo.

## Prompt experiments (what changed and why)

The planner and reranker prompts live in `src/stylist/llm/prompts.py` with a version
string that is part of the plan cache key. The changes that came out of running real
queries, in order:

1. **Listing-style queries.** The first planner wrote search queries that were shorter
   versions of the shopper's sentence. Asking for "audience + product type + 2-4
   attributes, never the shopper's sentence" moved type-match from raw-sentence levels
   (0.73) to 0.88 on the same retrieval.
2. **Keywords as type synonyms only.** Adjectives and colours in keywords boosted the
   wrong titles. Now: 2 to 6 product type synonyms, no adjectives, no audience words.
3. **One rerank call per slot.** A single call for five slots produced 20 s responses
   and timed out on a mid size model. Per slot, in parallel, the rerank takes 3 to 7 s.
4. **Type before price.** With unpriced items in the pool the reranker preferred a
   priced wooden ring over an unpriced blazer because the prompt said "prefer items in
   budget". Rewritten as a ranked list of criteria: product type first, then fit to the
   request, then price (only between equally good matches), then ratings. "A priced wooden
   ring is not a blazer" is in the prompt because that is what happened.
5. **Exclude words.** Cover-ups carry "swimsuit" in their titles, so the swimsuit slot
   filled with cover-ups. The planner now emits `exclude_keywords` per slot (swimsuit:
   "cover up"); retrieval penalises those titles and the reranker gets an
   `off_type_hint`. Then a second lesson: the planner excluded "swimsuit" from the
   cover-up slot, which is exactly the word real cover-ups contain. The prompt now says
   never to exclude words that correct listings commonly contain, and the penalty equals
   the keyword boost so an ambiguous title nets zero instead of disappearing.
6. **Budget splits.** For "budget 200 total" one run gave the blazer $10. A floor of 10%
   of the total per slot and proportional scaling now sit between the planner and the
   retriever; the prompt also asks for shoes and outerwear to get more than accessories.
7. **Shorter reasons.** Reasons capped at 15 words and one slot note of 20 words, which
   halves output tokens and therefore latency.
8. **Untrusted data framing.** Candidate fields are passed as json labelled as catalog
   data with an instruction never to follow text inside them; every returned id is
   validated, so the worst an injected title can do is pick a different candidate.

## Dead ends

* A hand built product taxonomy from titles: too much to maintain for a prototype, and
  the planner keywords plus exclude words cover the same risk well enough on the eval.
* Baking `group_key` into the index: one regex fix would have cost a 20 minute rebuild.
  Moved to query time.
* Strict prices everywhere: see ADR-008.
