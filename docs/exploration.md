# Exploration: notebooks, scripts, prompt experiments

What i used to understand the data, pick the models and tune the prompts, and what each
one found. Everything here is reproducible from the repo.

## notebooks/01_explore_data.ipynb (executed, outputs kept)

1. Field coverage over all 826,108 rows: `categories` and `bought_together` empty on every
   row, price 6.1%, description 7.2%, features 56%, department 14.5%, images and titles
   100%. Title length median 89 chars. Rating count median 4, 16,483 rows with 100+.
2. Coverage by popularity bucket (from the ingest stats): price 4.9% -> 26%, features 50%
   -> 79%, department 12% -> 49% going from 0-4 ratings to 100+. This is the evidence for
   the popular-first default index (ADR-005).
3. Audience heuristic and variant grouping on a sample of real titles, to eyeball the
   `derive_audience` and `group_key` rules before trusting them.
4. Three embedding models on 40K listings with 8 conversational queries, side by side
   with bm25: bge-small-en-v1.5 761 docs/s, all-MiniLM-L6-v2 1,411 docs/s,
   arctic-embed-xs 1,458 docs/s. bge-small gave the most sensible lists; bm25 failed on
   "outfit for the beach" and "keep my ears warm" and won on "wedding guest". Also showed
   gender leaking through dense retrieval ("men's chinos" returned a women's office suit),
   which is why audience is a mask, not a hope.
5. Planner examples: the beach query, a men's wedding outfit with a total budget (shows the
   split across slots), and a French query (shows translation into English listing
   queries).

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
