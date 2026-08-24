# Design notes

Longer version of the "why" behind the service. The README has the short one. These are
the decisions i went back and forth on, with the numbers that settled them where i had
numbers. Written while building, so some of it reads like a lab notebook.

## What the data allowed

The assignment pdf lists 14 fields. After profiling all 826,108 rows, five do real work:

* `title`: 100%, median 89 chars, packed with brand + gender + type + colour + size.
* `rating_number`, `average_rating`, `images`: 100%.
* `features`: 56% non-empty, median 11 chars ("Pull On closure").
* Dead: `categories` (empty everywhere), `bought_together` (always null). `price` 6.1%,
  `description` 7%.

That kills two ideas i had on day one: a category tree for filtering, and "people also
bought" bundles. Everything has to come out of the title text. Which is fine, its a very
informative title.

One more thing that decided the subset policy. Coverage goes up sharply with popularity:

| rating_number | rows | price | features | description | department |
|---|---|---|---|---|---|
| 0-4 | 450,202 | 4.9% | 50% | 6% | 12% |
| 5-19 | 274,398 | 5.3% | 61% | 6% | 14% |
| 20-99 | 85,025 | 11% | 68% | 12% | 25% |
| 100+ | 16,483 | 26% | 79% | 25% | 49% |

(measured at ingest, `data/processed/ingest_stats.json`)

## Subset: why popular-first by default

Popular-first, stated everywhere:

* Full catalog embed: ~19 min laptop gpu, 40+ min cpu, 3.3 GB RSS to serve; 100K most
  rated: ~3 min, 1 GB.
* Metadata quality climbs with popularity (price 5% -> 26%, features 50% -> 79%,
  department 12% -> 49%).
* The bias is stated in `index_info.sampling` and measured against random and full.

Popular-first is a bias, i know. The defence is the table above: the listings with many
ratings are the ones with prices, features and a department, which is exactly what the
LLM needs to explain a pick. The evaluation doc compares the three indexes on the same 20
queries so the cost of the bias is visible instead of hidden. Every response also carries
`index_info.sampling`, so a client can tell which index answered.

## Retrieval: hybrid, and why masks come before top-N

Dense only (bge-small) was clearly better than bm25 only on conversational queries in the
notebook experiment, but bm25 wins on exact phrases ("wedding guest", brand names) and it
is basicaly free. Reciprocal rank fusion with k=60 needs no score calibration between
the two, which is why i picked it over a weighted sum.

The part i'd defend hardest: constraints are boolean masks on the full score vectors,
applied *before* top-N.

* The naive version (retrieve 60, filter by price and audience) quietly empties slots
  when eligible items sit past the cut.
* With masks, one matching women's sandal under $30 anywhere in the index is found.
* Cost: an O(n) boolean op over 100K rows, microseconds.
* Small additive terms on top: keyword boost, Bayesian rating prior.

Two small additive terms sit on top of the fused score: +0.5 x RRF(rank 1) when a
planner keyword literally appears in the title, and +0.1 x RRF(rank 1) x normalised
Bayesian rating. Both are tunable by env var and the eval has a no-boost config so you
can see what they do.

All slot queries of a request are embedded in one batch and scored with a single matrix
multiply. Five slots = one matmul, not five.

## Variants and duplicates

* 56,720 titles repeat under different `parent_asin`; many more differ only by size or
  colour suffix.
* Nothing deleted at ingest (twin rows can carry different prices and images).
* `group_key` = lowercased title minus trailing parens, size tokens, colour/size
  comma-segments. Highest-scored row represents the group; one slot per group.
* A heuristic: it merges "Hanes socks, 6 pairs" with the 12-pack. Right call for
  recommendations.

## The planner

The LLM's job is translation: how people talk -> how listings are written.

* Beach query -> five slots (swimsuit, cover up, sandals, sun hat, sunglasses), each
  with a listing-style query + title keywords. Single items -> one slot.
* Structured output on both providers: no free-text parsing.
* A normaliser enforces what a json schema cant: max 5 slots, keywords deduped and
  capped at 6, budget allocations never above the total.

Without a key, or when the call fails or the deadline is close, a regex planner takes
over: one slot with the raw query, budget from patterns like "under $50" or "$20-40",
audience from words like "men's". It's english-only and it can't split outfits, but it
never fails, and the response says which planner ran.

## The reranker

* One call per slot, slots in parallel: output tokens dominate, so five slots cost
  about what one costs.
* Input: top 10 candidates as compact json (row id, title, price or null, rating,
  audience, store, material/colour/style, matched keywords, off_type_hint).
* Output: ordered picks + one-sentence reason + evidence fields.
* Everything checked: ids belong to the slot, duplicates dropped, the rest keeps
  retrieval order. Unpriced picks stay `price_known: false`.

Catalog text goes into the prompt as data inside a json blob, labelled untrusted, with an
instruction to never follow anything inside it. That's not a guarantee against prompt
injection, but the output validation means the worst a hostile title can do is pick a
different candidate from the same slot.

Why an LLM reranker and not a cross-encoder? It handles constraints a cross encoder can't
see ("for my 6 year old", "under 40 dollars total") and it writes the reasons, which the
UI shows. The price is 3-12 seconds of latency depending on the model, so it is one flag
away from off (`rerank=false`).

## Prices: strict when explicit, flagged when inferred

94% of items have no price.

* Explicit `max_price`: only known prices inside the window (a 100K index has ~11K
  priced rows).
* V1 applied that rule to inferred budgets too. Result: "husband, outdoor wedding,
  budget 200 total" put a priced wooden ring in the blazer slot while unpriced real
  blazers were filtered out.
* Now: inferred budgets keep unpriced items, flagged `price_known: false` + warning,
  with a small bonus for priced in-budget rows.
* `include_unpriced` overrides either way. Per-slot allocations get a 10% floor.

Planner failures do get a short negative cache (30 s, PLANNER_FAILURE_TTL_S) so an
outage is not retried on every single request; a waiter whose own budget ran out never
posions it for everyone else, only a real failure of the shared call does.

## Deadlines and failure

One request deadline (40 s). Planner gets at most 15 s, reranker at most 20 s, and neither
is started if less than its minimum remains. Every provider error is mapped to a typed
error (auth, rate limit, timeout, refusal, truncation, validation, transport) and every one
of them has a fallback path with a test. The service does not know or care which SDK is
underneath.

## What i would do next

1. Human relevance labels for the 20 eval queries. The keyword metric is a regression
   check, not a relevance judgement.
2. An ANN index (FAISS HNSW or a hosted vector db) once the catalog stops fitting in RAM.
   Exact search is fine at 100K-800K rows and i'd rather ship exact than tune recall.
3. A cross-encoder stage between retrieval and the LLM to cut the 10 candidates to 8 and
   shave rerank latency.
4. CLIP image embeddings. Every listing has an image and "something like this photo" is an
   obvious query type the current design can't serve.
5. Near-duplicate detection with embedding similarity, the group key is string based.
