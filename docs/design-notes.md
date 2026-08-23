# Design notes

Longer version of the "why" behind the service. The README has the short one. These are
the decisions i went back and forth on, with the numbers that settled them where i had
numbers. Written while building, so some of it reads like a lab notebook.

## What the data allowed

The assignment pdf lists 14 fields. After profiling all 826,108 rows, five of them do
real work: `title` (always there, median 89 chars, stuffed with brand + gender + type +
colour + size), `rating_number`, `average_rating`, `images` (100% have one) and `features`
(56% non-empty, but the median feature list is 11 characters long, so mostly "Pull On
closure"). `categories` is an empty list on every row. `bought_together` is null on every
row. `price` is a float on 6.1% of rows and null otherwise. `description` shows up 7% of the
time.

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

Embedding the whole catalog takes about 20 minutes on an M-series laptop and 40+ on CPU,
and needs ~1.2 GB of RAM just for the float32 matrix when serving. Nobody reviewing a
take-home wants to wait for that. So `make index` takes the 100K most-rated listings
(about 2.7 minutes on the laptop), `--sampling random` gives a seeded long-tail sample
instead, and `--sampling all` does the full thing.

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

The part i'd defend hardest: constraints are applied as boolean masks on the full score
vectors *before* taking top-N. The naive version (retrieve 60, then filter by price and
audience) quietly returns empty slots as soon as the eligible items sit at rank 61+. With
masks, if there is a single womens sandal under 30 dollars anywhere in the index, the slot
gets it. Cost: the mask is an O(n) boolean op over 100K rows, microseconds.

Two small additive terms sit on top of the fused score: +0.5 x RRF(rank 1) when a
planner keyword literally appears in the title, and +0.1 x RRF(rank 1) x normalised
Bayesian rating. Both are tunable by env var and the eval has a no-boost config so you
can see what they do.

All slot queries of a request are embedded in one batch and scored with a single matrix
multiply. Five slots = one matmul, not five.

## Variants and duplicates

56,720 titles appear more than once under different `parent_asin`, and many more differ
only by a size or colour suffix (`..., Black, Large` vs `..., White, Small`, or
`(Blue, Size 9-12)`). Deleting them at ingest felt wrong, the two rows can carry different
prices and images. So every row is kept and gets a `group_key`: lowercased title with
trailing parentheses, size tokens and colour/size comma-segments stripped. At query time
the highest scored row of a group represents it, and a group can only fill one slot of an
outfit. The key is a heuristic and it will merge "Hanes socks, 6 pairs" with the 12 pairs
listing, which i think is the right call for recommendations.

## The planner

The LLM's job is translation, from how people talk to how product listings are written.
"I need an outfit to go to the beach this summer" becomes five slots (swimsuit, cover up,
sandals, sun hat, sunglasses), each with a product-listing style query and a handful of
title keywords. Single item requests become one slot. Structured output (pydantic schema
on both providers) means i never parse free text, and a normaliser enforces the things a
json schema can't: at most 5 slots, keywords deduped and capped at 6, per-slot budget
allocations that don't add up to more than the total.

Without a key, or when the call fails or the deadline is close, a regex planner takes
over: one slot with the raw query, budget from patterns like "under $50" or "$20-40",
audience from words like "men's". It's english-only and it can't split outfits, but it
never fails, and the response says which planner ran.

## The reranker

One call per request, all slots at once, up to 15 candidates per slot as compact json
(row id, title, price, rating, audience, store, material/colour/style when present,
matched keywords). The model returns ordered picks with a one sentence reason and the
evidence fields it used. Everything it returns is checked: ids must belong to that slot,
duplicates dropped, the rest of the slot keeps retrieval order, and rows that were only
backfilled from the unpriced pool can never be promoted above in-window rows.

Catalog text goes into the prompt as data inside a json blob, labelled untrusted, with an
instruction to never follow anything inside it. That's not a guarantee against prompt
injection, but the output validation means the worst a hostile title can do is pick a
different candidate from the same slot.

Why an LLM reranker and not a cross-encoder? It handles constraints a cross encoder can't
see ("for my 6 year old", "under 40 dollars total") and it writes the reasons, which the
UI shows. The price is 3-12 seconds of latency depending on the model, so it is one flag
away from off (`rerank=false`).

## Prices: strict by default

94% of items have no price. If someone asks for "under $40" and we return an unpriced
item, we can't claim it fits. So a price bound only admits items with a known price inside
the window. That shrinks the pool a lot (a 100K index has ~11K priced items), which is why
`include_unpriced=true` exists: it tops a short slot up with unpriced items, each one
flagged `price_known: false`, plus a warning in the response. The default is the honest
one, the relaxed mode is opt-in.

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
3. A cross-encoder stage between retrieval and the LLM to cut the 15 candidates to 8 and
   shave rerank latency.
4. CLIP image embeddings. Every listing has an image and "something like this photo" is an
   obvious query type the current design can't serve.
5. Near-duplicate detection with embedding similarity, the group key is string based.
