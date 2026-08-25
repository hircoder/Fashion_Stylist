# Fashion Stylist: technical overview

Markdown mirror of `docs/overview.html` (the animated deck served at `/overview`). The html version adds the agenda, the animated data flow on slide 5 and the sample tabs; the content here is the same.


---

## Fashion Stylist: a semantic recommendation service

_1 / 19 · Take-home project_

The brief: a small service that takes a sentence like the one below, works out what the person actually needs, and finds it in the Amazon Fashion catalog (826,108 listings) using semantic search and an LLM. Exposed as an API. This page is the short version; the README, the PRD, the decision records and the evaluation doc in `docs/` have the long one.

shopper types: I need an outfit to go to the beach this summer

the planner turns it into slots, then each slot is searched and ranked on its own:
* swimsuit women's one piece or bikini
* cover-up lightweight beach dress
* sandals flat, open toe
* sun hat wide brim straw
* sunglasses uv protection

---

## Scope and business case

_2 / 19 · Scope and business case_

A catalog search box understands product words. "I need an outfit to go to the beach this summer" contains no product word at all, and the budget, the audience or the occasion hiding in a sentence get dropped by keyword search. The business value of fixing that: one search box that accepts situations, bundles instead of single items, and answers that explain themselves so the product team can trust and debug them.

I wrote a short PRD first (`docs/prd.md`) and kept it honest as the data pushed back. The requirements and how each one is met:

| requirement | how | evidence |
|---|---|---|
| parse a sentence into product types and constraints | LLM planner, structured output, 1 to 5 slots; regex planner as fallback | type-match 0.73 raw sentence, 0.88 with the planner (the query-rewrite experiment, slide 4) |
| find relevant products per type | hybrid retrieval: meaning search (embeddings) + exact-word search (bm25), merged by rank | 0.935 on the full catalog the service serves, 0.885 on the quick 100K build |
| respect audience and budget | eligibility filters applied before the top-N cut, budgets split per slot | 0 price violations, 0 empty slots across every configuration and index |
| explain every pick | per slot LLM rerank, reason + evidence fields, ids validated | samples on slide 9 |
| be honest about prices (94% unknown) | `price_known` flag, strict explicit bounds, flagged inferred ones | ADR-008 |
| API, CLI, page; works without a key | FastAPI + OpenAPI, argparse CLI, React page, `LLM_PROVIDER=none` | `make demo` in two minutes |

Non-goals: personalisation, image understanding, model training, auth. Success metrics: type-match above 0.85 with the planner (got 0.885 on the 100K subsets, 0.935 on the full catalog), zero empty slots and price violations (got both), a reviewer running it in under five minutes without a download or a key (about two: one for setup, one for the demo index build).

---

## Data: what the catalog contains and what it lacks

_3 / 19 · Data_

I profiled every row before writing code. Two of the fourteen fields in the brief are empty on every single row (`categories`, `bought_together`): no taxonomy, no "people also bought". Price exists for 6% of listings. The title is always there and it's dense: brand, gender, type, colour, size, in a median of 89 characters.

How often each field is filled in, across all 826,108 listings:

| field | non-empty |
|---|---|
| title, images, rating | 100% |
| features | 56% |
| department (inside details) | 14.5% |
| description | 7.2% |
| price | 6.1% |
| categories, bought_together | 0% |

The same fields, split by how many ratings a listing has. Read down a column: coverage climbs with popularity.

| ratings count | rows | price | features | dept. |
|---|---|---|---|---|
| 0 to 4 | 450,202 | 4.9% | 50% | 12% |
| 5 to 19 | 274,398 | 5.3% | 61% | 14% |
| 20 to 99 | 85,025 | 11% | 68% | 25% |
| 100+ | 16,483 | 26% | 79% | 49% |
* What the right table says: a listing with 100+ ratings has a price 26% of the time, features 79%, a department 49%. A listing with 0 to 4 ratings: 4.9%, 50%, 12%. More popular means better documented.
* Why that matters: the price filter can only act on rows that HAVE a price, and the reranker can only justify a pick with fields that exist. Thin rows can only be recommended on trust.
* The day-one decision: a quick index of the 100K most-rated listings, wich here means roughly everything with 20+ ratings (101,508 rows, so the last 1,508 of that tie miss the cut). It builds in 2.6 minutes. The deployed service has since moved to the full 826K catalog; the quick build stays for demos and comparisons.
* Indexing only the popular rows is a bias, so it is never hidden: every response names the subset it was answered from (`index_info.sampling`), and the evaluation on slide 10 runs the same queries against a random sample and the full catalog to measure what the subset costs.
* The catalog is full of near-duplicates: 56,720 titles appear more than once under different product ids, mostly the same item in another size or colour. At query time these collapse to one result each, through a group key computed from the title.
* Search by meaning alone leaks gender: "men's chinos" returned a women's office suit at rank 3, and plain keyword search on that query was no better. That is why rows labeled for another audience are excluded before ranking (rows with no guessable audience stay in), instead of hoping a similarity score gets it right.

---

## Exploration: scripts, notebooks and prompt experiments

_4 / 19 · Exploration_

### notebooks/01_explore_data.ipynb (executed)
* Two parts, twelve sections, ten charts; every section opens with the question in plain words and closes with the decision it forced.
* Part one, before any code: field coverage, coverage by popularity, the audience guess (checked against the department field: where both speak, they agree 78% of the time), and three embedding models against bm25 on 40K listings. bm25 failed "outfit for the beach" (toddler outfits, a cat purse) and won "wedding guest"; bge-small gave the best lists and became the default.
* Part two, while building: prices in depth, a worked example of the rating adjustment, variant twins (the biggest "group" is 151 dead listings titled "marked for archive"), brand survival under the popularity cut, the evaluation evidence, and where the seconds go.
* Planner examples: the beach query, a men's wedding outfit with a total budget, a French query translated into English listing queries.

(inline screenshot: see the html deck)

_From the notebook: title lengths (median 89 characters), how many ratings products have, and the known prices, over all 826,108 rows. Long dense titles, a lopsided rating pile, prices mostly under $40._

(inline screenshot: see the html deck)

_Field coverage per rating bucket: the chart behind the quick popular-first index (the table on the previous slide, drawn)._

### scripts/
* `evaluate.py` + `eval_queries.json`: 152 queries across sixteen classes (the original 20 conversational + 8 brand, then two added rounds: outfits, audiences, budget shapes, negations, materials, styles, non-English, misspellings, very short and rambling asks, fit needs, gifts, accessories, look-alike traps), hand written slot rules, up to 10 configurations, 3 indexes, paired plans, bootstrap intervals. `eval_report.py` makes the tables.
* `benchmark.py`: RSS after load, retrieval p50/p95 at concurrency 1, 2, 4.
* `make_fixture.py`: the 486 row sample that ships in the repo.

### Prompt experiments, in the order they happened
* Listing-style search queries instead of shortened sentences: type-match 0.73 to 0.88 on the same retrieval.
* Keywords restricted to product type synonyms; adjectives and colours were boosting the wrong titles.
* One rerank call per slot, in parallel: 20 s became 3 to 7 s.
* "Type before price" as a ranked list of criteria, after a priced wooden ring beat an unpriced blazer.
* Exclude words per slot (swimsuit: "cover up"), then a correction: never exclude words that correct listings commonly contain, and a penalty equal to the boost so ambiguous titles net zero.
* Budget splits floored at 10% per slot after a run gave the blazer $10 of $200.
* Reasons capped at 15 words, one 20 word note per slot: half the output tokens.
* Candidate data labelled as untrusted catalog text; every returned id validated.

Dead ends: a hand built product taxonomy (too much to maintain for the gain), baking the variant key into the index (one regex fix would cost a 20 minute rebuild), strict prices everywhere. Full notes in `docs/exploration.md`.

---

## Architecture and data flow

_5 / 19 · Architecture, one request animated_

In one line: client -> API -> planner -> retriever (reading the index files) -> reranker -> selector -> response; the offline lane at the bottom builds the index the retriever reads. The animation below walks one request through it.     clientReact page, curl, any HTTP client FastAPI · POST /recommendvalidate the request, request id, one 40 s deadline, per stage timings  1. plannerstructured plan: slots, audience, budget, keywords, exclude words 2. retrieverone matrix multiply for all slots, bm25, filters before top-N, RRF, grouping 3. rerankerone LLM call per slot in parallel, picks with reasons, ids validated 4. selectork per slot, a product fills one slot, warnings   LLM providerAnthropic, OpenAI or Bedrock, structured output, typed errors embeddings.npy826,108 x 384, bge-small, L2 normalised bm25 indexsame rows, same text catalog.parquetthe indexed rows: title, price, rating, image   raw jsonl.gz826,108 rows, 224 MB ingestprice, audience, doc text, 30 s catalog (all rows)typed parquet, 150 MB build-indexembed all rows, 18.4 min, bm25 data/index/row ids, checksums, meta     step 0 / 7

Every edge is labelled with what travels on it and the arrow gives the direction; the black dashed edges are the ones active in the current step, the moving boxes are the payloads. Arrow keys left/right step through. Static version: `docs/architecture.pdf`; the written walk-through with the questions each step raises: `docs/architecture.md`. Timing numbers are from real laptop runs (M4) with claude-sonnet-4-6; the index numbers describe the full 826,108-row build the deployed service serves.

---

## Design decisions and their rationale

_6 / 19 · Design decisions_

One line each here; each has a full record in `docs/adr/` (context, decision, why, consequences, alternatives).

| decision | logic | evidence | rejected |
|---|---|---|---|
| local embeddings, bge-small (ADR-001) | free, offline, reproducible, pinned revision; good enough on conversational queries | best lists of 3 models on 8 queries; 761 docs/s; 100K in 2.6 min, all 826K in 18.4 | hosted embeddings, bge-base |
| hybrid dense + bm25, RRF (ADR-002) | dense reads sentences, bm25 reads brands; RRF merges the two ranked lists by position, so their scores never need calibrating | brand hit 31/32 hybrid vs 25/32 dense; within 2 points on planner queries | dense only, weighted sums |
| masks before top-N (ADR-003) | post-filtering empties slots when eligible items sit below rank N; a mask is microseconds | 0 empty slots in every run | over-fetch then filter |
| exact search in RAM (ADR-004) | 826K rows fit; one matmul per request; 100% recall keeps masks and eval simple | p50 22 ms at 100K (cpu), 110 ms at 826K | FAISS / vector db |
| popular-first 100K quick build (ADR-005) | builds in 2.6 min; popular rows carry 5x the price coverage; bias stated per response | 0.885 popular vs 0.935 full vs 0.885 random on the same plans; the subset loses brands and the tail, not type, wich is why production serves the full catalog | full for every build, random |
| LLM planner, structured output (ADR-006) | the only way to split "outfit" into types without a taxonomy; schema + normaliser remove trust in free text | 0.73 to 0.88 type-match from the query rewrite alone | rules, taxonomy classifier |
| LLM rerank, one call per slot (ADR-007) | reads constraints a cross-encoder (a small model that scores query-item pairs) can't; output tokens dominate so parallel per slot | 20 s to 3 to 7 s for five slots; about a point of type-match on paired plans, the value is constraints and reasons | single call, cross-encoder, none |
| strict explicit prices, flagged inferred (ADR-008) | explicit bounds are promises, sentence budgets are hints; 94% unpriced made one rule wrong | the wooden ring in the blazer slot, then Eddie Bauer blazer flagged | always strict, always relaxed |
| variant grouping at query time (ADR-009) | nothing deleted, identity is a heuristic, a fix must not cost a rebuild | 56,720 duplicate titles; "(8 B(M) US, Silver)" cases | dedupe at ingest |
| three providers, one method, no framework (ADR-010) | 80 lines per adapter (Anthropic, OpenAI, Bedrock), contract tests, nothing to learn | all three adapters live tested, all error classes mapped | LangChain / LlamaIndex |
| one deadline, typed failures (ADR-011) | degrade and say so; keep the concurrency permit until the thread really ends | 504 test, stuck planner test, permit test | per call timeouts only |
| self contained index, checksums, baked image (ADR-012) | row misalignment is the worst retrieval bug; a deploy needs no volume | tamper tests for every artifact; Railway build bakes 40K rows | separate artifact store |
| FastAPI + React page + CLI, docker on Railway (ADR-013) | typed request and response models give validation and OpenAPI for free; one committed page demos it; the image bakes a demo index | /docs, /health, /ready; the docker smoke test in CI | flask, streamlit, a separate frontend service |
| eval by hand written type rules (ADR-014) | no labels exist; type is the failure that matters; ablations show what each part buys | 10 configs x 3 indexes, brand check, later a 78-query extended set | LLM-as-judge only |
| type gate and brand pass (ADR-015) | product type is the third hard constraint of a slot; for LLM plans the keywords are curated synonyms, so once k title matches exist the rest is dropped (head nouns count, accessory words veto); a named brand is ranked first and degrades to a preference with a warning | "running shoes for flat feet" was all insoles before; match@k 0.83 to 0.89 on 28 queries; brand queries from 1 of 4 to 3 or 4 of 4 on-brand where the catalog has them | a bigger keyword boost, brand as a keyword, a hand built taxonomy |
| request limits, fail-closed loading (ADR-016) | one request can cost six LLM calls and the index comes from a url: per-client token bucket, in-flight cap, body cap, proxy-aware client ip, security headers; https public hosts only for the index, archive caps, the real loader validates an install, strict model revision | 60 req/s per process measured; a burst of 8 LLM requests served without errors; every limit has a test that trips it | edge-only limits, an api key on /recommend (one middleware away) |

---

## Technology stack

_7 / 19 · Technology stack_

| Python 3.12, FastAPI, pydantic | typed request and response models, an automatic API reference at `/docs` (OpenAPI) for free, async pipeline, `/health` and `/ready` |
|---|---|
| sentence-transformers, bge-small-en-v1.5 | 384-number embeddings (text turned into vectors so similar meanings sit close), local on cpu or gpu, same model offline and online, no per-query cost |
| bm25s, numpy, pandas, pyarrow | the lexical channel, exact cosine with one matrix multiply, boolean masks, a typed catalog on disk |
| Anthropic, OpenAI and Bedrock SDKs | structured output on all three behind `complete_json(system, user, schema)`; provider by env, `none` runs keyless |
| React + Vite, plain CSS | one page with product cards; built bundle committed and served by the API |
| uv, pytest, ruff, Docker, GitHub Actions, Railway | 421 tests, ruff, bandit, pip-audit, a ui drift check, a cpu-only image that bakes a demo index, `/ready` as health check |

Why not a vector database or an orchestration framework: at this size they add operations without adding correctness, and a reviewer has to install them. Both are listed as the next step past a few million rows.

---

## Features and capabilities
* Outfit decomposition: "what to wear to an outdoor wedding" becomes shirt, pants, blazer, shoes, tie. A single item request stays one slot.
* Constraints from the sentence: audience ("my husband", "my 6 year old daughter"), occasion, season, per item or total budget. A total is split into per-slot shares (10% floor, parts never above the total); final picks are not re-optimised against it, and the answer warns when the priced top picks add up over it.
* A one sentence reason per pick, the evidence fields it used, the title keywords it matched.
* `price_known` on every item; strict when you give a bound, flagged when i inferred one.
* Look-alike control: planner exclude words push cover-ups out of the swimsuit slot; the reranker is told which candidates look off-type.
* Degrades instead of failing: no key, a timeout, a rate limit, a refusal, bad json; the regex planner and retrieval order take over and `warnings` says so.
* Same request model on three surfaces: the endpoint, `stylist recommend "..."`, the page. OpenAPI contract with examples.
* Per stage timings, which planner and which index answered, a request id in the logs, a 504 when retrieval itself blows the deadline.
* Index checksums and row alignment verified at startup; a prebuilt index can be installed from a url with sha256, a size cap and safe extraction.

_8 / 19 · Capabilities_

---

## Sample results

_9 / 19 · Sample results_

Recorded on the quick 100K index with claude-sonnet-4-6, the local baseline (production now serves the full catalog with faster models). Output trimmed to fit. The chirpy "note" at the end of the first sample is the model's own shopper-facing copy, shown verbatim; the per-item reasons above it are the evidence-checked part.

```
query: I need an outfit to go to the beach this summer
plan (llm): Outfit for a beach day this summer

[swimsuit]  search: women's swimsuit one piece or bikini summer beach
  1. SheIn Women's One Piece Swimsuit Sleeveless Asymmetrical Bikini Cut Out Monokini Orange La
     price n/a | 4.6 stars (26) | https://www.amazon.com/dp/B08R5YVDB9
     High rating 4.6, women's one-piece monokini, beach summer style
  2. Dokotoo Womens Ladies Summer Beach Stripes Color Block One Piece Bathing Suit Swimsuit Mon
     price n/a | 4.3 stars (36) | https://www.amazon.com/dp/B01N5IB4AO
     Good rating 4.3, beach stripes color block one-piece, summer vibe

[cover-up]  search: women's beach cover up dress summer lightweight
  1. Imagine Women's Summer Dress Strapless Floral Print Bohemian Casual Beach Dress Cover Ups
     price n/a | 4.1 stars (3,453) | https://www.amazon.com/dp/B07R1XGJZ2
     High rating 4.1, 3453 ratings, floral boho beach style, women
  2. Yonala Women's Summer Beach Wear Bikini Swimsuit Cover Up Swimwear Beach Dress,White, One
     price n/a | 4.1 stars (24) | https://www.amazon.com/dp/B0796L86KM
     Rating 4.1, classic white beach cover-up, women's summer

[sandals]  search: women's beach sandals summer flat
  1. Mu Dan Women's Thong Flat Gladiator Summer Sandals
     price n/a | 4.0 stars (79) | https://www.amazon.com/dp/B01ESNNRZE
     Flat gladiator sandal, women's, beach-ready, highest rating count
  2. Amlaiworld Women Walking Sandals Summer Bohemia Sweet Beaded Sandals Comfortable Flat Beac
     price n/a | 4.0 stars (35) | https://www.amazon.com/dp/B07QKFHWWB
     Flat beach water sandal, bohemian summer style, solid rating

[sun hat]  search: women's sun hat beach summer wide brim
  1. Sun Hat for Women Wide Brim Hat Sun Visor for Women UV Protection Summer Beach Fishing Hat
     price n/a | 4.5 stars (22) | https://www.amazon.com/dp/B08FGPGK82
     High rating, UV protection, wide brim, perfect beach summer style
  2. Women's Large Wide Brim Floppy Straw Hat Summer Beach Sun Hat w/ Bow Ribbon
     price n/a | 3.4 stars (84) | https://www.amazon.com/dp/B019WB49KY
     Classic straw floppy hat, great beach summer look, variety in style

[sunglasses]  search: women's sunglasses beach summer UV protection
  1. UV-BANS Polarized Aviator Sunglasses for Women Uv Protection, Round Sunglasses, Oversized
     price n/a | 4.3 stars (60) | https://www.amazon.com/dp/B07CSQ5FVJ
     Polarized, women's, multiple beach-ready styles, higher rating
  2. LianSan Fashion Sunglasses for women oversized Uv400 Protection Women's Sunglasses 13038 (
     price n/a | 3.9 stars (27) | https://www.amazon.com/dp/B00K3JSCB2
     Oversized UV400, women's, fun purple color for beach

note: Here are two lovely cover-ups to keep you stylish and sun-protected at the beach! Stay shaded in style with a UV-protective wide brim or a classic straw floppy hat! These stylish one-pieces are perfect for a fun summer beach day! These stylish shades will keep your eyes protected and beach-ready all summer! These cute flat sandals are perfect for strolling the beach this summer!

timings: {'plan_ms': 6698.5, 'retrieve_ms': 234.9, 'rerank_ms': 4623.6, 'total_ms': 11557.3}  (planner=llm, rerank=True, index rows=100000)
```

```
POST /recommend {"query": "warm waterproof boots for hiking in the snow", "k": 2, "max_price": 80}

{
 "request_id": "40e5b9943aa4",
 "query": "warm waterproof boots for hiking in the snow",
 "plan": {
  "intent": "Find warm waterproof boots suitable for hiking in snowy conditions",
  "audience": null,
  "season": "winter",
  "budget_max": null,
  "budget_scope": "unknown",
  "brand": null,
  "source": "llm",
  "slots": [
   {
    "name": "hiking boots",
    "search_query": "waterproof insulated snow hiking boots",
    "keywords": [
     "hiking boots",
     "snow boots",
     "winter boots",
     "trekking boots"
    ],
    "exclude_keywords": [
     "rain boot",
     "wellington",
     "casual"
    ],
    "budget_max": null
   }
  ]
 },
 "slots": [
  {
   "name": "hiking boots",
   "search_query": "waterproof insulated snow hiking boots",
   "keywords": [
    "hiking boots",
    "snow boots",
    "winter boots",
    "trekking boots"
   ],
   "budget_max": 80.0,
   "n_eligible": 10,
   "eligible_rows": 13181,
   "items": [
    {
     "rank": 1,
     "title": "Karrimor Hot Rock Mens Walking Boots Waterproof Lace Up",
     "price": 73.99,
     "price_known": true,
     "average_rating": 4.3,
     "rating_number": 42,
     "audience": "men",
     "url": "https://www.amazon.com/dp/B01MG8KYVS",
     "matched_keywords": [],
     "reason": "Waterproof hiking boot, within budget, good rating",
     "evidence": [
      "title",
      "price",
      "rating",
      "features"
     ]
    },
    {
     "rank": 2,
     "title": "Nevados Men's Klondike Mid Waterproof Hiking Boot | Lightweight for Trail, Walki",
     "price": 75.0,
     "price_known": true,
     "average_rating": 4.2,
     "rating_number": 418,
     "audience": "men",
     "url": "https://www.amazon.com/dp/B003VWBN46",
     "matched_keywords": [
      "hiking boots"
     ],
     "reason": "Waterproof hiking boot, memory foam, within budget",
     "evidence": [
      "title",
      "price",
      "rating",
      "features"
     ]
    }
   ]
  }
 ],
 "warnings": [],
 "llm_info": {
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "planner_used": "llm",
  "rerank_used": true,
  "rerank_model": null,
  "plan_cache_hit": false,
  "calls": 2,
  "failed_calls": 0,
  "input_tokens": 4087,
  "output_tokens": 236
 },
 "timings": {
  "plan_ms": 3577.3,
  "retrieve_ms": 89.1,
  "rerank_ms": 3497.3,
  "total_ms": 7163.9
 }
}
```

"what should my husband wear to an outdoor wedding in june, budget 200 total". Five slots, audience men, the planner split the total into shirt $38, pants $45, blazer $52, shoes $45, tie $20.
* V1 filtered every slot to priced items. The blazer slot: a $13 wooden ring and linen shorts. Only about 14K of 100K items carry a price; almost no blazer among them.
* That run changed the price policy (ADR-008) and the rerank prompt.
* Now: an Eddie Bauer travel blazer + oxford dress shoes, `price_known: false`, next to priced shirts, pants and a tie set inside their shares.

Honest caveat: planner output varies between runs (slot names and counts change), so eval differences under about 2 points are noise.

(inline screenshot: see the html deck)

---

## Evaluation and quality

_10 / 19 · Evaluation_

### The original 28 queries (20 conversational, 8 brand), scored at k=4

| configuration | popular 100K | random 100K | full 826K |
|---|---|---|---|
| keyword search only (bm25), no planner | 0.500 | 0.562 | 0.652 |
| meaning search only (embeddings), no planner | 0.696 | 0.696 | 0.786 |
| both channels fused, no planner | 0.625 | 0.679 | 0.750 |
| both channels + LLM planner | 0.881 | 0.881 | 0.935 |
| meaning only + LLM planner | 0.862 | 0.877 | 0.923 |
| full pipeline (+ LLM rerank) | 0.885 | 0.885 | 0.935 |

### What the number is, and how to read the table
* match@4 checks the top 4 items each slot returns: it is the share of those returned items whose title passes a hand-written rule for the slot (a sandals slot accepts sandal / flip flop / slide; a brand query's rule also demands the brand). Whole words, plural tolerant. The table pools every returned item (the micro average); a slot that returns fewer than 4 shows up in the empty-slot and success columns, not here. 0.885 means 89 of every 100 returned items passed their slot's rule.
* Why the rules can be trusted: the original 28 rules were written before i looked at any output, so they could not be tuned afterwards to flatter the results. Every planner configuration replays the same recorded plans on every index; the no-planner rows search the raw sentence, wich is exactly the difference being measured. So within a row, only the index changes; between a planner row and a no-planner row, only the planner does. A slot the planner invents that has no rule of its own is scored against all of the query's rules combined, so inventing slots cannot inflate the score.
* The intervals say what is real: the macro average (per query, so long outfits do not dominate) for the full pipeline is 0.857 (95% CI 0.76 to 0.94) popular and 0.967 (0.94 to 0.99) full; the wide spread is variation across queries, not doubt about a comparison. Comparisons use paired per-query differences, and there a couple of points sits inside the noise band while the planner's 25-point jump and the full catalog's 5 points do not.
* Read down a column: the planner is the feature. Read across a row: the full catalog beats both subsets. Zero empty slots and zero price violations in every run. This measures the right KIND of product, never taste; the rest of the caveats live in `docs/evaluation.md`.

### The wider net: 152 queries, measured on the live service

28 queries can catch a big regression but cannot separate close configurations, so the set grew in two rounds to 152 across sixteen classes: outfits, audiences, budget shapes, negations, materials, styles, non-English, misspellings, very short and rambling asks, fit needs, gifts, accessories, and look-alike traps (queries where a similar word tempts the system into the wrong product, insoles for running shoes). The original 28 stayed frozen; the added rules were drafted the same blind way, against how listing titles are written, but after the system was built: a challenge set, not pre-registration. Protocol: every query is POSTed once cold (starting its background plan), then once more after the plans land; the second pass is scored. All 152 plans landed (the model returned a usable plan every time).

| the run (live, full catalog, Nova Lite plans) | result |
|---|---|
| macro match@4 (average per query), with its 95% interval | 0.786 (0.74 to 0.83) |
| micro match@4 (all returned items pooled) | 0.687 |
| query success (every expected slot filled with a passing item) | 0.645 |
| empty slots / plans landed | 0 / 152 of 152 |

| query class (macro match@4 within the run, class size) | score |
|---|---|
| misspellings, very short asks, gifts, accessories (n=14 combined) | 1.00 |
| look-alike traps (n=7), materials (n=8) | 0.91 to 0.96 |
| the original 28 (standalone run: 0.872 macro, 0.819 micro, slide 15) | 0.866 |
| budgets (n=12), brands (n=14), fit needs and styles (n=12) | 0.80 to 0.85 |
| audiences (n=11), negations (n=8), rambling asks (n=4) | 0.69 to 0.77 |
| loose conversational asks (n=13), non-English (n=8) | 0.58 |
| open-ended outfits (n=13) | 0.52 |

What the size buys: 152 queries put a five-point interval on the headline instead of the ten-point one 28 gave, tight enough to trust the differences this table shows. The perfect scores on misspellings and two-word asks say the meaning-based channel absorbs sloppy input; the traps at 0.96 say the exclude rules earn their keep. The weak classes all lean on plan quality, splitting an open request into the right pieces, and the Sonnet-vs-Lite gap on the original set points the same way: the headroom is planner model, not retrieval. Raw-sentence baselines on the same 152 (bm25 0.476 with three empty slots, meaning 0.610, fused 0.576) confirm the wider set is simply a harder exam. Raw results: `docs/eval_full_extended_retrieval.json`, `deploy/aws/experiments/exp27` (the earlier 78-query round: exp26).

### What the numbers say
* The planner is the feature: raw sentences get 0.63 to 0.75 on hybrid retrieval; the planner's listing-style queries lift every index to 0.88 to 0.94 and give the outfit queries their slots (slot recall 0.88: it produced 88% of the expected outfit pieces).
* Dense beats bm25 on sentences by 12 points; under the planner the channels are within 2 points and hybrid stays because bm25 carries the brand token.
* The reranker adds about a point of type-match (paired mean +0.009); its value is reading the constraints, writing the reasons, and keeping wrong-type items out of a slot.
* The full catalog beats the subsets on the same plans: all 8 brand queries fully on-brand there, where the popular subset has no Levi's jeans and three Columbia fleeces for a rain jacket.

### Cost and latency

| index | rows | build (laptop gpu) | memory to serve | retrieval p50 |
|---|---|---|---|---|
| popular (quick build) | 100,000 | 2.6 min | 1.0 GB | 22 ms (cpu) |
| full (what production serves) | 826,108 | 18.4 min | 3.3 GB | 110 ms |

Full pipeline p50 11.5 s cold (plan 5.2 s, rerank 5.9 s in parallel across slots), 5.5 s with a warm plan cache; claude-sonnet-4-6 through an Azure endpoint, laptop numbers. Slide 11 has the throughput and cost measurements.

### Tests: 421
* Unit: parsers, plan normalisation, masks, fusion, grouping, budgets, type gate, brand pass.
* Failure paths: every LLM error class, a failing slot, stuck planner, waiter timeout, retrieval past deadline.
* HTTP: validation, rate limit, in-flight cap, body cap, security headers, error bodies, startup failures.
* Artifacts: archive bombs, private hosts, stale indexes, locks. Review rounds folded into regression tests.
* CI: ruff, bandit, pip-audit, docker smoke test, UI drift check.

---

## Latency, throughput and cost, measured

_11 / 19 · Production numbers_

### One process, retrieval only (use_llm=false), 100K index, CPU

| concurrency | p50 | p95 | req/s |
|---|---|---|---|
| 1 | 22 ms | 32 ms | 42 |
| 2 | 33 ms | 41 ms | 58 |
| 4 | 65 ms | 82 ms | 62 |
| 8 | 126 ms | 145 ms | 62 |

60 requests per setting, M4 laptop pinned to CPU (`scripts/benchmark.py --device cpu`, `docs/bench_cpu.json`). Throughput flattens at about 60 req/s per proces because the embedding call and the matrix multiply hold the GIL (python's one-thread-at-a-time lock); more capacity means more processes, not more threads. Memory after load (RSS): 1.0 GB (torch + model about 0.5 GB, index 0.25 GB, pandas catalog and bm25 the rest). Load time 0.1 s including sha256 of every file.

### Full pipeline with the LLM, a 20 query live run (recorded before the brand queries were added), claude-sonnet-4-6

| stage | p50 | p95 | share |
|---|---|---|---|
| plan (1 LLM call) | 5.2 s | 7.1 s | 45% |
| retrieve (all slots) | 0.13 s | 0.28 s | 1% |
| rerank (1 call per slot, parallel) | 5.9 s | 6.9 s | 51% |
| total | 11.5 s | 13.5 s |  |

Sequential run, cold plan cache, 2.9 slots and 3.9 LLM calls per request on average, 0 empty slots in 58, 12 warnings (mostly "3 of 4 items chosen by the reranker"). A burst of 8 concurrent requests finished in 18.5 s with no errors (p50 8.7 s, max 18.5 s) under LLM_CONCURRENCY=8. Source: `docs/live_run_sonnet.json`.

### Economics

| per request | mean | min | max |
|---|---|---|---|
| input tokens | 8,200 | 3,900 | 13,100 |
| output tokens | 760 | 250 | 1,380 |
| LLM cost at $3 / $15 per M (Sonnet class, the larger model tier) | $0.036 | $0.016 | $0.060 |
| same traffic on a $1 / $5 per M model (Haiku class, the smaller tier) | $0.012 | $0.005 | $0.020 |
| compute (a demo-size 2 vCPU container, $20 / month, 40K req/day) | $0.00002 |  |  |

Token counts come strait from the SDK usage fields and are returned in every response (`llm_info.calls / input_tokens / output_tokens`), so cost per request is a number in the logs, not an estimate. Roughly 2,100 input and 195 output tokens per call: the system prompts and the 10 candidate rows dominate.

### At 10,000 requests a day
* LLM: about $360 / day on the Sonnet class as measured. The plan cache removes 1 of 3.9 calls, so a 90% hit rate plus prompt caching on the system prompts (marked cacheable in the adapter) lands near $200; a Haiku-class model on the rerank calls too takes it under $100.
* Compute: one 2 vCPU / 4 GB container does about 0.43 req/s with the LLM (measured: 8 concurrent requests in 18.5 s), call it 35K a day per replica, or several million retrieval-only; the provider's rate limit is the ceiling, not the CPU.
* Ways to cut the bill, in order of payoff: cache plans across replicas (Redis), rerank with a smaller model (the paired eval says the reranker adds about a point of type-match, so check the constraints and reasons survive), cut candidate rows to 8, shorten system prompts, one rerank call for all slots when slots <= 2.

---

## Recommended production setup

_12 / 19 · Production deployment_

### Shape

| piece | recommendation | why |
|---|---|---|
| runtime | the existing container (python:3.12-slim, CPU torch); 8 GB per replica serving the full 826K index (the deployed shape, 3.3 GB in use per worker), 4 GB is enough for the quick 100K build | memory measured: 1.0 GB for 100K rows, 3.3 GB for 826K; query embedding needs 5 to 10 ms on CPU, a GPU buys nothing at this size |
| processes | 1 uvicorn worker per replica, 3 replicas minimum across zones, autoscale on in-flight requests (target 6 per replica) | the process is stateless; one worker keeps LLM_CONCURRENCY and the rate limiter meaningful per replica |
| index artifact | tarball in object storage (S3 / GCS), pinned by INDEX_URL + INDEX_SHA256, rebuilt by a scheduled job when the catalog changes, rolled out by changing two variables | already implemented: verified download, safe extraction, atomic install, refuses to serve a mismatched index |
| platform | Railway or Fly.io for a team of one; ECS Fargate, Cloud Run or Kubernetes when it joins a platform that already runs one of them | no database, no queue, no GPU, any container host with health checks works |
| LLM | Anthropic Claude (Sonnet class for plan + rerank, Haiku class as the cost option), prompt caching on, a second provider configured as failover | the adapters already hide the SDK; ANTHROPIC / OPENAI switch by environment |
| shared state | Redis for the plan cache (key: normalised query + model + prompt version, TTL 24 h) and the rate limiter | today both are per process; with 3 replicas the hit rate drops and limits triple |
| edge | TLS, CDN for `/assets`, WAF rate rules, request size limit at the edge too | the in-app limits are the second line, not the first |
| secrets | platform secret store or Vault, rotated keys, no key in images or logs | the settings repr hides keys; error messages are scrubbed; LOG_QUERIES is off by default |

### Sizing rule of thumb
* Retrieval-only traffic: 60 req/s per vCPU pair, scale linearly with replicas.
* LLM traffic: 0.43 req/s per replica observed at LLM_CONCURRENCY=8 (the concurrency formula's 0.7 is an upper bound the burst never reached); the provider's tokens-per-minute (TPM) quota usually binds first (8,200 input tokens x 0.7 req/s = 350K TPM at the ceiling).
* Memory: index bytes x 2 (float16 on disk, float32 in memory) + 600 MB fixed.
* Startup: 10 to 20 s incl. the index download; point the readiness probe at `/ready` with a 60 s initial delay, liveness at `/health`.

### Rollout
* Build: CI produces the image and the index tarball as versioned artifacts; the index is rebuilt only when the catalog or PIPELINE_VERSION changes.
* Deploy: rolling replicas behind readiness; an index change is a config change (two variables), reversible in one step.
* Canary: 5% of traffic on the new image or prompt version for an hour, compared on fallback rate, p95, tokens per request and the offline eval run against the same index.

### Configuration that matters in production

This is the generic recommendation; the deployed fast profile on slide 14 overrides some of it (PLANNER_BUDGET_S=0.10, RERANK_DEFAULT=0, the caches on).

REQUEST_DEADLINE_S=40 PLANNER_BUDGET_S=15 RERANK_BUDGET_S=20 LLM_CONCURRENCY=8 MAX_INFLIGHT_REQUESTS=16 RATE_LIMIT_PER_MINUTE=60 MAX_BODY_BYTES=16384 CORS_ALLOW_ORIGINS=https://app.example STARTUP_FAIL_FAST=1 LOG_QUERIES=0 INDEX_URL=... INDEX_SHA256=...

---

## Getting the total under one second

_13 / 19 · Under one second_

97% of the 11.5 s is two LLM calls that run one after the other. Tuning them wont reach one second, taking them out of the synchronous path does. Each step below is independent and measurable on the eval set before it ships.

| step | what changes | plan | rerank |
|---|---|---|---|
| 1. plan cache + prompt caching (in the code) | repeat queries cost nothing to plan; misses still pay | 0 ms / 5 s | 5.9 s |
| 2. semantic plan cache | embed the query (5 ms, model already loaded), reuse the nearest plan at cosine >= 0.92; real traffic repeats itself heavily, so 60 to 80% should hit | 5 ms / 5 s | 5.9 s |
| 3. distilled planner | a small local model trained on the current planner's own outputs (Qwen2.5-1.5B or Llama-3.2-3B on 10 to 20K request-plan pairs), served on one modest GPU with vLLM; 1 to 2 s on CPU with a compressed build | 150 to 300 ms | 5.9 s |
| 4. cross-encoder reranker | bge-reranker-v2-m3 or MiniLM-L6 over 50 (query, title + attributes) pairs; constraints stay in the filters; reasons from the deterministic template or streamed to the page after the response (SSE) | 150 to 300 ms | 60 to 120 ms |
| 5. offline enrichment | LLM once per catalog row at index time (type, occasion, season, style, audience); $100 to $800 one-off for 826K rows; the type gate becomes exact | same | same, better |
| 6. retrieval | pre-tokenised bm25, int8 embeddings (smaller numbers, less memory), approximate nearest-neighbour search (HNSW) only past a million rows | 130 to 40 ms |  |

After step 4: plan 10 to 300 ms + retrieval 40 to 130 ms + rerank 60 to 120 ms + select 1 ms = p50 about 250 to 600 ms, p95 under a second, about $0.001 per request with the GPU amortised. Steps 1, 2 and the cache half of 6 are no longer a plan: the next two slides show them deployed on AWS, now serving the full catalog, and measured.

### What it costs in quality, and how to keep it
* The large model's judgement on unusual requests and its written reasons go out of the hot path. Keep that path as a background refinement ("better picks in 8 s", pushed over SSE) and as the generator of the distillation data.
* Gate every step on the eval set: the distilled planner ships when slot recall and mapped precision are within 3 points of Sonnet's; the cross-encoder when match@k on the eval queries is within 2 points of the LLM reranker.
* Streaming alone (per-slot SSE) puts the first products on screen in about 200 ms with no ranking change; its the cheapest percieved-latency win and needs only the UI and one endpoint.

### Order of work
1. SSE partial results and the semantic plan cache (2 days, no new model).
1. Cross-encoder reranker behind a flag, compared on the eval set (3 days).
1. Distillation set from logged plans, fine-tune, vLLM deployment (1 to 2 weeks).
1. Offline enrichment as a catalog build step (1 week, mostly pipeline work).

---

## The AWS build: CloudFront to Tokyo to Bedrock

_14 / 19 · AWS deployment_

The `AWS_deployment` branch is steps 1, 2 and half of 6, deployed and measured, serving the full 826,108-row catalog. One request path: CloudFront at the edge, an EC2 origin in Tokyo, Bedrock in the same region for the planner. No docker on the box; systemd runs uvicorn straight from a source tarball out of S3.

| piece | what runs | why it is that |
|---|---|---|
| edge | CloudFront, HTTP/3 on, `/assets` cached, everything else passes through | TLS terminates near the user; port 8000 on the origin admits CloudFront address ranges and nothing else |
| origin | c7i.xlarge in ap-northeast-1a, one uvicorn worker under systemd, elastic IP | the demo user is in Japan (about 200 ms of the old number was the Pacific); one worker because the full index is 3.3 GB a process and two copies would not fit in 8 GB |
| planner | Bedrock, `apac.amazon.nova-lite-v1:0`, structured output through one forced tool | in-region call, and no key on the box: the instance role is the credential |
| bounded wait | PLANNER_BUDGET_S=0.10, call timeout 20 s, at most 8 background plans at once | measured first: 0 of 12 plans landed even inside 350 ms, so any wait is pure cost; 100 ms keeps the worst case tiny while still catching an instant cache hit |
| caches | response cache (300 s, warning-free answers only), semantic plan cache (cosine 0.92), slot-embedding LRU | a repeat costs sub-ms server time; a paraphrase reuses the nearest plan with zero LLM calls |
| warmup | two passes over the six UI examples on every restart | pass one starts the background plans, pass two caches the planned answers |

### How the fast profile answers
1. A request checks the exact plan cache, then the semantic one.
1. On a miss it waits up to 100 ms for the planner, then answers with the regex plan.
1. The Bedrock call keeps running (about 1 s) and lands in both plan caches.
1. Every later request, paraphrases included, gets the LLM plan.

### What a review round changed before launch
* Background plans only fed the exact-string cache, so paraphrases re-paid Bedrock forever. Found in review, fixed, proven live.
* The response cache could freeze a fallback answer for its whole TTL. Now only warning-free responses are cached, and a hit says `served_from_cache: true` with its real serve time.
* The Tokyo move went first and code tuning after, so every knob was set against numbers it would actualy see.

Deployment is five short scripts plus a teardown (`deploy/aws/`), region generic: the same scripts built us-east-1 first and Tokyo after it. Every experiment is a file, exp01 to exp26, in `deploy/aws/experiments/`.

---

## Measured from Japan: 13 ms warm, 464 ms cold, all 826,108 rows

_15 / 19 · AWS results_

Two measurement rounds. First the move and the fast profile, on the quick 100K index:

| path (100K index) | us-east-1 | Tokyo |
|---|---|---|
| steady keep-alive p50 | 344 ms | 57 ms |
| warm repeat (response cache) | n/a | 13 to 44 ms |
| cold unique query | ~640 ms | 210 ms |
| paraphrase via semantic cache | 215 ms | 116 ms |

Then the switch to the full catalog (8.3x the rows, one worker), re-measured end to end:

| path (full 826K, serving now) | measured |
|---|---|
| steady keep-alive p50 (n=100) | 13.0 ms |
| fresh TLS connection per request (n=50) | 36.5 ms |
| response-cache hit | 15 ms |
| cold unique query (regex plan) | 464 ms |
| uncached answer on an LLM plan | 0.7 to 1.0 s |

Wall time from a Tokyo client through CloudFront. Cached paths did not move when the catalog grew 8.3x, because they never touch the big computation. What paid the bill: an uncached answer costs up to a second of server time, and unique-query capacity fell from 10-13 to about 3.8 requests a second per box. That number now sizes replicas.

### Load

Repeat-mode ramp on the full catalog: p50 of 36 to 48 ms from c=1 through c=8. The earlier 1-vs-2 worker comparison kept 2 for the better p95; the full index forced 1 (memory), wich also means the rate limiter and caches now behave exactly as configured, one bucket, one copy. The first ramp attempt measured the rate limiter instead of the service (a wall of 429s); benchmark past your own guardrails, then put them back.

### Quality, against the live endpoint (the original 28 queries)

| planner and index | match@4 micro | success | plans landed |
|---|---|---|---|
| Nova Micro, 100K | 0.705 | 0.39 | 27 / 28 |
| Nova Lite, 100K (0.800 after the round-three fix) | 0.772 | 0.50 | 28 / 28 |
| Nova Lite, full catalog (serving now) | 0.819 | 0.57 | 28 / 28 |
| Sonnet plans, local eval, full catalog | 0.935 |  |  |

The evaluation queries POSTed at production and scored by the same rules as the offline harness. Lite costs the user nothing extra because planning is background work by design. The remaining gap to Sonnet is planner model quality, not pipeline; the 152-query extended set on slide 10 shows the same pattern from a wider angle.

### Measured, then decided
* Planner completion on the box: Micro p50 999 ms, Lite 1154 ms, 0% inside 350 ms. Thats where the 100 ms wait comes from.
* The quality probe picked Lite over Micro: +6.7 points micro, +11 on success.
* The full-catalog switch was decided the same way (exp23 to exp26): quality up on every measure, cached latency unchanged, cold latency and capacity paid the bill, and the docs say so.

Live at `d3bys47v9rho9.cloudfront.net` while the demo runs. About $4 a day; one teardown script removes all of it.

---

## The acceptance battery: 11,500 requests, one bug found and fixed

_16 / 19 · AWS verification_

Latency probes prove the fast path. The battery (`deploy/aws/test_battery.py`, ten sections, results in exp20 to exp22; the full-catalog re-run is exp23 to exp26) asks the rest: does the contract hold, do guardrails fail the right way, what does a deploy cost, where does the machine saturate.

| section | result |
|---|---|
| contract: 20 checks, 422/413/405 + full schema | 20/20 |
| headers: CSP, nosniff, frame DENY, h3, edge-cached assets | all pass |
| guardrail: 70 rapid requests | first 429 at #65, throttles in 12 ms, clean recovery |
| steady keep-alive, n=100 | p50 12.8 ms, p99 16.9, max 17.5 |
| fresh TLS each request, n=50 | p50 34.9 ms |
| ramp: 21 cells, c=1..32, three workloads | 672 requests, zero errors |
| soak: 5 min at c=6 mixed | 5,010 requests, p95 flat 0.87 to 1.01 s, memory flat |
| restart under traffic | 6.7 s of 5xx, full recovery |
| cache ladder + consistency x5 | asserted, identical answers |

Capacity in that 100K round: repeat traffic 36 to 52 ms p50 up to c=32 (100 req/s peak); unique cold traffic saturated 4 vCPUs at 10 to 13 req/s, p50 2.7 s at c=32, zero 5xx. The full-catalog re-run moved the cold ceiling to ~3.8 req/s (slide 15), wich is now the replica-sizing number. The 11,500 total is the ten sections plus their ramp cells added up; SUMMARY.json itemizes.

### The bug the battery caught
* Symptom: 0.4 ms on one worker, 2.44 s on the other. Idle box.
* Trigger: a 5-slot plan with a $135 total budget. Tight price masks starve the type gate; the ranker widens x4 per pass; every pass hydrates every window row via per-row pandas iloc.
* 19,783 row hydrations per request; 10.4 s under 12-way load.
* The response cache refused to hide it, by design: degraded answers are never cached.

* Fix: one vectorized column gather per window for the scoring fields; the full row walk only for rows a slot returns.
* Proof: captured plan replays 2,831 ms before, 985 ms after; live steady p50 2,442 to 12.8 ms; a regression test counts hydrations.
* Quality unmoved: match@4 micro 0.800, success 0.50, 28/28 plans landed on the rerun.
* It never showed on us-east-1 (Micro rarely emits budgets) or the local eval (Sonnet plans). Only live traffic walked it. That is the argument for the battery.

Stated limits: a deploy on one origin costs ~7 s of 5xx (failover removes it); limits and caches are per process, so they behave exactly as configured with today's one worker and multiply the moment replicas exist; a single worker is also a single failure domain until the box grows.

---

## Guardrails, observability, fallback and monitoring

_17 / 19 · Guardrails, observability, reliability_

### Guardrails in place today
* Input: strict schema (unknown fields rejected, 1 to 500 chars, k 1 to 10, finite prices, min <= max), 16 KB body cap, 60 req/min per client, 16 in-flight requests per process.
* LLM: structured output only, every id validated against the offered candidates, picks capped at k, reasons capped at 15 words with urls and emails stripped, catalog text labelled untrusted in the prompt, the user sentence passed as data inside the json, no tool use.
* Budgets: one 40 s deadline, 15 s plan, 20 s rerank, global cap of 8 LLM calls in flight, retrieval pool bounded with deadline-aware queueing, typed LLM errors (auth, rate limit, timeout, refusal, truncation, validation, transport).
* Data: checksum-verified index, loader fails closed on any mismatch, safe tarball extraction (scheme allowlist, member caps, fixed permissions), keys never in repr / errors / logs, queries not logged unless LOG_QUERIES=1.

### Fallback ladder (each step is reported as a warning)

| failure | behaviour |
|---|---|
| LLM planner error or 15 s timeout | regex planner: budget, audience and content words; one slot; a 30 s negative cache stops a stampede |
| one rerank call fails or is late | that slot keeps retrieval order, the others keep their picks |
| reranker returns nothing usable | retrieval order plus deterministic reasons (keywords, rating, price vs budget) |
| a slot below k in the price window | flagged unpriced backfill when the bound was inferred, a warning when it was explicit |
| index missing or invalid at boot | /health stays up with a curated message, /ready and /recommend return 503, or the process exits with STARTUP_FAIL_FAST |
| no key at all | LLM_PROVIDER=none: the whole service runs as search |

### What to add before real traffic

| area | add | tool |
|---|---|---|
| logs | json lines with request_id, stage timings, tokens, fallback reasons; sampled query text behind a flag | structlog, shipped to Loki or CloudWatch |
| traces | one span per stage and per LLM call, propagated request id, provider latency separated from queueing | OpenTelemetry SDK + FastAPI and httpx instrumentation, Tempo or Honeycomb |
| metrics | histograms per stage, LLM errors by class, fallback rate, cache hit rate, tokens per request, in-flight and 429 counts | prometheus-client at /metrics, Grafana dashboards |
| alerts / SLOs | p95 total < 20 s, fallback rate < 10%, 5xx < 0.5%, LLM 429 burst, empty-slot rate, readiness flaps | Grafana alerting or PagerDuty |
| provider resilience | a circuit breaker: stop calling a provider that keeps failing (open after 5 failures in 30 s, plain-search mode meanwhile), one retry with jitter inside the budget, second provider failover | small in-process breaker; the other adapters exist already |
| content guardrails | off-domain and unsafe request filter before the planner (cheap classifier or moderation endpoint), profanity and PII scrub on reasons, an allowlist of evidence fields (exists) | provider moderation API or a Haiku-class classifier |
| quality monitoring | daily offline eval against the live index, sampled LLM-as-judge on production traffic, drift on empty-slot and unpriced rates | scripts/evaluate.py on a schedule, results to the metrics store |
| errors | exception tracking with request id and scrubbed context | Sentry |

---

## What is missing, and the plan to close it

_18 / 19 · Gaps and next actions_

### Production blockers, from the audit (the full plan with efforts is `TODO.md`)

| prio | gap | fix |
|---|---|---|
| P0 | no authentication: anyone with the URL can burn Bedrock tokens | API key verified at the edge (CloudFront function), per-key quotas; ALB auth later |
| P0 | single origin: a deploy costs a measured 6.7 s of 5xx, an instance loss is an outage | ALB + autoscaling group (min 2, two AZs) or an ECS service; rolling deploys take the window to zero and make the origin hop TLS |
| P0 | two deploy paths: the Dockerfile exists but AWS runs a bare systemd tarball | one artifact: CI builds the image, ECR, ECS Fargate runs it; trivy scan in CI |
| P0 | no alarms; logs die with the instance | JSON logs to CloudWatch, alarms on p95 / 5xx / fallback rate / Bedrock throttles / budget, one dashboard. The metrics already sit in the request log line |
| P1 | caches and the limiter live in one process: coherent with today's one worker, split the moment replicas exist | ElastiCache Redis for plan + response caches and the limiter, before any scale-out |
| P1 | one 8 GB box fits one full-index worker: a single failure domain and a ~3.8 req/s cold ceiling | a 16 GB instance restores two workers, or replicas behind the ALB from item 2 |
| P1 | hand-run deploys; static catalog | GitHub Actions to ECR to ECS with canary + auto-rollback; a scheduled index rebuild that flips INDEX_URL |
| P1 | provider outages retry every 30 s | a real circuit breaker (closed / open / half-open) around each provider |

### Product and ML gaps

| gap | why it matters | plan |
|---|---|---|
| relevance is judged by 28 hand written rules | type-match says nothing about fit, style or whether a human would buy it | 200 labelled queries (3 graders, majority vote), nDCG@4; an LLM judge calibrated against the labels, nightly in CI |
| plans vary across identical calls, even at temperature 0 | cold answers differ between requests; provider-side nondeterminism | plan pinning by query hash in the shared cache now; the distilled 1-3B planner (deterministic, ~200 ms) later |
| no online signal | offline metrics drift from what shoppers do | week 3: impression and click logging with request_id, a holdout, CTR and add-to-cart per configuration; A/B switch on prompt version and reranker model |
| full rerank is 1.4-1.8 s with Nova Micro on the box (5.9 s with Sonnet locally); results arrive all at once | good, not great; perceived latency is the page filling in | cross-encoder rerank on the origin's idle vCPUs (60-120 ms for 50 pairs) behind a flag; SSE per slot so retrieval order renders in ~200 ms |
| no outfit coherence | slots are ranked independently; a floral shirt and a striped blazer can both win | week 4-5: a coherence pass over the top 3 per slot (colour and style fields, one LLM call on the cross product), evaluated on the labelled set |
| catalog scale | exact scoring is fine to about 1M rows x 384 numbers; beyond that memory and p95 grow linearly | when needed: filtered approximate search (HNSW via faiss or usearch) with over-fetch and an exact fallback so the filter guarantee survives, int8 embeddings, the catalog served from parquet row groups instead of pandas |
| single-tenant, single-language | the prompts and the audience heuristic are English only | multilingual embedding model (bge-m3) and a language field in the plan; per-tenant index and prompt version |
| personalisation | size, past purchases and brand affinity are ignored | a user profile object merged into the plan the same way request fields are: explicit beats inferred |

### Tests still to run

| test | tool | pass condition |
|---|---|---|
| load and a short soak ran in round three (21 ramp cells to c=32, 5 min at c=6, zero errors). Still to run: 12 h at 20% capacity | k6 + Grafana | no memory growth, no thread pool leak, cache bounded |
| chaos: LLM returns 500 / garbage / hangs, index corrupted mid-run, replica killed during install | toxiproxy, a fake provider, kill -9 | every case ends in a warning or a clean 503, never a 500 or a wrong answer |
| prompt injection corpus (catalog text and queries) | a fixture of 100 hostile strings | no url, no instruction echo, ids still validated |
| property-based: planner normalisation and budget allocation | hypothesis | invariants hold for any input (sum <= total, floors, caps) |
| contract: recorded LLM responses replayed | cassette fixtures per prompt version | a prompt change that breaks parsing fails CI, not production |
| security: image and dependency scanning, a scan against the running API | trivy, pip-audit, ZAP baseline, bandit (already run) | no high findings |
| UI end to end | Playwright | query, results, warnings, error states render |
| multi-replica install race | 3 containers booting against one INDEX_URL | one download, all ready |

### Stack to add

OpenTelemetry + Prometheus + Grafana (or Datadog), Sentry, Redis, S3 for artifacts, k6, hypothesis, trivy and pip-audit in CI, Playwright, a nightly eval job. Nothing here changes the shape of the service code: the stages allready have boundaries, ids and timings to hang all of it on.

---

## Repository structure, documentation and deployment

_19 / 19 · Files and deployment_

```
src/stylist/
  catalog.py      raw jsonl to parquet
  embeddings.py   bge wrapper, hash embedder for tests
  index.py        build, load, checksums, scores
  planner.py      QueryPlan, LLM + regex planner
  retrieval.py    masks, rrf, boosts, grouping
  reranker.py     per-slot LLM rerank
  service.py      the pipeline, deadlines
  schemas.py  api.py  cli.py  artifacts.py
  llm/            protocol, three adapters, prompts
ui/               react page, dist/ committed
scripts/          evaluate, benchmark, fixture
notebooks/        data exploration, executed
docs/             prd, adr/ (16), design notes,
                  evaluation, exploration, diagram,
                  architecture.md walk-through
tests/            421 tests + 486 row fixture
deploy/aws/       five scripts + teardown,
                  experiments exp01 to exp26
Dockerfile  railway.toml  Makefile  ci.yml
```

### Documented

README for setup, sample usage and the design decisions. `docs/prd.md` for requirements and metrics, `docs/adr/` for the sixteen decision records, `docs/architecture.md` for the step-by-step walk-through with the questions each step raises, `docs/design-notes.md` for the narrative, `docs/exploration.md` for the notebook, scripts and prompt experiments, `docs/evaluation.md` for the numbers, `docs/architecture.pdf` for the diagram. `/docs` on the running service is the OpenAPI contract.

### Served
* `make demo`: fixture index in a minute, no download, no key. `make data` / `ingest` / `index` / `serve` for the real catalog; page at `/`, this deck at `/overview`.
* Docker: 40K index baked at build; Railway with `/ready` as health check; a key switches the LLM stages on.
* `AWS_deployment` branch: same app from CloudFront + Tokyo EC2 + Bedrock planning, serving the full 826,108-row catalog. Numbers: `docs/aws-latency.md`.

### Next, with another week

Human relevance labels on pooled results of the 28 queries. An ANN index with filtering past the in-memory range. A cross-encoder between retrieval and the LLM to cut candidates and latency. CLIP on the images every listing has. Streaming slots to the page as they finish. Embedding based near-duplicate detection.

Data: Amazon Reviews 2023, fashion metadata, McAuley Lab (UCSD), research use.
