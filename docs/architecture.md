# The architecture, explained

This file walks the whole system once: the parts, each step of a request, the data
that moves between them, and the questions each step tends to raise. It is the
written companion to the diagram (`docs/architecture.pdf`) and the animated flow on
slide 5 of `/overview`. Words are kept plain; where a technical term is unavoidable
it gets one gloss in parentheses.

The one-sentence version: a request is a sentence; an LLM turns it into a small
shopping list; ordinary search fills each list entry from 826,108 real listings
under hard filters; an LLM picks and explains the best few; everything that could
degrade says so in the response.

## The parts

| part | its one job | where |
|---|---|---|
| ingest | raw file -> one typed table, nothing guessed | `catalog.py` |
| index build | text -> vectors + keyword index + checksums | `index.py`, `embeddings.py` |
| planner | sentence -> slots and constraints, with a regex fallback | `planner.py` |
| retriever | score every row, filter first, rank, group variants | `retrieval.py` |
| reranker | one LLM call per slot, picks validated, reasons capped | `reranker.py` |
| service | deadlines, caches, selection, warnings | `service.py` |
| API | validation, limits, health, the UI and this deck | `api.py`, `schemas.py` |
| deploy kit | five scripts to build the AWS path, probes to measure it | `deploy/aws/` |

## Step 0: offline, before any request

The raw file is 826,108 gzip json lines (224 MB). Ingest streams it once into a
typed parquet table (a compressed column format, 150 MB, about 30 seconds). Two
rules there: a price becomes a number or null, never a guess; and the audience
(women / men / girls / boys / baby / unisex) is derived from the department field
when present, from title words otherwise, or stays "unknown".

The index build embeds each row's text (title, up to four feature lines, the
useful detail fields, the store) into a 384-number vector with a small local model
(bge-small-en-v1.5, pinned to one exact revision), and feeds the same text to a
keyword index (BM25). Both land in one directory with a `meta.json` that records
counts, the model and revision, and a sha256 fingerprint per file. At startup the
loader recomputes those fingerprints and refuses to serve on any mismatch, because
a misaligned vector silently returns wrong products, which is worse than a crash.

Numbers: the full catalog builds in 18.4 minutes on a laptop GPU and serves from
about 3.3 GB of memory; a quick 100K popular-rows build (2.6 minutes, 1 GB) exists
for demos and comparisons. The deployed service serves the full catalog.

Questions this step raises:

- Why a local embedding model and not a hosted one? Free, offline, reproducible
  (the exact model revision is pinned and checked), and good enough at this size;
  the notebook compares three candidates on real queries.
- Why keep BM25 at all? Vectors read sentences; BM25 reads exact words, which is
  what carries brand names like "levi's".
- Why is description not indexed? It exists on 7% of rows; the title is the field
  that exists everywhere, so the index is built around it.
- What happens if the build crashes halfway? It builds in a scratch directory and
  swaps in atomically at the end; a crash leaves the old index untouched.
- Why float16 on disk but float32 in memory? Half-size artifact; full precision
  for scoring after one conversion at load.
- Why exact search instead of an approximate index (FAISS, a vector database)?
  At 826K rows one full scoring pass is fast enough, gives 100% recall, and keeps
  the filters simple. Approximate search is the documented next step past about a
  million rows.

## Step 1: the request lands

`POST /recommend` with `{"query": "I need an outfit to go to the beach this
summer", "k": 4}` and optional fields: per-item `max_price` / `min_price`, an
`audience` override, `include_unpriced`, `use_llm`, `rerank`. Validation rejects
unknown fields, blank or over-500-character queries, k outside 1 to 10, and
impossible prices. A per-client rate limit and an in-flight cap answer 429 and 503
with a Retry-After header instead of queueing without bound. The request gets a
short id and one 40-second deadline that every later stage respects.

Questions:

- Why reject unknown fields instead of ignoring them? A typo like `max_prise`
  silently ignored would filter nothing and look like a bug in the results.
- Is the 40 s deadline a wall around everything? Almost: validation and admission
  run before the clock starts, and a retrieval computation already running cannot
  be killed mid-flight; it is bounded by never starting work the deadline cannot
  afford.
- Are the limits global? They are per process. The deployed box runs one worker
  today, so the configured numbers are the effective ones; with more workers or
  replicas they multiply, which is why a shared store (Redis) is on the gap list.
- Can a client spoof its address past the limiter? Forwarded headers are ignored
  unless the service is told it sits behind a proxy; the deployed origin only
  accepts traffic from CloudFront's address ranges.

## Step 2: plan

The sentence goes to the model as data (json-wrapped, labelled as data, never as
instructions) together with a schema the answer must fit. Back comes a plan:
intent, audience, occasion, season, a budget with its scope (per item or total),
style words, an optional brand, and 1 to 5 slots, each with a listing-style search
query, a few title keywords, and up to four exclude words. A normalizer then
enforces what a schema cannot: at most five slots, deduped lowercase keywords,
budgets that are finite and add up, a 10% floor per slot of a total.

Plans cache by normalized query (plus provider, model and prompt version), and
identical concurrent requests share one call. On any failure or timeout a regex
planner produces a one-slot plan from the raw sentence, the response says so in
`warnings`, and the failure is remembered for 30 seconds so an outage is not
retried on every request.

Example: "what should my husband wear to an outdoor wedding in june, budget 200
total" becomes five slots (dress pants, shirt, blazer, shoes, tie) with the $200
split across them, shoes getting more than the tie, no slot under $20.

Questions:

- Why an LLM here at all? The catalog has no category tree, and "outfit" is not a
  product word. Translating how people talk into how listings are written is the
  measured biggest quality jump in the whole pipeline (match@4 0.65 to 0.94 on
  the full catalog).
- Why structured output instead of parsing model text? No free-text parsing means
  no parsing bugs and a schema the provider itself enforces; all three adapters
  (Anthropic, OpenAI, Bedrock) speak the same one-method contract.
- What if the model returns nonsense inside the schema? The normalizer caps,
  cleans and re-floors everything; a plan with no usable slot falls back to the
  raw sentence.
- Can two users get different plans for the same query? Yes, model output varies
  between runs. The caches pin a plan once one lands, which is also why cold
  answers can differ from warm ones for about a second.
- What does a planner outage look like to users? Plain search with a warning,
  not an error page.

## Step 3: retrieve

The plan plus the request become one constraint window per slot (request fields
win). All slot queries are embedded in one batch and scored against every indexed
row with a single matrix multiplication (one big vectorized computation, no loop);
BM25 scores every row per slot. Then the order of operations that matters most in
the whole design: audience and price filters apply to the FULL score lists BEFORE
any top-N cut. Filtering after a cut quietly empties a slot whenever the eligible
items sit just past the cutoff; filtering first means one matching sandal under
$30 anywhere in 826,108 rows is found.

The two rankings merge by rank position (reciprocal rank fusion, so the two score
scales never need calibrating), then three small nudges: a boost when a planner
keyword appears in the title, the same size penalty for an exclude word, and a
rating prior that trusts 4.8 stars from 500 ratings over 5.0 from one.

Questions:

- Why not just filter the top 60 afterwards? That was the naive first version;
  it returned empty slots while eligible products existed at rank 61+.
- Is the audience filter strict? Rows labelled for a different audience are
  excluded; "unknown" rows stay in, because a quarter of the catalog is unknown
  and hiding it would be worse than occasionally showing a wrong-audience item.
- What are the retrieval costs on the full catalog? About 110 ms for a plan's
  worth of slots on a laptop CPU; 0.5 to 1.0 second server-side for an uncached
  five-slot plan on the deployed 4-vCPU box.
- What stops insoles from filling a "running shoes" slot? A type gate: once
  enough candidates whose titles name the product type exist, off-type rows drop.
  An accessory word next to the type word ("shoe insoles") vetoes the match.
- What about a named brand? The brand's rows are ranked first on their own; with
  enough typed rows the brand is a hard filter, otherwise other brands follow and
  a warning says so.

## Step 4: group and hydrate

The surviving rows collapse to one listing per product: a group key derived from
the title (trailing brackets, size tokens, colour segments stripped) merges size
and colour twins, and the ranking window widens by 4x until enough distinct
groups exist. Scoring fields for a window are gathered in one vectorized pass;
full row data is fetched only for candidates that survive. That distinction is
not cosmetic: the earlier per-row version cost one live request 19,783 row reads
and multi-second latencies before the acceptance battery caught it.

When the budget came from the sentence rather than an explicit filter, unpriced
rows stay in the running, flagged `price_known: false`, and priced in-budget rows
get a small bonus. An explicit `max_price` stays strict.

Questions:

- Why group at query time instead of deleting duplicates at ingest? Twin rows
  carry different prices and images, and a grouping bug fix must not cost a
  20-minute index rebuild.
- Why not group by product id? Every row has its own `parent_asin`, one id per
  listing, so the id cannot connect variants; only the title can.
- What does a wrong group cost? A false merge hides one distinct product; a
  missed merge shows the same item twice. The notebook measures the rule's reach
  (a third of the catalog sits in multi-listing groups) and shows its worst case.
- Why allow unpriced items under a budget at all? 94% of the catalog has no
  price. The strict version filled a blazer slot with a $13 wooden ring, because
  almost no blazer carries a price. Flagged honesty beat silent wrongness.

## Step 5: rerank

One LLM call per slot, all slots in parallel, each seeing the request, a plan
summary, and its top 10 candidates as compact json labelled untrusted catalog
data. The model returns up to k picks with a reason (capped at 15 words) and the
evidence fields it used, plus a short note. Validation is strict: only offered
ids, no duplicates, off-type picks dropped when enough on-type candidates exist,
links and emails stripped from every reason. A slot whose call fails or runs late
keeps retrieval order; the other slots keep their picks.

Questions:

- Why per slot instead of one call for the outfit? Output length drives latency,
  so five parallel calls cost about what one does, and one bad slot cannot spoil
  the other four.
- Why an LLM and not a cross-encoder (a small model that scores query-item
  pairs)? The LLM reads constraints a scorer cannot ("for my 6 year old", "$200
  total") and writes the reasons the page shows. The cross-encoder is the
  documented sub-second alternative, unbuilt.
- What can a hostile product title do? The prompt labels catalog text untrusted,
  the output is schema-bound, ids are validated, and reasons lose links; the
  worst case is picking a different candidate from the same slot.
- How strict is "only offered ids" exactly? Validation accepts ids from the
  slot's retrieved pool, which is slightly wider than the 10 shown to the model;
  tightening it to the shown slice is a noted small follow-up.
- What did it measure? About one point of match@4 over retrieval order on paired
  plans. Its real value is constraint reading, the written reasons, and keeping
  off-type items out.

## Step 6: select

Round robin across slots: every slot takes its best available product before any
slot takes its second, and a product group can fill only one slot per outfit, so
an early slot cannot drain a later one. Items the model did not explain get a
deterministic reason built from matched keywords, rating and price. Every
fallback that happened anywhere lands in `warnings`.

Questions:

- Does round robin remove all ordering bias? Nearly: when two slots want the
  same group in the same round, the earlier slot in the plan wins.
- Does a total budget guarantee the outfit fits it? No optimizer reshuffles
  picks; per-slot shares guide retrieval, and the service warns when the priced
  top picks add up over the stated total. Prices are also a snapshot, not live
  Amazon state.

## Step 7: respond

The body carries the plan as used, the slots with their items (title, price and
`price_known`, rating, image and product links built only from validated ids and
approved image hosts), the stylist note, all warnings, `index_info` (rows,
sampling, build time: which index answered), `llm_info` (which planner ran,
whether rerank ran, cache hits, calls and token counts), and per-stage timings.
The design rule behind the shape: every degradation must be visible, so the team
running this can debug an answer from the answer itself.

## The deployed serving profile

Same code, different knobs, on AWS: CloudFront at the edge (TLS, HTTP/3, static
assets cached), one EC2 origin in Tokyo running one uvicorn worker on the full
826,108-row index, Bedrock in-region for the models (Nova Lite plans, Nova Micro
reranks on request; the instance role is the only credential, no keys on the
box). The fast path:

1. A request checks the exact plan cache, then the semantic one (a paraphrase
   within cosine 0.92 reuses the nearest plan, guarded so a different budget or
   audience never crosses over).
2. On a miss it waits at most 0.10 s for the planner, then answers with the
   regex plan. The wait is measured, not taste: zero of twelve Nova plans ever
   landed inside 350 ms.
3. The Bedrock call keeps running (about 1 s) and lands in both plan caches.
4. Later requests, paraphrases included, get the LLM plan. Identical requests
   within 5 minutes come straight from a response cache that only ever stores
   warning-free answers, so a degraded answer is never frozen.

Measured through CloudFront from a client in Japan, on the full catalog:

| path | measured |
|---|---|
| steady repeat, connection kept alive | p50 13.0 ms (server 0.4 ms) |
| fresh TLS connection per request | p50 36.5 ms |
| response-cache hit | 15 ms |
| cold unique query (regex plan) | 464 ms |
| uncached answer on an LLM plan | 0.7 to 1.0 s server-side |
| unique-query capacity | ~3.8 requests/s before queueing |
| live quality, Nova Lite plans | match@4 0.819, success 0.571, zero empty slots |

The honest trade against the old 100K deployment: cached paths and quality
improved or held; uncached work costs about 5x more compute, so cold p50 rose
from 210 ms to 464 ms and per-box capacity fell from ~11 to ~3.8 unique
requests/s. Replica sizing starts from that last number.

## Cross-cutting questions

- Where does state live? Entirely in-process: plan, semantic, response and
  query-vector caches, the rate limiter, the in-flight cap. One worker means one
  coherent copy today; scaling out multiplies limits and splits caches until a
  shared store exists (first item on the hardening list).
- What happens in a Bedrock outage? Every request degrades to the regex planner
  with a warning, retried at most every 30 seconds; there is no circuit breaker
  or second-provider failover yet, both on the list.
- What is the biggest availability risk? One origin. A deploy costs a measured
  ~7 seconds of errors; an instance loss is an outage until failover exists.
- What breaks at 10x the catalog? Memory and the full scoring pass both grow
  linearly; past about a million rows the plan is approximate search with the
  same filters applied after, int8 vectors, and sharding.
- What breaks at 10x the traffic? Unique cold queries saturate the CPU first;
  repeat traffic rides the caches to 100 requests/s per box. Replicas plus a
  shared cache are the answer, and the provider's token quota binds before CPU
  once LLM reranking is on.
- Is the evaluation trustworthy? It is a type-correctness floor with honesty
  rules (rules written before output existed, paired plans, invented slots scored
  against the union). It cannot see style or taste; human labels are the stated
  next step.
- What is deliberately not built? Authentication, alarms, shipped logs, a second
  origin, CI-driven deploys. Each sits in `TODO.md` with a priority, a fix and an
  effort estimate; the audit found process gaps, not correctness ones.

In one line: filters before rankings, schemas before trust, caches before model
calls, warnings before silence, and a measurement before every number in this
file.
