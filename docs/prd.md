# PRD: semantic recommendation for a fashion catalog

Short product requirements doc, written before the code and updated as the data forced
changes. The README has the how, this is the what and the why.

## Problem

The catalog search box understands product words ("t-shirt", "shorts"). A large share of
shopping intent is not a product word, it's a situation: "an outfit for the beach this
summer", "what should my husband wear to an outdoor wedding", "something to keep my ears
warm". Today those queries return nothing useful, and the constraints inside them (who it's
for, how much, when) are lost completely.

## Users

* Shoppers typing a sentence into the fashion store's search box.
* The product team that owns search and needs to trust what comes back (and debug it).
* Engineers who will run the service next to the existing keyword search.

## Goals

1. Accept a natural language request and return relevant products, grouped by the product
   types the request implies (an outfit = several slots, a single item = one slot).
2. Respect constraints stated in the sentence: audience, occasion, season, budget.
3. Explain every pick in one sentence and show the evidence behind it.
4. Be honest about what the catalog can't support (94% of listings have no price).
5. Expose it as an HTTP API, with a CLI for operators and a small page for demos.
6. Run without an LLM key in a reduced mode, so the service never has a hard dependency
   on a third party for basic search.

## Non-goals

* Personalisation or session history.
* Image understanding (CLIP), even though every listing has an image.
* Training or fine tuning any model.
* Authentication, rate limiting, multi tenant deployment.

## Functional requirements and how they are met

| requirement | how | evidence |
|---|---|---|
| parse a sentence into product types + constraints | LLM planner with structured output (1 to 5 slots, audience, occasion, season, budget and scope, keywords, exclude words), regex planner as fallback | `planner.py`, 34 planner tests, sample plans in the README |
| find relevant products per type | hybrid retrieval: bge-small embeddings + bm25, fused with RRF | eval: 0.88 type-match with the LLM planner vs 0.46 bm25 only |
| respect audience and price constraints | boolean masks on the full score vectors before top-N | 0 price violations, 0 empty slots in 798 slot results |
| explain picks | LLM reranker returns reason + evidence fields per item, validated against the candidate set | `reranker.py`, samples in the README |
| honest prices | `price_known` on every item, strict explicit bounds, flagged unpriced items for inferred budgets | ADR-008 |
| outfit coherence | a product fills at most one slot, total budgets split across slots with a floor | `service.py` selection, tests |
| API, CLI, page | FastAPI `POST /recommend` with OpenAPI, `stylist` CLI, React page at `/` | `api.py`, `cli.py`, `ui/` |
| reduced mode without a key | `LLM_PROVIDER=none`, regex planner, retrieval order | `make demo` path, tests |

## Non-functional requirements

* Latency: under 15 s typical with the LLM, under 100 ms retrieval alone, a hard request
  deadline (40 s) so the service never hangs. Measured: ~10 s p50 with the LLM, 32 ms
  retrieval p50 on the 100K index.
* Footprint: the default index serves in ~1 GB RSS; the full catalog in 3.3 GB.
* Cost: embeddings are local (free). Two LLM calls per slot-less query, 1 + slots for an
  outfit; a few cents per request on a mid size model.
* Reliability: every provider failure has a typed fallback; the response states what ran.
* Operability: `/health`, `/ready`, per stage timings, index checksums verified at startup,
  a self contained index that can be shipped as a tarball, a cpu-only container.
* Privacy: queries are logged only at DEBUG level; no catalog or query data leaves the
  machine except the LLM calls (planner text and candidate titles).

## Success metrics

* Product type correctness on a fixed query set (`keyword_match@k`): target above 0.85
  with the LLM planner. Achieved 0.88 to 0.92 across three indexes.
* Zero empty slots and zero price violations on the eval set. Achieved.
* A reviewer can run the demo in under five minutes without a dataset download or a key.
  `make setup && make demo` does it in about two.

## Risks and open questions

* The catalog's price coverage (6%) limits any budget feature; mitigated by the flagged
  unpriced policy, not solved.
* The eval uses keyword rules, not human labels; it measures type, not taste.
* LLM output varies between runs; slot names and counts change. Everything downstream is
  validated, but two runs of the same query can differ in detail.
* Popularity-first indexing hides the long tail; the full catalog is one flag away.
