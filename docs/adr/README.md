# Architecture decision records

One page each. Context, decision, why, consequences, alternatives.

* [ADR-0001](0001-local-embeddings.md): Local embeddings with bge-small-en-v1.5
* [ADR-0002](0002-hybrid-retrieval-rrf.md): Hybrid retrieval: dense + bm25 fused with reciprocal rank fusion
* [ADR-0003](0003-masks-before-top-n.md): Apply constraints as masks before top-N
* [ADR-0004](0004-exact-search-in-memory.md): Exact brute-force search in memory instead of an ANN index
* [ADR-0005](0005-popular-first-default-index.md): Default index = the 100K most rated listings
* [ADR-0006](0006-llm-planner-structured-output.md): LLM query planner with structured output and a regex fallback
* [ADR-0007](0007-llm-reranker-per-slot.md): LLM reranker, one call per slot, run in parallel
* [ADR-0008](0008-price-policy.md): Price handling: strict explicit bounds, flagged unpriced items for inferred budgets
* [ADR-0009](0009-variant-grouping-at-query-time.md): Variant grouping computed at query time, nothing deleted at ingest
* [ADR-0010](0010-two-providers-one-protocol.md): Two LLM providers behind one async method, no framework
* [ADR-0011](0011-deadlines-and-typed-failures.md): One request deadline, stage budgets, typed failures with fallbacks
* [ADR-0012](0012-self-contained-index-and-baked-image.md): Self-contained index directory with checksums; demo index baked into the container image
* [ADR-0013](0013-serving-stack.md): FastAPI + React page + CLI; Docker on Railway
* [ADR-0014](0014-evaluation-approach.md): Evaluation with hand written type rules and ablations, not human labels
* [ADR-0015](0015-type-gate-and-brand-handling.md): A type gate on LLM plans, and a named brand as a first pass
* [ADR-0016](0016-request-limits-and-startup-validation.md): Request limits in the app, and an index that fails closed
