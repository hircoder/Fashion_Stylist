# ADR-0001: Local embeddings with bge-small-en-v1.5

Status: accepted

## Context
Semantic retrieval needs an embedding model. Options: a hosted API (OpenAI
text-embedding-3-small/large, Voyage) or a local sentence-transformers model. The catalog
has 826K rows; the take-home must be runnable by a reviewer.

## Decision
Use `BAAI/bge-small-en-v1.5` (384-d) locally through sentence-transformers, with the bge
query instruction prefix. Make the embedder an interface so a hosted model is a config
change.

## Why
* Zero per-query and per-build cost, fully offline after one model download (130 MB),
  reproducible builds (the revision is pinned in Docker).
* On 8 conversational fashion queries over 40K listings it gave the most sensible top
  lists of the three small models tried; MiniLM-L6 was faster and clearly worse,
  arctic-embed-xs close but slightly behind.
* Throughput is enough: 761 docs/s on a laptop gpu, 100K rows in 2.7 minutes.

## Consequences
* Weaker than large hosted models on subtle semantics; accepted for a prototype.
* The index stores the model name and revision and the loader refuses a mismatch.

## Alternatives considered
Hosted embeddings (better quality, adds a key and cost to the build, not reproducible
without network), larger local models (bge-base: 2x slower for a small gain here).
