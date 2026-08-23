# ADR-0004: Exact brute-force search in memory instead of an ANN index

Status: accepted

## Context
The catalog is 826K rows at most; the default index is 100K.

## Decision
Keep L2-normalised float32 embeddings in memory and score all slot queries of a request
with a single matrix multiply. No FAISS, no vector database.

## Why
* Measured p50 retrieval: 22 ms at 100K rows on cpu (28 ms on the laptop gpu), 110 ms at 826K, including bm25 and masks.
* Exact results make the eval and the masks simple to reason about; recall is 100%.
* One fewer moving part for a reviewer to install.

## Consequences
* RAM: 1.0 GB RSS for 100K, 3.3 GB for the full catalog. Past a few million rows an ANN
  index with filtering (HNSW) or a hosted vector db is the next step, documented.

## Alternatives considered
FAISS HNSW (faster at scale, approximate, filtering awkward), LanceDB/Qdrant (more
infrastructure than the problem needs).
