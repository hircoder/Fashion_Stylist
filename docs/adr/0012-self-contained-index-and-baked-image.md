# ADR-0012: Self-contained index directory with checksums; demo index baked into the container image

Status: accepted

## Context
Serving needs embeddings, bm25, the covered rows and metadata to agree exactly. Deploying
to a container host needs the index available at boot.

## Decision
The index directory holds everything (`embeddings.npy`, `bm25/`, `row_ids.npy`,
`catalog.parquet` of the indexed rows, `meta.json` with versions and sha256 of every
file). The loader verifies counts, `catalog.row_id == row_ids`, checksums and model
name/dimension, and refuses otherwise. Builds happen in a sibling directory and are
swapped in. The Docker image bakes a 40K row index at build time; alternatively an index
tarball can be installed at boot (`INDEX_URL` + `INDEX_SHA256`, size capped, safe
extraction, atomic install under a lock).

## Why
* Silent misalignment between an embedding row and a product row is the worst bug a
  retrieval system can have; hashing 600 MB at startup costs under a second.
* A baked index makes the Railway deploy need no volume and no external hosting.

## Consequences
* Image build takes ~5 minutes longer; CI builds without the baked index.
