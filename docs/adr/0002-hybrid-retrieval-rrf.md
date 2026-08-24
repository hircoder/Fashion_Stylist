# ADR-0002: Hybrid retrieval: dense + bm25 fused with reciprocal rank fusion

## Context
Dense retrieval handles conversational phrasing; bm25 handles brand names and exact
phrases. The eval queries are conversational, real shop traffic has both.

## Decision
Score every row with both channels and fuse with reciprocal rank fusion (k=60). Either
channel can be switched off (`CHANNELS`).

## Why
* Measured: on raw conversational queries dense alone beats hybrid by 5 to 10 points of
  type-match, but with planner written queries they are within 1 to 2 points, and on 8
  brand queries hybrid finds the brand in 31/32 top-4 results vs 25/32 for dense alone.
* RRF needs no score calibration between channels, so there is nothing to tune.

## Consequences
* A small cost on conversational queries in the no-LLM path, accepted for brand safety.
* Both channels must stay row-aligned; the index stores checksums and row ids for both.

## Alternatives considered
Dense only (simpler, loses brands), weighted score sum (needs calibration), a learned
fusion (no labels to learn from).
