# ADR-0003: Apply constraints as masks before top-N

## Context
Requests carry constraints (audience, price window). The usual shortcut is to retrieve
top-N and filter afterwards.

## Decision
Compute full score vectors for both channels and apply boolean eligibility masks before
taking top-N. Keep the full vectors so a second mask (unpriced pool) can be ranked without
a second retrieval.

## Why
* Post-filtering silently returns empty slots when eligible items sit below rank N; with
  masks, one eligible item anywhere in the index is enough.
* Cost is a boolean operation over 100K to 800K floats, microseconds to a millisecond.

## Consequences
* Exact search is required (or an ANN index that supports filtering).
* The eval shows zero empty slots across every configuration and index (docs/evaluation.md).

## Alternatives considered
Post-filter with over-fetch (still fails for rare constraints), pre-filtered sub-indexes
per audience (combinatorial).
