# ADR-0009: Variant grouping computed at query time, nothing deleted at ingest

## Context
56,720 titles repeat under different ids and many more differ only by a size or colour
suffix ("(8 B(M) US, Silver)", ", Black, Large", "Wedding Shoes 8" vs "8.5").

## Decision
Keep every row. Derive a `group_key` from the title at query time (strip trailing
brackets incl. nested ones, size tokens, colour/size comma segments, bare trailing numbers
up to 20). One row per group survives a slot, and a group can fill at most one slot.

## Why
* Deleting at ingest guesses product identity and loses prices/images that only some
  variants carry.
* Computing at query time means a grouping fix never needs a 20 minute rebuild; it cost
  one bug fix to learn this.

## Consequences
* A few ms per request; a string heuristic that will merge "6 pairs" with "12 pairs" and
  miss colourways with different titles.

## Alternatives considered
Dedupe at ingest (destructive), embedding similarity dedupe (next step).
