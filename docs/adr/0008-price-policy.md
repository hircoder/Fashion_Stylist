# ADR-0008: Price handling: strict explicit bounds, flagged unpriced items for inferred budgets

Status: accepted

## Context
94% of listings have no price. A strict filter on a budget read from the sentence left
slots with the only priced items that vaguely matched (a $13 wooden ring in a blazer
slot) while real blazers without a price were excluded.

## Decision
An explicit `max_price`/`min_price` on the request admits only items with a known price
inside the window (unless `include_unpriced=true`). A budget the planner inferred keeps
unpriced items in the same ranking, flagged `price_known=false`, with a small score bonus
for priced in-budget items and a warning in the response (`include_unpriced=false`
switches it off). The reranker sees both pools and is told type comes before price.

## Why
* Explicit bounds are promises; inferred budgets are hints. Treating them the same was
  measurably wrong for 94% of the catalog.
* Flags keep the response honest: nothing claims to fit a budget without a price.

## Consequences
* Three-state `include_unpriced` (null = auto) in the API, CLI and page.
* Per-slot allocations of a total budget get a 10% floor so the planner cannot starve a
  slot; allocations are scaled so they never exceed the total.

## Alternatives considered
Always strict (empty or junk slots), always relaxed (explicit bounds would lie), price
imputation (no basis in the data).
