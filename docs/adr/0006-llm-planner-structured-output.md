# ADR-0006: LLM query planner with structured output and a regex fallback

## Context
A sentence like "outfit for the beach" must become product types plus constraints before
retrieval can do anything useful. The catalog has no taxonomy to map to.

## Decision
One LLM call returns a `PlannerOutput` (pydantic schema enforced by the provider's
structured output): intent, audience, occasion, season, budget and scope, style words,
and 1 to 5 slots with a listing-style query, keywords, exclude words and a budget share.
A normaliser enforces what a schema can't (caps, dedupe, allocations that add up). A regex
planner produces a valid single-slot plan when there is no key, a failure or a timeout.

## Why
* The planner is the feature: type-match goes from 0.73 (raw sentence) to 0.88 (planner
  queries) on the same retrieval.
* Structured output removes json parsing of free text; validation removes trust in the
  model's arithmetic (budget splits are re-checked and floored).

## Consequences
* 3 to 7 s of latency per request; cached per query, provider, model and prompt version.
* Slot names vary between runs; everything downstream matches by membership, not name.

## Alternatives considered
Hand written intent rules (cannot split outfits), a classifier over a taxonomy (no
taxonomy in the data), free text prompting with regex parsing (brittle).
