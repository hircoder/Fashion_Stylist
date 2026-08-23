# ADR-0007: LLM reranker, one call per slot, run in parallel

Status: accepted

## Context
Retrieval returns a ranked pool per slot. Picking the final items and explaining them
needs the request context (audience, occasion, budget) that a similarity score ignores.

## Decision
One LLM call per slot with the top 10 candidates as compact json, returning ordered picks
with a reason and the evidence fields used. Calls for all slots run concurrently. Every id
is validated against the slot's candidates; a failing slot keeps retrieval order.

## Why
* Output tokens dominate latency: one call for five slots took 20 s on a mid size model,
  five parallel calls take 3 to 7 s, the time of one.
* Isolation: one refusal, timeout or bad json only affects its slot.
* A cross-encoder cannot read "for my 6 year old" or "200 total" and does not write
  reasons.

## Consequences
* 1 + slots LLM calls per request; the reranker is one flag away from off.
* Reasons are model text and can only be spot checked; the evidence field names are the
  verifiable part.

## Alternatives considered
Single call for all slots (slow), cross-encoder rerank (no constraint reasoning, no
explanations), no rerank (retrieval order, measured 3 points worse on the default index).
