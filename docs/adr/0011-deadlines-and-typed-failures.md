# ADR-0011: One request deadline, stage budgets, typed failures with fallbacks

Status: accepted

## Context
Two LLM stages plus retrieval can hang a request on a slow provider. The service must
degrade, not fail, and say what happened.

## Decision
A single request deadline (40 s) bounds every stage including retrieval; the planner gets
at most 15 s, the reranker 20 s, and a stage is skipped when less than its minimum is
left. Retrieval past the deadline is a 504 and the concurrency permit is held until the
worker thread actually finishes. Every provider failure is a typed error with a fallback
path and a test; the response lists each fallback in `warnings`.

## Why
* A timeout that releases its permit while the thread keeps running lets a burst of slow
  requests exceed `RETRIEVAL_CONCURRENCY`; holding it keeps the bound honest.
* Operators need to know why a response looks like retrieval order.

## Consequences
* Worst case latency is the deadline, not unbounded.
* Slightly more code in the service than a naive pipeline.
