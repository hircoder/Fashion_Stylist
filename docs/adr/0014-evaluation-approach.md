# ADR-0014: Evaluation with hand written type rules and ablations, not human labels

Status: accepted

## Context
There are no relevance labels for this catalog and no time to create good ones.

## Decision
28 human style queries (20 conversational, 8 brand) with, per expected slot, a rule of title words that must and must
not appear, written before any output was seen. Report `keyword_match@k`, empty slots,
price violations and latency across retrieval configurations and three indexes, plus a
separate brand query check. State plainly that this measures product type, not taste.

## Why
* Reproducible and free; catches the failure that matters most (wrong product type) and
  regressions between changes.
* Ablations (channels on/off, planner on/off, rerank on/off) show what each part buys.

## Consequences
* Slots the planner invents that the rules did not anticipate are under counted.
* Human labels on pooled results are the first next step.
