# ADR-0005: Default index = the 100K most rated listings

Status: accepted

## Context
Embedding all 826K rows takes 19 minutes on a laptop gpu, 40+ on cpu, and 3.3 GB to
serve. Metadata coverage rises with popularity: price known for 26% of listings with 100+
ratings vs 4.9% with 0 to 4, features 79% vs 50%, department 49% vs 12%.

## Decision
`make index` builds the 100K most rated rows by default; `--sampling random` and
`--sampling all` are available. Every response carries `index_info.sampling`.

## Why
* Builds in 2.7 minutes; the LLM has better text to reason and explain with.
* Measured with the full pipeline: 0.915 type-match on popular-100K vs 0.893 on the full
  catalog and 0.871 on a random 100K, so the default is not hurting product type.

## Consequences
* Long tail items are absent by default; priced items are fewer (about 11K of 100K).
* The bias is explicit and measured rather than hidden.

## Alternatives considered
Full catalog by default (reviewer hostile build time), random sample (worse metadata, no
quality gain in the eval).
