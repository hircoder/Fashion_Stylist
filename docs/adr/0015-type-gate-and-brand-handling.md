# ADR-0015: A type gate on LLM plans, and a named brand as a first pass

Status: accepted

## Context
Two failure modes showed up in the evaluation once the rules matched whole words. A query
like "running shoes with arch support for flat feet" filled the shoes slot with insoles:
every channel scores "arch support flat feet" higher than "running shoes", and a 0.5 unit
keyword boost cannot lift a shoe above twenty insoles. And "nike running shoes" came back
with one Nike item: the planner kept the brand out of the type keywords (on purpose) and
nothing downstream used it.

## Decision
1. The planner returns `brand` separately from the type keywords.
2. Retrieval ranks the brand's own rows first. With k or more rows of the right type the
   brand is a hard filter; otherwise its typed rows, at most two of its untyped rows and
   the typed rows of other brands follow, with a warning that says so.
3. For LLM plans (whose keywords are curated type synonyms) a type gate applies: once at
   least k candidates carry a type word in the title, off-type rows are dropped. The head
   noun of a multi-word keyword counts ("rain jacket" accepts any jacket), an accessory
   word that precedes the type word or forms a compound with it vetoes ("Shoe Insoles",
   "Insoles for Running Shoes"), unless the slot asks for that accessory.
4. The reranker's picks go through the same admissibility rule, and slots are only ever
   padded with type matches. An empty slot with a warning beats a ball pump.

## Why
* Masks before top-N already hold for audience and price; product type is the third
  hard constraint of a slot and the one users notice first.
* The planner's keywords are the cheapest, most reliable type signal available without a
  product taxonomy (the dataset has none). They are trusted only when the planner is the
  LLM; the regex planner's keywords are just query words.
* A brand is a strong constraint but the catalog coverage per brand is thin (one Levi's
  jeans row in the popular index), so it degrades to a preference with a warning instead
  of returning nothing.

## Consequences
* match@k on the 28 query set moved from 0.83 to about 0.89 with the full pipeline, and
  the brand queries from 1 of 4 on-brand to 3 to 4 of 4 where the catalog has the items.
* Titles that never name their type (model names only) can be gated out; the gate only
  bites when k typed alternatives exist, and the warning names the slot.
* The accessory list is a small curated regex; a title classifier is the next step.

## Alternatives considered
A bigger keyword boost (no: a boost cannot overturn twenty strong scores), brand as a
keyword (weak, and it pollutes the type signal), a hand built taxonomy (too much to
maintain for the gain in a take-home), asking the reranker alone to fix type (it only
sees the top 10, which were all insoles).
