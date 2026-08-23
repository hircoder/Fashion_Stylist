"""Build the small committed fixture used by tests and `make demo`.

Takes the raw Amazon Fashion metadata and writes ~500 rows: 400 stratified by audience
(seeded, so it is reproducible) plus ~100 rows picked by product keywords so the demo
has something sensible to return for beach / winter / office style queries.

usage: python scripts/make_fixture.py data/raw/meta_Amazon_Fashion.jsonl.gz tests/fixtures/sample_500.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stylist.catalog import derive_audience, iter_raw  # noqa: E402

KEYWORDS = [
    "swim trunks",
    "bikini",
    "one piece swimsuit",
    "flip flops",
    "sandals",
    "sun hat",
    "sunglasses",
    "linen shirt",
    "beach cover up",
    "snow boots",
    "winter gloves",
    "scarf",
    "beanie",
    "ear warmers",
    "puffer jacket",
    "wool sweater",
    "chino pants",
    "dress shirt",
    "blazer",
    "pencil skirt",
    "running shoes",
    "rain jacket",
    "hiking boots",
    "cocktail dress",
    "leggings",
    "hoodie",
    "socks",
    "belt",
    "wallet",
    "backpack",
]
PER_KEYWORD = 3
STRATIFIED = 400
SEED = 42
MIN_RATINGS_FOR_KEYWORD_PICK = 20


def main(raw: str, out: str) -> None:
    rng = random.Random(SEED)
    by_aud: dict[str, list[dict]] = defaultdict(list)
    kw_hits: dict[str, list[dict]] = defaultdict(list)
    seen_titles: set[str] = set()
    for rec in iter_raw(Path(raw)):
        title = (rec.get("title") or "").strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        aud = derive_audience(title, (rec.get("details") or {}).get("Department"))
        # reservoir per audience, capped so memory stays small
        bucket = by_aud[aud]
        if len(bucket) < 5000:
            bucket.append(rec)
        elif rng.random() < 5000 / max(len(seen_titles), 1):
            bucket[rng.randrange(5000)] = rec
        if (rec.get("rating_number") or 0) >= MIN_RATINGS_FOR_KEYWORD_PICK:
            low = title.lower()
            for kw in KEYWORDS:
                if kw in low and len(kw_hits[kw]) < 200:
                    kw_hits[kw].append(rec)
                    break

    picked: list[dict] = []
    picked_ids: set[str] = set()
    for kw in KEYWORDS:
        hits = kw_hits.get(kw, [])
        rng.shuffle(hits)
        for rec in hits[:PER_KEYWORD]:
            if rec["parent_asin"] not in picked_ids:
                picked.append(rec)
                picked_ids.add(rec["parent_asin"])

    total_strat = sum(len(v) for v in by_aud.values())
    for aud, bucket in sorted(by_aud.items()):
        n = max(1, round(STRATIFIED * len(bucket) / total_strat))
        rng.shuffle(bucket)
        for rec in bucket[:n]:
            if rec["parent_asin"] not in picked_ids:
                picked.append(rec)
                picked_ids.add(rec["parent_asin"])

    rng.shuffle(picked)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for rec in picked:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(picked)} rows to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
