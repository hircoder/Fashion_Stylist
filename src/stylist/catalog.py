"""Ingest the raw Amazon Fashion metadata into a flat, typed catalog (parquet).

Everything that touches the raw records lives here: price parsing, the audience
heuristic (the dataset has no usable category taxonomy, `categories` is empty for
every row), the grouping key used to collapse size/colour variants at query time,
and the text that gets embedded / indexed.

Nothing is dropped at ingest. Variant grouping (`group_key`) is computed from the title
at query time and only used for result diversification, so every row is kept.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PIPELINE_VERSION = "2"  # bump when any derived column changes (forces a rebuild)

AUDIENCES = ("women", "men", "girls", "boys", "baby", "unisex", "unknown")

# details keys worth keeping (the rest is packaging / model numbers / dates)
DETAIL_FIELDS = {
    "Department": "department",
    "Brand": "brand",
    "Material": "material",
    "Style": "style",
    "Color": "color",
    "Pattern": "pattern",
    "Age Range (Description)": "age_range",
}
DOC_TEXT_DETAILS = ("department", "material", "style", "color", "pattern", "brand")

CATALOG_COLUMNS = [
    "row_id",
    "parent_asin",
    "title",
    "average_rating",
    "rating_number",
    "price",
    "price_status",
    "store",
    "features",
    "description",
    "department",
    "brand",
    "material",
    "style",
    "color",
    "pattern",
    "age_range",
    "image_url",
    "audience",
    "doc_text",
]

_RX_WOMEN = re.compile(r"\b(women|woman|womens|women's|ladies|lady|female)\b", re.I)
_RX_MEN = re.compile(r"\b(men|man|mens|men's|male|gentlemen)\b", re.I)
_RX_GIRLS = re.compile(r"\b(girl|girls|girls')\b", re.I)
_RX_BOYS = re.compile(r"\b(boy|boys|boys')\b", re.I)
_RX_BABY = re.compile(r"\b(baby|babies|toddler|toddlers|infant|infants|newborn)\b", re.I)
_RX_UNISEX = re.compile(r"\bunisex\b", re.I)

_RX_PRICE_NUM = re.compile(r"^\$?\s*(\d[\d,]*(?:[.,]\d+)?)\s*$")
_RX_PRICE_RANGE = re.compile(r"\d.*[-–].*\d")

# trailing size noise: "US Size 8", "size 9-12", "XL", "2XL", "one size". Plain trailing
# numbers are NOT stripped (they are model numbers as often as sizes, e.g. "Pegasus 38").
_RX_SIZE_TAIL = re.compile(
    r"(?:[\s,\-]+(?:(?:us|uk|eu)\s+)?size\s*\d{1,2}(?:\.\d)?(?:\s*-\s*\d{1,2})?"
    r"|[\s,\-]+(?:x{0,3}s|x{0,3}l|xxl|xxxl|\dxl|m|one size|plus size|plus))+\s*$",
    re.I,
)
_RX_WS = re.compile(r"\s+")

# trailing ", Black, Large" / ", Multi/Flor" / ", M/US 4-6" style variant segments
_COLOR_WORDS = {
    "black",
    "white",
    "blue",
    "red",
    "green",
    "grey",
    "gray",
    "navy",
    "pink",
    "purple",
    "yellow",
    "brown",
    "beige",
    "khaki",
    "orange",
    "multi",
    "multicolor",
    "multicolour",
    "ivory",
    "cream",
    "tan",
    "gold",
    "silver",
    "olive",
    "burgundy",
    "wine",
    "teal",
    "coral",
    "floral",
    "camo",
    "leopard",
    "nude",
    "rose",
    "charcoal",
    "turquoise",
    "mint",
    "lavender",
    "maroon",
    "mustard",
    "denim",
    "print",
    "printed",
    "stripe",
    "striped",
    "solid",
    "dark",
    "light",
    "small",
    "medium",
    "large",
    "xlarge",
    "plus",
}
_RX_SIZE_TOKEN = re.compile(
    r"^(?:x{0,3}s|x{0,3}l|xxl|xxxl|\dx?l|m|\d{1,2}(?:\.\d)?(?:-\d{1,2})?|us|uk|eu|size|one|"
    r"months?|years?|[a-z]*\d+[a-z\d\-]*|\d+[a-z]*)$",
    re.I,
)


def _is_variant_segment(seg: str) -> bool:
    words = seg.replace("/", " ").split()
    if not words or len(words) > 3:
        return False
    hits = [w.lower() in _COLOR_WORDS or bool(_RX_SIZE_TOKEN.match(w)) for w in words]
    # at least one colour/size token, the rest short (titles are often truncated: "Flor")
    return any(hits) and all(h or len(w) <= 6 for h, w in zip(hits, words, strict=True))


def parse_price(value: object) -> tuple[float | None, str]:
    """Return (price_in_usd, status). Status is one of float/string/range/none/unparsed."""
    if value is None:
        return None, "none"
    if isinstance(value, bool):
        return None, "unparsed"
    if isinstance(value, int | float):
        if isinstance(value, float) and math.isnan(value):
            return None, "none"
        if not math.isfinite(value) or value < 0:
            return None, "unparsed"
        return float(value), "float"
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None, "none"
        m = _RX_PRICE_NUM.match(text)
        if m:
            return _number_from_string(m.group(1)), "string"
        if _RX_PRICE_RANGE.search(text):
            return None, "range"
        return None, "unparsed"
    return None, "unparsed"


def _number_from_string(num: str) -> float:
    """'1,299' -> 1299, '1,299.50' -> 1299.5, '12,99' -> 12.99 (comma as decimal)."""
    if "," in num and "." not in num and re.fullmatch(r"\d+,\d{1,2}", num):
        return float(num.replace(",", "."))
    return float(num.replace(",", ""))


def _audience_from_department(department: str | None) -> str | None:
    if not department:
        return None
    d = department.strip().lower()
    if d.startswith("baby") or "baby" in d:
        return "baby"
    if d.startswith("unisex"):
        return "unisex"
    if d.startswith("women") or d == "ladies":
        return "women"
    if d.startswith("men"):
        return "men"
    if d.startswith("girl"):
        return "girls"
    if d.startswith("boy"):
        return "boys"
    return None


def derive_audience(title: str, department: str | None) -> str:
    """Audience label. Department (when present) wins; otherwise look at the title words."""
    from_dept = _audience_from_department(department)
    if from_dept:
        return from_dept
    t = title or ""
    if _RX_UNISEX.search(t):
        return "unisex"
    women = bool(_RX_WOMEN.search(t))
    men = bool(_RX_MEN.search(t))
    if women and men:
        return "unisex"
    if women:
        return "women"
    if men:
        return "men"
    if _RX_BABY.search(t):
        return "baby"
    girls = bool(_RX_GIRLS.search(t))
    boys = bool(_RX_BOYS.search(t))
    if girls and boys:
        return "unisex"
    if girls:
        return "girls"
    if boys:
        return "boys"
    return "unknown"


def _strip_trailing_group(t: str) -> str:
    """Remove one trailing (...) or [...] group, nested brackets allowed: 'x (8 B(M) US)' -> 'x'."""
    t = t.rstrip()
    if not t or t[-1] not in ")]":
        return t
    close = t[-1]
    open_ = "(" if close == ")" else "["
    depth = 0
    for i in range(len(t) - 1, -1, -1):
        if t[i] == close:
            depth += 1
        elif t[i] == open_:
            depth -= 1
            if depth == 0:
                return t[:i].rstrip()
    return t  # unbalanced, leave it alone


def group_key(title: str) -> str:
    """Lowercased title with trailing variant noise (parentheses, sizes) removed.
    Computed at query time from the title, so it can change without rebuilding indexes."""
    t = (title or "").lower().strip()
    while True:
        stripped = _strip_trailing_group(t)
        if stripped == t:
            break
        t = stripped
    t = _RX_SIZE_TAIL.sub("", t)
    parts = [p.strip() for p in t.split(",")]
    while len(parts) > 1 and _is_variant_segment(parts[-1]):
        parts.pop()
    t = ", ".join(parts)
    t = _RX_SIZE_TAIL.sub("", t)
    t = _strip_trailing_small_number(t)
    t = _RX_WS.sub(" ", t).strip(" ,-")
    return t


_RX_TRAILING_NUMBER = re.compile(r"^(.*\S)\s+(\d{1,2}(?:\.5)?)$")


def _strip_trailing_small_number(t: str) -> str:
    """'... Wedding Shoes 8' -> '... Wedding Shoes': a bare trailing number up to 20 is a
    shoe/clothing size far more often than a model number ('Pegasus 38' is kept)."""
    m = _RX_TRAILING_NUMBER.match(t)
    if m and float(m.group(2)) <= 20:
        return m.group(1)
    return t


def _finite(value: object) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) and f >= 0 else 0.0


def _count(value: object) -> int:
    """Non-negative int32 rating count; anything unparsable counts as 0."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(f) or f < 0:
        return 0
    return int(min(f, 2**31 - 1))


def _first_image_url(images: object) -> str | None:
    if not isinstance(images, list) or not images:
        return None
    main = next((im for im in images if isinstance(im, dict) and im.get("variant") == "MAIN"), None)
    im = main or next((im for im in images if isinstance(im, dict)), None)
    if im is None:
        return None
    for key in ("large", "thumb", "hi_res"):
        url = im.get(key)
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def _clean_list(values: object, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        if isinstance(v, str):
            v = _RX_WS.sub(" ", v).strip()
            if v:
                out.append(v[:max_chars])
        if len(out) >= max_items:
            break
    return out


def build_doc_text(rec: dict) -> str:
    """Text fed to the embedder and BM25: title | features | useful details | store."""
    details = rec.get("details") or {}
    if not isinstance(details, dict):
        details = {}
    parts: list[str] = [_RX_WS.sub(" ", str(rec.get("title") or "")).strip()]
    feats = _clean_list(rec.get("features"), max_items=4, max_chars=120)
    if feats:
        parts.append("; ".join(feats))
    detail_bits = []
    for raw_key, col in DETAIL_FIELDS.items():
        if col in DOC_TEXT_DETAILS:
            val = details.get(raw_key)
            if isinstance(val, str) and val.strip():
                detail_bits.append(val.strip()[:60])
    if detail_bits:
        parts.append(", ".join(detail_bits))
    store = rec.get("store")
    if isinstance(store, str) and store.strip():
        parts.append(store.strip()[:60])
    return " | ".join(parts)[:600]


def normalize_record(raw: dict, row_id: int) -> dict:
    """Flatten one raw json record into the catalog row schema."""
    details = raw.get("details") or {}
    if not isinstance(details, dict):
        details = {}
    price, status = parse_price(raw.get("price"))
    title = _RX_WS.sub(" ", str(raw.get("title") or "")).strip()
    description = " ".join(_clean_list(raw.get("description"), max_items=3, max_chars=300))[:300]
    row: dict = {
        "row_id": int(row_id),
        "parent_asin": str(raw.get("parent_asin") or ""),
        "title": title,
        "average_rating": _finite(raw.get("average_rating")),
        "rating_number": _count(raw.get("rating_number")),
        "price": price,
        "price_status": status,
        "store": (raw.get("store") or None),
        "features": _clean_list(raw.get("features"), max_items=4, max_chars=120),
        "description": description,
    }
    for raw_key, col in DETAIL_FIELDS.items():
        val = details.get(raw_key)
        row[col] = val.strip()[:80] if isinstance(val, str) and val.strip() else None
    row["image_url"] = _first_image_url(raw.get("images"))
    row["audience"] = derive_audience(title, row["department"])
    row["doc_text"] = build_doc_text(raw)
    return row


def iter_raw(path: Path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_RATING_BUCKETS = ((0, 4), (5, 19), (20, 99), (100, None))


def _bucket(n: int) -> str:
    for lo, hi in _RATING_BUCKETS:
        if hi is None or n <= hi:
            if n >= lo:
                return f"{lo}-{hi if hi is not None else 'inf'}"
    return "0-4"


@dataclass
class IngestStats:
    rows: int = 0
    price_status: Counter = field(default_factory=Counter)
    coverage: dict[str, float] = field(default_factory=dict)
    by_rating_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    audience: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "price_status": dict(self.price_status),
            "coverage": self.coverage,
            "by_rating_bucket": self.by_rating_bucket,
            "audience": dict(self.audience),
        }


_COVERAGE_FIELDS = ("price", "features", "description", "image_url", "department", "material")

_SCHEMA = pa.schema(
    [
        ("row_id", pa.int64()),
        ("parent_asin", pa.string()),
        ("title", pa.string()),
        ("average_rating", pa.float32()),
        ("rating_number", pa.int32()),
        ("price", pa.float32()),
        ("price_status", pa.string()),
        ("store", pa.string()),
        ("features", pa.list_(pa.string())),
        ("description", pa.string()),
        ("department", pa.string()),
        ("brand", pa.string()),
        ("material", pa.string()),
        ("style", pa.string()),
        ("color", pa.string()),
        ("pattern", pa.string()),
        ("age_range", pa.string()),
        ("image_url", pa.string()),
        ("audience", pa.string()),
        ("doc_text", pa.string()),
    ]
)


def _has(row: dict, col: str) -> bool:
    v = row.get(col)
    if v is None:
        return False
    if isinstance(v, list | str):
        return len(v) > 0
    return True


def ingest(
    raw_path: Path, out_path: Path, limit: int | None = None, chunk_size: int = 50_000
) -> IngestStats:
    """Stream the raw jsonl(.gz) and write the flat catalog parquet. Returns coverage stats."""
    stats = IngestStats()
    cov = Counter()
    bucket_tot: Counter = Counter()
    bucket_cov: dict[str, Counter] = {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, _SCHEMA, compression="zstd")
    try:
        chunk: list[dict] = []
        for row_id, raw in enumerate(iter_raw(Path(raw_path))):
            if limit is not None and row_id >= limit:
                break
            row = normalize_record(raw, row_id)
            stats.rows += 1
            stats.price_status[row["price_status"]] += 1
            stats.audience[row["audience"]] += 1
            b = _bucket(row["rating_number"])
            bucket_tot[b] += 1
            bc = bucket_cov.setdefault(b, Counter())
            for col in _COVERAGE_FIELDS:
                if _has(row, col):
                    cov[col] += 1
                    bc[col] += 1
            chunk.append(row)
            if len(chunk) >= chunk_size:
                writer.write_table(_to_table(chunk))
                chunk = []
        if chunk or stats.rows == 0:
            writer.write_table(_to_table(chunk))
    finally:
        writer.close()
    n = max(stats.rows, 1)
    stats.coverage = {c: round(cov[c] / n, 4) for c in _COVERAGE_FIELDS}
    stats.by_rating_bucket = {
        b: {c: round(bucket_cov[b][c] / max(bucket_tot[b], 1), 4) for c in _COVERAGE_FIELDS}
        | {"rows": bucket_tot[b]}
        for b in sorted(bucket_tot, key=lambda x: int(x.split("-")[0]))
    }
    return stats


def _to_table(rows: Iterable[dict]) -> pa.Table:
    rows = list(rows)
    columns = {c: [r.get(c) for r in rows] for c in CATALOG_COLUMNS}
    return pa.table(columns, schema=_SCHEMA)


def select_rows(
    df: pd.DataFrame, limit: int | None, sampling: str = "popular", seed: int = 42
) -> pd.DataFrame:
    """Pick which catalog rows get indexed. Result is sorted by row_id (stable order)."""
    if sampling not in ("popular", "random", "all"):
        raise ValueError(f"sampling must be popular, random or all, got {sampling!r}")
    if sampling == "all" or limit is None or limit >= len(df):
        picked = df
    elif sampling == "popular":
        picked = df.sort_values(["rating_number", "row_id"], ascending=[False, True]).head(limit)
    else:
        picked = df.sample(n=limit, random_state=seed)
    return picked.sort_values("row_id").reset_index(drop=True)


def load_catalog_subset(
    catalog_path: Path, limit: int | None, sampling: str = "popular", seed: int = 42
) -> pd.DataFrame:
    """Read the parquet, pick rows with `select_rows` on two small columns, and only then
    materialise the chosen rows in pandas (keeps the 826K-row build under ~2 GB)."""
    table = pq.read_table(catalog_path)
    small = table.select(["row_id", "rating_number"]).to_pandas()
    small["_pos"] = range(len(small))
    picked = select_rows(small, limit=limit, sampling=sampling, seed=seed)
    subset = table.take(pa.array(picked["_pos"].to_numpy())).to_pandas()
    return subset.reset_index(drop=True)
