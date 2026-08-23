"""Candidate retrieval for every slot of a plan.

Pipeline per request:
  1. embed all slot queries in one batch, score every index row with one matmul
  2. per slot: BM25 scores for every row, eligibility masks (audience, price window)
     applied to BOTH score vectors *before* top-N, so a constraint can never produce an
     empty slot when eligible items exist further down the ranking
  3. reciprocal rank fusion of the two channels, plus two small additive terms:
     a keyword boost when a planner keyword literally appears in the title (and a
     penalty when one of the slot's exclude_keywords does), and a Bayesian-average
     rating prior
  4. one representative per `group_key` (size/colour variants collapse), highest score wins
  5. when a price window is set and unpriced items are allowed, a second ranked pool of
     unpriced items (flagged `in_window=False`) is merged into the same score order,
     with a small bonus for the priced in-budget ones

Scores are only meaningful within one slot; do not compare them across slots.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stylist.catalog import group_key
from stylist.config import Settings
from stylist.embeddings import Embedder
from stylist.index import SearchIndex
from stylist.planner import QueryPlan, Slot, SlotWindow

EXCLUDE_PENALTY = 1.0  # x keyword_boost: a title matching both a keyword and an exclude word
# nets zero (ambiguous, let the reranker decide), one matching only the exclude word drops
IN_WINDOW_BONUS = 0.25  # in units of RRF(rank 1), added to priced in-budget items when a
# budget exists and unpriced items are also in play (half of the keyword boost)
RATING_PRIOR_M = 20  # pseudo-count for the Bayesian rating (p75 of rating_number is 10; 20 = a
# deliberately stronger pull toward the mean for thinly rated items)
AUDIENCE_MATCH_BONUS = 0.15  # x unit: a row whose audience equals the requested one edges
# out unisex / unknown rows on otherwise equal relevance (they stay eligible)


@dataclass
class Candidate:
    idx: int
    row_id: int
    score: float
    group_key: str = ""
    title: str = ""
    dense_rank: int | None = None
    bm25_rank: int | None = None
    matched_keywords: list[str] = field(default_factory=list)
    excluded_keywords: list[str] = field(default_factory=list)
    in_window: bool = True
    price: float | None = None
    average_rating: float | None = None
    rating_number: int = 0
    store: str | None = None
    audience: str = "unknown"
    image_url: str | None = None
    parent_asin: str = ""
    material: str | None = None
    color: str | None = None
    style: str | None = None
    brand: str | None = None
    features: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def price_known(self) -> bool:
        return self.price is not None


@dataclass
class SlotCandidates:
    slot: Slot
    window: SlotWindow
    candidates: list[Candidate]  # score desc; in-window and (flagged) unpriced pool merged
    n_eligible: int  # in-window candidates returned (capped by n_candidates)
    warnings: list[str] = field(default_factory=list)
    eligible_rows: int = 0  # index rows that pass the window at all (before any ranking)


# --------------------------------------------------------------------------- masks


def _column(index, name: str) -> np.ndarray:
    """Typed numpy view of a catalog column, converted once per index and cached on it
    (the masks run on every request, the conversion must not)."""
    cache = getattr(index, "_column_cache", None)
    if cache is None:
        cache = {}
        try:
            index._column_cache = cache
        except AttributeError:  # plain objects in tests
            pass
    arr = cache.get(name)
    if arr is None:
        col = index.catalog[name]
        if name == "price":
            arr = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
        else:
            arr = col.to_numpy()
        cache[name] = arr
    return arr


def eligibility_masks(index, window: SlotWindow) -> tuple[np.ndarray, np.ndarray]:
    """(eligible, unpriced_pool) boolean arrays over index rows.

    eligible: passes the audience filter and, when a price bound exists, has a known
    price inside the window. unpriced_pool: passes audience, price unknown, only
    populated when a bound exists and include_unpriced is set.
    """
    n = index.n_rows
    ok = np.ones(n, dtype=bool)
    if window.audience:
        aud = _column(index, "audience")
        ok &= (aud == window.audience) | (aud == "unisex") | (aud == "unknown")
    has_bound = window.min_price is not None or window.max_price is not None
    if not has_bound:
        return ok, np.zeros(n, dtype=bool)
    price = _column(index, "price")
    known = ~np.isnan(price)
    in_range = known.copy()
    if window.min_price is not None:
        in_range &= price >= window.min_price
    if window.max_price is not None:
        in_range &= price <= window.max_price
    eligible = ok & in_range
    pool = (ok & ~known) if window.include_unpriced else np.zeros(n, dtype=bool)
    return eligible, pool


# --------------------------------------------------------------------------- fusion


def rrf_fuse(
    dense: np.ndarray | None,
    bm25: np.ndarray | None,
    mask: np.ndarray,
    top_n: int,
    k: int,
) -> list[Candidate]:
    """Reciprocal rank fusion over masked rows. A channel passed as None is switched off
    (it earns no rank points at all, which matters for clean ablations)."""
    if not mask.any():
        return []
    idxs = np.flatnonzero(mask)
    top_n = min(top_n, len(idxs))
    scores: dict[int, Candidate] = {}

    if dense is not None:
        d = dense[idxs]
        d_order = idxs[np.argsort(-d, kind="stable")[:top_n]]
        for rank, i in enumerate(d_order.tolist()):
            c = scores.setdefault(i, Candidate(idx=i, row_id=-1, score=0.0))
            c.dense_rank = rank
            c.score += 1.0 / (k + rank + 1)
    if bm25 is not None:
        b = bm25[idxs]
        b_nonzero = b > 0  # zero means "no query term in the doc", not a rank
        b_idxs = idxs[b_nonzero]
        b_order = b_idxs[np.argsort(-b[b_nonzero], kind="stable")[:top_n]]
        for rank, i in enumerate(b_order.tolist()):
            c = scores.setdefault(i, Candidate(idx=i, row_id=-1, score=0.0))
            c.bm25_rank = rank
            c.score += 1.0 / (k + rank + 1)
    return sorted(scores.values(), key=lambda c: (-c.score, c.idx))


def keyword_matches(title: str, keywords: list[str]) -> list[str]:
    low = (title or "").lower()
    out = []
    for kw in keywords:
        if kw and re.search(r"\b" + re.escape(kw) + r"(?:s|es)?\b", low):
            out.append(kw)
    return out


def bayes_rating(avg: float | None, count: int, m: int, prior: float) -> float:
    """Bayesian average: shrinks low-count ratings toward `prior` (the catalog mean at
    serving time), with `m` pseudo-ratings of weight. No rating at all means the prior."""
    if avg is None:
        return prior
    v = max(int(count or 0), 0)
    return (v / (v + m)) * float(avg) + (m / (v + m)) * prior


def diversify_by_group(cands: list[Candidate], seen: set[str]) -> list[Candidate]:
    """Keep the first (highest scored) candidate of each group_key; updates `seen`."""
    out = []
    for c in cands:
        key = c.group_key or f"__{c.idx}"
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# --------------------------------------------------------------------------- retriever

_HYDRATE_COLS = [
    "row_id",
    "parent_asin",
    "title",
    "average_rating",
    "rating_number",
    "price",
    "store",
    "audience",
    "image_url",
    "material",
    "color",
    "style",
    "brand",
    "features",
    "description",
]


def _none_if_nan(v):
    """None for None/NaN/NA scalars (numpy float32 NaN is not a Python float)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass  # lists / arrays
    return v


class Retriever:
    def __init__(self, index: SearchIndex, embedder: Embedder, settings: Settings):
        self.index = index
        self.embedder = embedder
        self.s = settings
        rated = index.catalog.loc[index.catalog["rating_number"] > 0, "average_rating"]
        prior = float(rated.mean()) if len(rated) else float("nan")
        self.rating_prior = prior if math.isfinite(prior) else 4.0
        self._unit = 1.0 / (settings.rrf_k + 1)  # RRF contribution of a rank-1 hit
        self._group_keys: dict[int, str] = {}  # row position -> group key, filled on demand

    def _group_key(self, idx: int, title: str) -> str:
        key = self._group_keys.get(idx)
        if key is None:
            key = self._group_keys[idx] = group_key(title)
        return key

    def _hydrate(self, c: Candidate) -> Candidate:
        row = self.index.catalog.iloc[c.idx]
        c.row_id = int(row["row_id"])
        c.parent_asin = str(row["parent_asin"])
        c.title = str(row["title"])
        rating = _none_if_nan(row["average_rating"])
        c.average_rating = float(rating) if rating is not None else None
        c.rating_number = int(_none_if_nan(row["rating_number"]) or 0)
        price = _none_if_nan(row["price"])
        c.price = round(float(price), 2) if price is not None else None  # float32 -> cents
        c.store = _none_if_nan(row["store"])
        c.audience = str(row["audience"])
        c.image_url = _none_if_nan(row["image_url"])
        c.group_key = self._group_key(c.idx, c.title)
        c.material = _none_if_nan(row["material"])
        c.color = _none_if_nan(row["color"])
        c.style = _none_if_nan(row["style"])
        c.brand = _none_if_nan(row["brand"])
        feats = row["features"]
        c.features = list(feats) if feats is not None and len(feats) else []
        c.description = str(_none_if_nan(row["description"]) or "")
        return c

    def _rank(
        self,
        dense: np.ndarray | None,
        bm25: np.ndarray | None,
        mask: np.ndarray,
        slot: Slot,
        top_n: int,
        audience: str | None = None,
    ) -> list[Candidate]:
        fused = rrf_fuse(dense, bm25, mask, top_n=top_n, k=self.s.rrf_k)
        for c in fused:
            self._hydrate(c)
            if audience and c.audience == audience:
                c.score += AUDIENCE_MATCH_BONUS * self._unit
            c.matched_keywords = keyword_matches(c.title, slot.keywords)
            if c.matched_keywords:
                c.score += self.s.keyword_boost * self._unit
            c.excluded_keywords = keyword_matches(c.title, slot.exclude_keywords)
            if c.excluded_keywords:  # look-alike type
                c.score -= EXCLUDE_PENALTY * self.s.keyword_boost * self._unit
            quality = bayes_rating(
                c.average_rating, c.rating_number, m=RATING_PRIOR_M, prior=self.rating_prior
            )
            c.score += self.s.quality_weight * self._unit * (quality / 5.0)
        fused.sort(key=lambda c: (-c.score, c.idx))
        return fused

    def _rank_distinct(
        self,
        dense: np.ndarray | None,
        bm25: np.ndarray | None,
        mask: np.ndarray,
        slot: Slot,
        needed: int,
        seen: set[str],
        audience: str | None = None,
    ) -> list[Candidate]:
        """Rank and collapse variants; widen the per-channel window (x4 each time) while
        the masked pool still has rows and we have fewer than `needed` distinct groups."""
        n_masked = int(mask.sum())
        top_n = self.s.top_n_per_channel
        while True:
            fused = self._rank(dense, bm25, mask, slot, top_n, audience)
            distinct = diversify_by_group(fused, set(seen))
            if len(distinct) >= needed or top_n >= n_masked:
                seen.update(c.group_key or f"__{c.idx}" for c in distinct)
                return distinct
            top_n = min(top_n * 4, n_masked)

    def retrieve(
        self,
        plan: QueryPlan,
        windows: list[SlotWindow],
        n_candidates: int,
        k: int,
    ) -> list[SlotCandidates]:
        """Up to n_candidates eligible rows per slot, plus up to k unpriced backfill rows
        when the window allows it. All slot queries are embedded and scored at once."""
        use_dense = "dense" in self.s.channels
        use_bm25 = "bm25" in self.s.channels
        queries = [s.search_query for s in plan.slots]
        dense_all = (
            self.index.dense_scores(self.embedder.encode_queries(queries)) if use_dense else None
        )
        out: list[SlotCandidates] = []
        for i, (slot, window) in enumerate(zip(plan.slots, windows, strict=True)):
            warnings: list[str] = []
            eligible, pool = eligibility_masks(self.index, window)
            dense = dense_all[i] if dense_all is not None else None
            bm25 = self.index.bm25_scores(slot.search_query) if use_bm25 else None
            seen: set[str] = set()
            aud = window.audience
            ranked = self._rank_distinct(dense, bm25, eligible, slot, n_candidates, seen, aud)
            ranked = ranked[:n_candidates]
            n_eligible = len(ranked)
            if pool.any():
                # unpriced pool, same depth, flagged. Merged into ONE score order: a known
                # in-budget price earns a small bonus, it does not trump relevance (a priced
                # wooden ring must not outrank an unpriced blazer in a blazer slot)
                extra = self._rank_distinct(dense, bm25, pool, slot, n_candidates, seen, aud)
                for c in ranked:
                    c.score += IN_WINDOW_BONUS * self._unit
                for c in extra[:n_candidates]:
                    c.in_window = False
                    ranked.append(c)
                ranked.sort(key=lambda c: (-c.score, c.idx))
                if n_eligible < k:
                    warnings.append(
                        f"slot '{slot.name}': only {n_eligible} items with a known price in the "
                        f"window, items with unknown price may be used (flagged)"
                    )
            elif n_eligible == 0:
                warnings.append(f"slot '{slot.name}': no eligible items")
            out.append(
                SlotCandidates(
                    slot, window, ranked, n_eligible, warnings, eligible_rows=int(eligible.sum())
                )
            )
        return out
