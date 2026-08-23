"""Query understanding: natural language -> QueryPlan (slots + constraints).

Two planners share one output type:

* `LLMPlanner` asks the model for a structured `PlannerOutput` and then runs it through
  `normalize_plan`, which enforces every invariant the schema cannot express (slot count,
  keyword caps, budget allocations that add up).
* `HeuristicPlanner` is the no-LLM fallback: one slot with the raw query, a regex budget
  and an audience guess. English only, but it can never fail.

`merge_constraints` then combines the plan with explicit request filters into one
price/audience window per slot. Request fields win over inferred ones, field by field.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from stylist.catalog import derive_audience
from stylist.llm import LLMClient
from stylist.llm.prompts import PLANNER_SYSTEM, planner_user

HEURISTIC_VERSION = "1"
MAX_SLOTS = 5
MAX_KEYWORDS = 6
MAX_QUERY_CHARS = 500  # same limit as the API request

Audience = Literal["women", "men", "girls", "boys", "baby", "unisex"]
BudgetScope = Literal["per_item", "total", "unknown"]


class SlotOutput(BaseModel):
    """One product type to retrieve. This is the LLM-facing shape (no constraints)."""

    name: str
    search_query: str
    keywords: list[str] = Field(default_factory=list)
    budget_max: float | None = None


class PlannerOutput(BaseModel):
    """What the LLM fills in. Kept free of validation constraints so the JSON schema
    stays simple for every provider; `normalize_plan` enforces the real rules."""

    intent: str = ""
    audience: Audience | None = None
    occasion: str | None = None
    season: str | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    budget_scope: BudgetScope = "unknown"
    style_keywords: list[str] = Field(default_factory=list)
    slots: list[SlotOutput] = Field(default_factory=list)


class Slot(BaseModel):
    name: str
    search_query: str
    keywords: list[str] = Field(default_factory=list)
    budget_max: float | None = None


class QueryPlan(BaseModel):
    intent: str = ""
    audience: Audience | None = None
    occasion: str | None = None
    season: str | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    budget_scope: BudgetScope = "unknown"
    style_keywords: list[str] = Field(default_factory=list)
    slots: list[Slot] = Field(min_length=1, max_length=MAX_SLOTS)
    source: Literal["llm", "heuristic"] = "llm"
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- budgets

_NUM = r"\$?\s*(\d+(?:\.\d+)?)\s*(?:usd|dollars?|bucks)?"
_RX_RANGE = re.compile(rf"(?:between\s+)?{_NUM}\s*(?:-|–|to|and)\s*{_NUM}", re.I)
_RX_MAX = re.compile(
    rf"(?:under|below|less than|cheaper than|max(?:imum)?|up to|at most|no more than|within)"
    rf"\s*{_NUM}",
    re.I,
)
_RX_MIN = re.compile(rf"(?:over|above|more than|at least|min(?:imum)?)\s*{_NUM}", re.I)
_RX_AROUND = re.compile(rf"(?:around|about|approximately|roughly)\s*{_NUM}", re.I)
_RX_BARE_DOLLAR = re.compile(r"\$\s*(\d+(?:\.\d+)?)")


def parse_budget(text: str) -> tuple[float | None, float | None]:
    """(min, max) in USD from phrases like 'under $50', '$20-40', 'at least 100'."""
    if not text:
        return None, None
    m = _RX_RANGE.search(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    m = _RX_MAX.search(text)
    if m:
        return None, float(m.group(1))
    m = _RX_MIN.search(text)
    if m:
        return float(m.group(1)), None
    m = _RX_AROUND.search(text)
    if m:
        v = float(m.group(1))
        return round(v * 0.7, 2), round(v * 1.3, 2)
    return None, None


# --------------------------------------------------------------------------- heuristic

_STOPWORDS = {
    "a",
    "an",
    "the",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "it",
    "its",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "and",
    "or",
    "but",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "am",
    "be",
    "been",
    "was",
    "were",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "need",
    "needs",
    "want",
    "wants",
    "would",
    "like",
    "looking",
    "look",
    "find",
    "get",
    "buy",
    "something",
    "some",
    "any",
    "please",
    "can",
    "could",
    "should",
    "go",
    "going",
    "wear",
    "wearing",
    "outfit",
    "outfits",
    "clothes",
    "clothing",
    "thing",
    "things",
    "from",
    "as",
    "so",
    "very",
    "really",
    "just",
    "also",
    "into",
    "about",
    "what",
    "which",
    "when",
    "where",
    "there",
    "here",
    "not",
    "no",
    "under",
    "over",
    "than",
    "less",
    "more",
    "up",
    "out",
    "dollars",
    "dollar",
    "usd",
    "bucks",
    "around",
    "between",
    "max",
    "min",
}
_RX_WORD = re.compile(r"[a-z][a-z'\-]+")


def content_words(text: str, limit: int = MAX_KEYWORDS) -> list[str]:
    out: list[str] = []
    for w in _RX_WORD.findall(text.lower()):
        w = w.strip("'-")
        if len(w) < 3 or w in _STOPWORDS or w in out:
            continue
        out.append(w)
        if len(out) >= limit:
            break
    return out


class HeuristicPlanner:
    """Regex-only planner used when no LLM is configured or the LLM stage fails."""

    version = HEURISTIC_VERSION

    def plan(self, query: str) -> QueryPlan:
        query = query.strip()
        lo, hi = parse_budget(query)
        aud = derive_audience(query, None)
        plan = QueryPlan(
            intent=query[:200],
            audience=None if aud == "unknown" else aud,  # type: ignore[arg-type]
            budget_min=lo,
            budget_max=hi,
            budget_scope="per_item" if (lo is not None or hi is not None) else "unknown",
            slots=[
                Slot(
                    name="items",
                    search_query=query[:MAX_QUERY_CHARS],
                    keywords=content_words(query),
                )
            ],
            source="heuristic",
        )
        if hi is not None:
            plan.slots[0].budget_max = hi
        return plan


# --------------------------------------------------------------------------- normalize


def _clean_keywords(raw: list[str]) -> list[str]:
    out: list[str] = []
    for k in raw or []:
        k = re.sub(r"\s+", " ", str(k)).strip().lower()[:40]
        if k and k not in out:
            out.append(k)
        if len(out) >= MAX_KEYWORDS:
            break
    return out


def _nonneg(v: float | None) -> float | None:
    """Budgets must be finite and non-negative, anything else counts as 'not given'."""
    if v is None or not math.isfinite(v) or v < 0:
        return None
    return float(v)


def _unique_names(slots: list[Slot]) -> None:
    seen: dict[str, int] = {}
    for s in slots:
        base = s.name.strip().lower() or "items"
        n = seen.get(base, 0) + 1
        seen[base] = n
        s.name = base if n == 1 else f"{base} {n}"


def normalize_plan(out: PlannerOutput, query: str) -> QueryPlan:
    """Enforce every invariant on an LLM plan and return a QueryPlan (source=llm)."""
    warnings: list[str] = []
    slots: list[Slot] = []
    for s in out.slots:
        sq = re.sub(r"\s+", " ", s.search_query or "").strip()
        if not sq:
            continue
        slots.append(
            Slot(
                name=(s.name or "items").strip()[:40] or "items",
                search_query=sq[:MAX_QUERY_CHARS],
                keywords=_clean_keywords(s.keywords),
                budget_max=_nonneg(s.budget_max),
            )
        )
    if len(slots) > MAX_SLOTS:
        warnings.append(f"planner returned {len(slots)} slots, keeping the first {MAX_SLOTS}")
        slots = slots[:MAX_SLOTS]
    _unique_names(slots)
    if not slots:
        warnings.append("planner returned no usable slot, searching the raw query")
        slots = [
            Slot(
                name="items",
                search_query=query.strip()[:MAX_QUERY_CHARS],
                keywords=content_words(query),
            )
        ]

    budget_min = _nonneg(out.budget_min)
    budget_max = _nonneg(out.budget_max)
    if budget_min is not None and budget_max is not None and budget_min > budget_max:
        warnings.append(f"budget_min {budget_min} above budget_max {budget_max}, dropping min")
        budget_min = None

    scope: BudgetScope = out.budget_scope
    if budget_max is None and budget_min is None:
        scope = "unknown"
    elif scope == "unknown":
        scope = "per_item"

    if budget_max is not None:
        if scope == "per_item" or len(slots) == 1:
            for s in slots:
                s.budget_max = budget_max
        else:  # total budget across several slots
            allocs = [s.budget_max for s in slots]
            if any(a is None for a in allocs):
                warnings.append("total budget split evenly across slots (planner gave no split)")
                for s in slots:
                    s.budget_max = round(budget_max / len(slots), 2)
            else:
                total = sum(a for a in allocs if a is not None)
                if total > budget_max + 1e-6:
                    factor = budget_max / total
                    warnings.append(
                        f"planner allocation {total:.2f} exceeded total budget "
                        f"{budget_max:.2f}, scaled down"
                    )
                    for s in slots:
                        s.budget_max = round((s.budget_max or 0.0) * factor, 2)
    else:
        for s in slots:
            s.budget_max = None

    return QueryPlan(
        intent=(out.intent or query)[:200],
        audience=out.audience,
        occasion=(out.occasion or None),
        season=(out.season or None),
        budget_min=budget_min,
        budget_max=budget_max,
        budget_scope=scope,
        style_keywords=_clean_keywords(out.style_keywords)[:8],
        slots=slots,
        source="llm",
        warnings=warnings,
    )


class LLMPlanner:
    def __init__(self, llm: LLMClient, max_tokens: int = 1500):
        self._llm = llm
        self._max_tokens = max_tokens

    async def plan(self, query: str, timeout: float = 10.0) -> QueryPlan:
        """Raises LLMError on any provider/validation problem; callers decide the fallback."""
        out = await self._llm.complete_json(
            system=PLANNER_SYSTEM,
            user=planner_user(query),
            schema=PlannerOutput,
            max_tokens=self._max_tokens,
            timeout=timeout,
        )
        return normalize_plan(out, query)


# --------------------------------------------------------------------------- constraints


@dataclass(frozen=True)
class SlotWindow:
    """Effective filters for one slot after merging plan + request."""

    min_price: float | None
    max_price: float | None
    audience: str | None
    include_unpriced: bool


def merge_constraints(
    plan: QueryPlan,
    *,
    min_price: float | None = None,
    max_price: float | None = None,
    audience: str | None = None,
    include_unpriced: bool | None = None,
) -> tuple[list[SlotWindow], list[str]]:
    """Request fields override the plan's for that field; inferred bounds that make the
    window empty are dropped with a warning (request bounds were validated upstream).

    Unpriced policy: an explicit request bound is a hard filter (unpriced items excluded
    unless include_unpriced=True). A budget the planner read out of the sentence is a
    softer signal, so unpriced items are allowed (flagged) unless include_unpriced=False.
    """
    warnings: list[str] = []
    eff_audience = audience or plan.audience
    explicit_bound = min_price is not None or max_price is not None
    allow_unpriced = include_unpriced if include_unpriced is not None else not explicit_bound
    windows: list[SlotWindow] = []
    for slot in plan.slots:
        lo = (
            min_price
            if min_price is not None
            else (plan.budget_min if plan.budget_scope == "per_item" else None)
        )
        hi = max_price if max_price is not None else slot.budget_max
        if lo is not None and hi is not None and lo > hi:
            if min_price is None:
                warnings.append(f"inferred minimum {lo} conflicts with max {hi}, ignoring it")
                lo = None
            else:
                warnings.append(f"inferred maximum {hi} conflicts with min {lo}, ignoring it")
                hi = None
        windows.append(SlotWindow(lo, hi, eff_audience, allow_unpriced))
    return windows, warnings
