"""Request / response models for the API (and the CLI, which prints the same thing)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stylist.planner import Audience, QueryPlan

AMAZON_URL = "https://www.amazon.com/dp/{asin}"


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural-language request, e.g. 'an outfit for the beach this summer'.",
        examples=["I need an outfit to go to the beach this summer"],
    )
    k: int = Field(4, ge=1, le=10, description="Items to return per slot.")
    max_price: float | None = Field(
        None, ge=0, allow_inf_nan=False, description="Per-item maximum price in USD."
    )
    min_price: float | None = Field(
        None, ge=0, allow_inf_nan=False, description="Per-item minimum price in USD."
    )
    audience: Audience | None = Field(
        None, description="Override the audience guessed from the query."
    )
    include_unpriced: bool | None = Field(
        None,
        description=(
            "Whether items with an unknown price (94% of the catalog) may be returned when a "
            "price bound applies. Default (null): strict for explicit max_price/min_price, "
            "allowed for budgets inferred from the query text. Unpriced items are always "
            "flagged price_known=false and never claimed to fit the budget."
        ),
    )
    use_llm: bool = Field(True, description="False = regex planner + retrieval order only.")
    rerank: bool = Field(
        True, description="LLM rerank of the candidates (ignored if use_llm=false)."
    )

    @model_validator(mode="after")
    def _check(self) -> RecommendRequest:
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError("min_price must not exceed max_price")
        return self


class Item(BaseModel):
    rank: int
    row_id: int
    parent_asin: str
    title: str
    price: float | None = Field(None, allow_inf_nan=False)
    price_known: bool
    average_rating: float | None = Field(None, allow_inf_nan=False)
    rating_number: int
    store: str | None
    audience: str
    image_url: str | None
    url: str | None
    score: float = Field(..., allow_inf_nan=False)
    matched_keywords: list[str]
    reason: str
    evidence: list[str] = Field(default_factory=list)


class SlotResult(BaseModel):
    name: str
    search_query: str
    keywords: list[str]
    exclude_keywords: list[str] = Field(default_factory=list)
    budget_max: float | None
    n_eligible: int  # candidates retrieved inside the window (capped by the candidate depth)
    eligible_rows: int = 0  # index rows that pass the slot's audience / price masks at all
    items: list[Item]


class IndexInfo(BaseModel):
    rows: int
    sampling: str
    limit: int | None
    embedding_model: str
    built_at: str


class LLMInfo(BaseModel):
    provider: str | None
    model: str | None
    planner_used: str  # llm | heuristic
    rerank_used: bool
    rerank_model: str | None = None  # when a cheaper model reranks (LLM_RERANK_MODEL)
    plan_cache_hit: bool = False  # the plan came from the cache: no planner call was made
    calls: int = 0  # LLM calls attempted for this request (a cached plan costs none)
    failed_calls: int = 0  # attempts that ended in an error (still billed by most providers)
    input_tokens: int = 0
    output_tokens: int = 0


class RecommendResponse(BaseModel):
    request_id: str
    query: str
    plan: QueryPlan
    slots: list[SlotResult]
    note: str
    warnings: list[str]
    index_info: IndexInfo
    llm_info: LLMInfo
    timings: dict[str, float]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorBody(BaseModel):
    error: ErrorDetail


_RX_ASIN = re.compile(r"^[A-Z0-9]{10}$")
_IMAGE_HOSTS = (".media-amazon.com", ".ssl-images-amazon.com", ".images-amazon.com")


def safe_image_url(url: str | None) -> str | None:
    """Catalog image urls are untrusted text: only https links on Amazon's image hosts are
    handed to a browser (the page's content security policy allows exactly those)."""
    if not url or not isinstance(url, str):
        return None
    try:
        from urllib.parse import urlsplit

        parts = urlsplit(url.strip())
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host or not host.endswith(_IMAGE_HOSTS):
        return None
    return parts.geturl()


def product_url(parent_asin: str) -> str | None:
    """Amazon product page for a well formed ASIN (10 upper-case alphanumerics), else None.
    Catalog ids are untrusted text; they must never be spliced into a link unchecked."""
    if parent_asin and _RX_ASIN.match(parent_asin):
        return AMAZON_URL.format(asin=parent_asin)
    return None
