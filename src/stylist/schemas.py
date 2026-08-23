"""Request / response models for the API (and the CLI, which prints the same thing)."""

from __future__ import annotations

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
    average_rating: float = Field(..., allow_inf_nan=False)
    rating_number: int
    store: str | None
    audience: str
    image_url: str | None
    url: str
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
    n_eligible: int
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


def product_url(parent_asin: str) -> str:
    return AMAZON_URL.format(asin=parent_asin)
