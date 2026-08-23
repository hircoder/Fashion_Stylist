"""The recommendation pipeline, one request at a time.

    plan (LLM or regex)  ->  merge constraints  ->  retrieve per slot  ->  rerank (LLM)
      ->  select k per slot with cross-slot uniqueness  ->  response

One deadline covers the whole request. Each LLM stage gets at most its own budget and
never more than what is left; when the planner fails or times out we fall back to the
regex planner, when the reranker cannot finish we keep retrieval order. The response
always says which path was taken (`llm_info`) and why something was skipped (`warnings`).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import OrderedDict

from stylist.config import Settings
from stylist.embeddings import Embedder
from stylist.index import SearchIndex
from stylist.llm import LLMClient, LLMError
from stylist.llm.prompts import PROMPT_VERSION
from stylist.planner import (
    HeuristicPlanner,
    LLMPlanner,
    QueryPlan,
    merge_constraints,
)
from stylist.reranker import LLMReranker, RankedSlot, RerankResult, deterministic_reason, fused_slot
from stylist.retrieval import Candidate, Retriever
from stylist.schemas import (
    IndexInfo,
    Item,
    LLMInfo,
    RecommendRequest,
    RecommendResponse,
    SlotResult,
    product_url,
)

log = logging.getLogger(__name__)

MIN_RERANK_SECONDS = 2.0  # do not start a rerank call with less than this left
MIN_PLAN_SECONDS = 0.5


class RecommendationService:
    def __init__(
        self,
        index: SearchIndex,
        embedder: Embedder,
        settings: Settings,
        llm: LLMClient | None = None,
    ):
        self.index = index
        self.settings = settings
        self.llm = llm
        self.retriever = Retriever(index, embedder, settings)
        self.heuristic = HeuristicPlanner()
        self.llm_planner = LLMPlanner(llm) if llm else None
        self.reranker = LLMReranker(llm) if llm else None
        self._plan_cache: OrderedDict[tuple, QueryPlan] = OrderedDict()
        self._retrieval_sem = asyncio.Semaphore(max(1, settings.retrieval_concurrency))

    # ------------------------------------------------------------------ planning

    def _cache_key(self, query: str, mode: str) -> tuple:
        norm = re.sub(r"\s+", " ", query.strip().lower())
        provider = self.llm.provider if self.llm else None
        model = self.llm.model if self.llm else None
        return (norm, mode, provider, model, PROMPT_VERSION, self.heuristic.version)

    def _cache_get(self, key: tuple) -> QueryPlan | None:
        plan = self._plan_cache.get(key)
        if plan is not None:
            self._plan_cache.move_to_end(key)
        return plan

    def _cache_put(self, key: tuple, plan: QueryPlan) -> None:
        self._plan_cache[key] = plan
        while len(self._plan_cache) > max(1, self.settings.plan_cache_size):
            self._plan_cache.popitem(last=False)

    async def _plan(
        self, req: RecommendRequest, deadline: float, warnings: list[str]
    ) -> tuple[QueryPlan, str]:
        use_llm = req.use_llm and self.llm_planner is not None
        mode = "llm" if use_llm else "heuristic"
        key = self._cache_key(req.query, mode)
        cached = self._cache_get(key)
        if cached is not None:
            return cached.model_copy(deep=True), mode
        if use_llm:
            remaining = deadline - time.monotonic()
            budget = min(self.settings.planner_budget_s, remaining)
            if budget >= MIN_PLAN_SECONDS:
                try:
                    plan = await asyncio.wait_for(
                        self.llm_planner.plan(req.query, timeout=budget),  # type: ignore[union-attr]
                        timeout=budget + 0.5,
                    )
                    self._cache_put(key, plan)
                    return plan.model_copy(deep=True), "llm"
                except (LLMError, asyncio.TimeoutError) as exc:
                    warnings.append(f"planner fell back to regex rules ({type(exc).__name__})")
                    log.warning("planner failed: %s: %s", type(exc).__name__, exc)
            else:
                warnings.append("planner fell back to regex rules (request deadline)")
        plan = self.heuristic.plan(req.query)
        self._cache_put(self._cache_key(req.query, "heuristic"), plan)
        return plan.model_copy(deep=True), "heuristic"

    # ------------------------------------------------------------------ selection

    @staticmethod
    def _select(ranked: list[RankedSlot], k: int, warnings: list[str]) -> list[list[Candidate]]:
        """Top k per slot in order, a product group is used by at most one slot."""
        used: set[str] = set()
        out: list[list[Candidate]] = []
        for rs in ranked:
            picked: list[Candidate] = []
            for c in rs.ordered:
                key = c.group_key or f"row:{c.row_id}"
                if key in used:
                    continue
                used.add(key)
                picked.append(c)
                if len(picked) >= k:
                    break
            if not picked:
                warnings.append(f"slot '{rs.slot.name}': nothing matched the constraints")
            elif len(picked) < k:
                warnings.append(f"slot '{rs.slot.name}': only {len(picked)} of {k} items found")
            out.append(picked)
        return out

    # ------------------------------------------------------------------ main entry

    async def recommend(self, req: RecommendRequest) -> RecommendResponse:
        t0 = time.monotonic()
        deadline = t0 + self.settings.request_deadline_s
        request_id = uuid.uuid4().hex[:12]
        warnings: list[str] = []
        timings: dict[str, float] = {}

        plan, planner_used = await self._plan(req, deadline, warnings)
        warnings.extend(plan.warnings)
        timings["plan_ms"] = round((time.monotonic() - t0) * 1000, 1)

        windows, merge_warnings = merge_constraints(
            plan,
            min_price=req.min_price,
            max_price=req.max_price,
            audience=req.audience,
            include_unpriced=req.include_unpriced,
        )
        warnings.extend(merge_warnings)

        do_rerank = req.use_llm and req.rerank and self.reranker is not None
        n_candidates = self.settings.rerank_candidates if do_rerank else max(req.k * 2, req.k)
        t1 = time.monotonic()
        async with self._retrieval_sem:
            slot_cands = await asyncio.to_thread(
                self.retriever.retrieve, plan, windows, n_candidates, req.k
            )
        for sc in slot_cands:
            warnings.extend(sc.warnings)
        timings["retrieve_ms"] = round((time.monotonic() - t1) * 1000, 1)

        rerank_used = False
        note = ""
        t2 = time.monotonic()
        if do_rerank:
            remaining = deadline - time.monotonic()
            budget = min(self.settings.rerank_budget_s, remaining)
            if budget < MIN_RERANK_SECONDS:
                warnings.append("rerank skipped (request deadline), results are in retrieval order")
                ranked = [fused_slot(sc) for sc in slot_cands]
            else:
                try:
                    result: RerankResult = await asyncio.wait_for(
                        self.reranker.rerank(req.query, plan, slot_cands, req.k, budget),  # type: ignore[union-attr]
                        timeout=budget + 0.5,
                    )
                    ranked, rerank_used, note = result.slots, result.used_llm, result.note
                    warnings.extend(result.warnings)
                except asyncio.TimeoutError:
                    warnings.append("rerank skipped (timeout), results are in retrieval order")
                    ranked = [fused_slot(sc) for sc in slot_cands]
        else:
            ranked = [fused_slot(sc) for sc in slot_cands]
        timings["rerank_ms"] = round((time.monotonic() - t2) * 1000, 1)

        picked = self._select(ranked, req.k, warnings)
        slots_out: list[SlotResult] = []
        for rs, items in zip(ranked, picked, strict=True):
            out_items = []
            for rank, c in enumerate(items, start=1):
                reason = rs.reasons.get(c.row_id)
                out_items.append(
                    Item(
                        rank=rank,
                        row_id=c.row_id,
                        parent_asin=c.parent_asin,
                        title=c.title,
                        price=c.price,
                        price_known=c.price is not None,
                        average_rating=round(c.average_rating, 2),
                        rating_number=c.rating_number,
                        store=c.store,
                        audience=c.audience,
                        image_url=c.image_url,
                        url=product_url(c.parent_asin),
                        score=round(c.score, 6),
                        matched_keywords=c.matched_keywords,
                        reason=reason.reason if reason else deterministic_reason(c, rs.window),
                        evidence=reason.evidence if reason else [],
                    )
                )
            slots_out.append(
                SlotResult(
                    name=rs.slot.name,
                    search_query=rs.slot.search_query,
                    keywords=rs.slot.keywords,
                    budget_max=rs.window.max_price,
                    n_eligible=rs.n_eligible,
                    items=out_items,
                )
            )

        timings["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
        meta = self.index.meta
        log.info(
            "request %s: slots=%d planner=%s rerank=%s timings=%s warnings=%d",
            request_id,
            len(slots_out),
            planner_used,
            rerank_used,
            timings,
            len(warnings),
        )
        log.debug("request %s query=%r", request_id, req.query)
        return RecommendResponse(
            request_id=request_id,
            query=req.query,
            plan=plan,
            slots=slots_out,
            note=note,
            warnings=warnings,
            index_info=IndexInfo(
                rows=self.index.n_rows,
                sampling=meta.sampling,
                limit=meta.limit,
                embedding_model=meta.embedding_model,
                built_at=meta.built_at,
            ),
            llm_info=LLMInfo(
                provider=self.llm.provider if self.llm else None,
                model=self.llm.model if self.llm else None,
                planner_used=planner_used,
                rerank_used=rerank_used,
            ),
            timings=timings,
        )
