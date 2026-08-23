"""The recommendation pipeline, one request at a time.

    plan (LLM or regex)  ->  merge constraints  ->  retrieve per slot  ->  rerank (LLM)
      ->  select k per slot with cross-slot uniqueness  ->  response

One deadline covers the whole request, including waiting for a retrieval slot. Each LLM
stage gets at most its own budget and never more than what is left; when the planner
fails or times out we fall back to the regex planner, when a slot's rerank cannot finish
that slot keeps retrieval order. The response always says which path was taken
(`llm_info`) and why something was skipped (`warnings`).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from stylist.catalog import derive_audience
from stylist.config import Settings
from stylist.embeddings import Embedder
from stylist.index import SearchIndex
from stylist.llm import LLMClient, LLMError, ThrottledLLM, Usage, usage_scope
from stylist.llm.prompts import PROMPT_VERSION
from stylist.planner import HeuristicPlanner, LLMPlanner, QueryPlan, merge_constraints, parse_budget
from stylist.reranker import (
    LLMReranker,
    RankedSlot,
    RerankResult,
    deterministic_reason,
    fused_slot,
)
from stylist.retrieval import Candidate, Retriever
from stylist.schemas import (
    IndexInfo,
    Item,
    LLMInfo,
    RecommendRequest,
    RecommendResponse,
    SlotResult,
    product_url,
    safe_image_url,
)

log = logging.getLogger(__name__)

MIN_RERANK_SECONDS = 2.0  # do not start a rerank call with less than this left
MIN_PLAN_SECONDS = 0.5
SEMANTIC_PLAN_TTL_S = 3600.0  # an hour; catalog and prompts move slowly, plans should too


class RequestTimeout(Exception):
    """The request deadline passed before the response was built (the API maps this to 504)."""


class RecommendationService:
    def __init__(
        self,
        index: SearchIndex,
        embedder: Embedder,
        settings: Settings,
        llm: LLMClient | None = None,
        plan_cache: OrderedDict[tuple, QueryPlan] | None = None,
        rerank_llm: LLMClient | None = None,
    ):
        self.index = index
        self.settings = settings
        # one global cap on llm calls in flight, shared by the planner and the reranker
        sem = asyncio.Semaphore(settings.llm_concurrency)
        self.llm = ThrottledLLM(llm, sem) if llm else None
        self.rerank_llm = ThrottledLLM(rerank_llm, sem) if rerank_llm else self.llm
        self.retriever = Retriever(index, embedder, settings)
        self._embedder = embedder
        self.heuristic = HeuristicPlanner()
        self.llm_planner = LLMPlanner(self.llm) if self.llm else None
        self.reranker = (
            LLMReranker(self.rerank_llm, candidates=settings.rerank_candidates)
            if self.rerank_llm
            else None
        )
        self._closed = False
        # a caller may share one cache between services (the evaluation pairs configs)
        self._plan_cache: OrderedDict[tuple, QueryPlan] = (
            plan_cache if plan_cache is not None else OrderedDict()
        )
        self._plan_inflight: dict[tuple, asyncio.Future] = {}
        # semantic plan cache: near-duplicate queries reuse the nearest llm plan
        self._sem_capacity = 2048
        self._sem_vecs: np.ndarray | None = None
        self._sem_plans: list[tuple[QueryPlan, str, float] | None] = []
        self._sem_next = 0
        self._plan_failed_until: dict[tuple, float] = {}
        self._retrieval_sem = asyncio.Semaphore(max(1, settings.retrieval_concurrency))
        # a dedicated, bounded pool: retrieval is cpu bound numpy and must not spill into
        # the default executor and pile up under load
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.retrieval_concurrency), thread_name_prefix="retrieval"
        )

    # ------------------------------------------------------------------ planning

    def _cache_key(self, query: str, mode: str) -> tuple:
        norm = re.sub(r"\s+", " ", query.strip().lower())
        provider = self.llm.provider if self.llm else None
        model = self.llm.model if self.llm else None
        return (norm, mode, provider, model, PROMPT_VERSION, self.heuristic.version)

    def _cache_get(self, key: tuple) -> QueryPlan | None:
        if self.settings.plan_cache_size <= 0:
            return None
        plan = self._plan_cache.get(key)
        if plan is not None:
            self._plan_cache.move_to_end(key)
        return plan

    def _cache_put(self, key: tuple, plan: QueryPlan) -> None:
        if self.settings.plan_cache_size <= 0:
            return
        self._plan_cache[key] = plan
        while len(self._plan_cache) > self.settings.plan_cache_size:
            self._plan_cache.popitem(last=False)

    @staticmethod
    def _query_constraints(query: str) -> tuple:
        """The two constraint families cheap to read off the text and dangerous to get
        wrong across a semantic cache hit: the budget and the audience words."""
        budget = parse_budget(query)
        aud = derive_audience(query, None)
        return (budget, aud)

    def _semantic_lookup(self, query: str) -> tuple[QueryPlan | None, np.ndarray | None]:
        """(plan, query vector). The vector comes back either way so a later store does
        not embed twice. Only llm plans live in this cache, entries expire after an hour,
        and a cosine hit still has to agree with the query on budget and audience: 0.95
        of similarity happily bridges "men's" to "women's" or "$100" to "$1000", the
        deterministic guard does not."""
        if not self.settings.semantic_plan_cache:
            return None, None
        vec = self._embedder.encode_queries([query])[0].astype(np.float32)
        if self._sem_vecs is None or not len(self._sem_plans):
            return None, vec
        n = len(self._sem_plans)
        sims = self._sem_vecs[:n] @ vec
        best = int(np.argmax(sims))
        if float(sims[best]) < self.settings.semantic_plan_threshold:
            return None, vec
        entry = self._sem_plans[best]
        if entry is None:
            return None, vec
        plan, origin_query, stored_at = entry
        if time.monotonic() - stored_at > SEMANTIC_PLAN_TTL_S:
            return None, vec
        if self._query_constraints(query) != self._query_constraints(origin_query):
            return None, vec
        return plan.model_copy(deep=True), vec

    def _semantic_store(self, vec: np.ndarray | None, query: str, plan: QueryPlan) -> None:
        if vec is None or not self.settings.semantic_plan_cache or plan.source != "llm":
            return
        if self._sem_vecs is None:
            self._sem_vecs = np.zeros((self._sem_capacity, vec.shape[0]), dtype=np.float32)
        entry = (plan.model_copy(deep=True), query, time.monotonic())
        if len(self._sem_plans) < self._sem_capacity:
            self._sem_vecs[len(self._sem_plans)] = vec
            self._sem_plans.append(entry)
        else:  # ring buffer: the oldest entry goes
            self._sem_vecs[self._sem_next] = vec
            self._sem_plans[self._sem_next] = entry
            self._sem_next = (self._sem_next + 1) % self._sem_capacity

    def close(self) -> None:
        """Release the retrieval thread pool (the API calls this on shutdown, the CLI at exit)."""
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)

    async def _plan(
        self, req: RecommendRequest, deadline: float, warnings: list[str]
    ) -> tuple[QueryPlan, str, bool]:
        """(plan, planner used, cache hit)."""
        use_llm = req.use_llm and self.llm_planner is not None
        mode = "llm" if use_llm else "heuristic"
        key = self._cache_key(req.query, mode)
        cached = self._cache_get(key)
        if cached is not None:
            return cached.model_copy(deep=True), mode, True
        self._last_plan_shared = False
        qvec: np.ndarray | None = None
        if use_llm:
            near, qvec = self._semantic_lookup(req.query)
            if near is not None:
                # a near-duplicate query was planned before: reuse it, skip the LLM
                self._cache_put(key, near)
                return near.model_copy(deep=True), "llm", True
            if time.monotonic() < self._plan_failed_until.get(key, 0.0):
                warnings.append("planner fell back to regex rules (recent planner failure)")
            else:
                plan = await self._plan_with_llm(req.query, key, deadline, warnings)
                if plan is not None:
                    self._semantic_store(qvec, req.query, plan)
                    return plan.model_copy(deep=True), "llm", self._last_plan_shared
        plan = self.heuristic.plan(req.query)
        self._cache_put(self._cache_key(req.query, "heuristic"), plan)
        return plan.model_copy(deep=True), "heuristic", False

    def _cache_shared_result(self, key: tuple, fut: asyncio.Future) -> None:
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            # every waiter may have left before the call failed: without this, a planner
            # that always breaks looks like a service that just prefers regex plans
            log.warning("planner failed in the background: %s: %s", type(exc).__name__, exc)
            if self.settings.planner_failure_ttl_s > 0:
                self._plan_failed_until[key] = (
                    time.monotonic() + self.settings.planner_failure_ttl_s
                )
            return
        plan = fut.result()
        if plan is not None:
            self._cache_put(key, plan)

    async def _plan_with_llm(
        self, query: str, key: tuple, deadline: float, warnings: list[str]
    ) -> QueryPlan | None:
        """One planner call per key at a time: concurrent identical requests share it.
        A failure is remembered for PLANNER_FAILURE_TTL_S so an outage is not retried on
        every request."""
        remaining = deadline - time.monotonic()
        if remaining < MIN_PLAN_SECONDS:
            # the request is nearly out of time: do not even start a shared call for it
            warnings.append("planner fell back to regex rules (request deadline)")
            return None
        # the WAIT can be far below the floor (the fast profile waits 0.35 s and serves
        # heuristic results while the shared call finishes for the next request)
        budget = max(0.05, min(self.settings.planner_budget_s, remaining))
        inflight = self._plan_inflight.get(key)
        joined_existing = inflight is not None
        if inflight is None:
            # the shared call gets its own, longer timeout: a request only WAITS for
            # `budget`, but a plan that outlives every waiter still completes in the
            # background and lands in the cache for the next request (the fast profile
            # runs with a 0.35 s wait and a 20 s call timeout for exactly this reason)
            call_timeout = max(self.settings.planner_call_timeout_s, budget)
            inflight = asyncio.ensure_future(
                self.llm_planner.plan(query, timeout=call_timeout)  # type: ignore[union-attr]
            )
            self._plan_inflight[key] = inflight
            inflight.add_done_callback(lambda _f: self._plan_inflight.pop(key, None))
            # the shared call outlives a waiter that gave up: when it succeeds its plan is
            # cached for the next request even if noone is left waiting for it
            inflight.add_done_callback(lambda f: self._cache_shared_result(key, f))
        try:
            plan = await asyncio.wait_for(asyncio.shield(inflight), timeout=budget)
            if joined_existing:
                # the call and its tokens belong to the request that started it; for this
                # request the plan effectively came from a (very fresh) cache
                self._last_plan_shared = True
        except (LLMError, TimeoutError) as exc:
            warnings.append(f"planner fell back to regex rules ({type(exc).__name__})")
            # only a failure of the shared call itself goes into the negative cache; a
            # waiter whose own budget ran out must not poison the query for everyone
            own_timeout = isinstance(exc, TimeoutError) and not inflight.done()
            if own_timeout:
                # routine in the fast profile: the call keeps running for the next request
                log.info("plan not ready in %.2fs, serving the regex plan meanwhile", budget)
            else:
                log.warning("planner failed: %s: %s", type(exc).__name__, exc)
            if self.settings.planner_failure_ttl_s > 0 and not own_timeout:
                self._plan_failed_until[key] = (
                    time.monotonic() + self.settings.planner_failure_ttl_s
                )
            return None
        self._cache_put(key, plan)
        self._plan_failed_until.pop(key, None)
        return plan

    # ------------------------------------------------------------------ retrieval

    async def _retrieve_with_deadline(self, plan, windows, n_candidates: int, k: int, deadline):
        """Wait for a retrieval permit only until the deadline (a request that times out in
        the queue never starts work), then run retrieval in the bounded pool. A running
        thread cannot be cancelled, so the permit is released when the thread ends, not
        when the client gives up: RETRIEVAL_CONCURRENCY stays honest under a burst."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:  # the deadline already passed: never even queue the work
            raise RequestTimeout("request deadline exceeded before retrieval started")
        try:
            await asyncio.wait_for(self._retrieval_sem.acquire(), timeout=max(remaining, 0.01))
        except TimeoutError as exc:
            raise RequestTimeout("waited for a retrieval slot past the request deadline") from exc
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(
            self._executor, self.retriever.retrieve, plan, windows, n_candidates, k
        )
        fut.add_done_callback(lambda _f: self._retrieval_sem.release())
        try:
            return await asyncio.wait_for(
                asyncio.shield(fut), timeout=max(deadline - time.monotonic(), 0.01)
            )
        except TimeoutError as exc:
            fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            raise RequestTimeout("retrieval did not finish before the request deadline") from exc

    # ------------------------------------------------------------------ selection

    @staticmethod
    def _select(ranked: list[RankedSlot], k: int, warnings: list[str]) -> list[list[Candidate]]:
        """Top k per slot, a product group used by at most one slot. Round robin by rank
        (every slot takes its best available item before any slot takes its second) so an
        early slot cannot drain the shared candidates of a later one."""
        used: set[str] = set()
        out: list[list[Candidate]] = [[] for _ in ranked]
        cursors = [0] * len(ranked)
        progress = True
        while progress:
            progress = False
            for i, rs in enumerate(ranked):
                if len(out[i]) >= k:
                    continue
                while cursors[i] < len(rs.ordered):
                    c = rs.ordered[cursors[i]]
                    cursors[i] += 1
                    key = c.group_key or f"row:{c.row_id}"
                    if key in used:
                        continue
                    used.add(key)
                    out[i].append(c)
                    progress = True
                    break
        for rs, picked in zip(ranked, out, strict=True):
            if not picked:
                warnings.append(f"slot '{rs.slot.name}': nothing matched the constraints")
            elif len(picked) < k:
                warnings.append(f"slot '{rs.slot.name}': only {len(picked)} of {k} items found")
        return out

    # ------------------------------------------------------------------ main entry

    async def recommend(self, req: RecommendRequest) -> RecommendResponse:
        with usage_scope() as usage:
            return await self._recommend(req, usage)

    async def _recommend(self, req: RecommendRequest, usage: Usage) -> RecommendResponse:
        t0 = time.monotonic()
        deadline = t0 + self.settings.request_deadline_s
        request_id = uuid.uuid4().hex[:12]
        warnings: list[str] = []
        timings: dict[str, float] = {}

        plan, planner_used, plan_cache_hit = await self._plan(req, deadline, warnings)
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

        # an unset rerank field falls to the deployment default (the fast profile turns it
        # off and clients can still opt back in per request)
        rerank_wanted = (
            req.rerank if "rerank" in req.model_fields_set else self.settings.rerank_default
        )
        do_rerank = req.use_llm and rerank_wanted and self.reranker is not None
        # a pool deep enough that cross-slot de-duplication can still fill every slot even
        # when all slots overlap (the reranker only sees the top rerank_candidates of it)
        n_candidates = max(self.settings.rerank_candidates, req.k * max(3, len(plan.slots)))
        t1 = time.monotonic()
        try:
            slot_cands = await self._retrieve_with_deadline(
                plan, windows, n_candidates, req.k, deadline
            )
        except RequestTimeout:
            log.warning("request %s: retrieval exceeded the deadline", request_id)
            raise
        for sc in slot_cands:
            warnings.extend(sc.warnings)
        timings["retrieve_ms"] = round((time.monotonic() - t1) * 1000, 1)

        rerank_used = False
        note = ""
        t2 = time.monotonic()
        if do_rerank:
            budget = min(self.settings.rerank_budget_s, deadline - time.monotonic())
            if budget < MIN_RERANK_SECONDS:
                warnings.append("rerank skipped (request deadline), results are in retrieval order")
                ranked = [fused_slot(sc) for sc in slot_cands]
            else:
                # the reranker bounds itself per slot and keeps the slots that finished
                result: RerankResult = await self.reranker.rerank(  # type: ignore[union-attr]
                    req.query, plan, slot_cands, req.k, budget
                )
                ranked, rerank_used, note = result.slots, result.used_llm, result.note
                warnings.extend(result.warnings)
        else:
            ranked = [fused_slot(sc) for sc in slot_cands]
        timings["rerank_ms"] = round((time.monotonic() - t2) * 1000, 1)
        if time.monotonic() > deadline:
            log.warning("request %s: past the deadline after reranking", request_id)
            raise RequestTimeout("the request deadline passed before the response was built")

        picked = self._select(ranked, req.k, warnings)
        has_bound = any(w.min_price is not None or w.max_price is not None for w in windows)
        if has_bound and any(c.price is None for items in picked for c in items):
            warnings.append(
                "some returned items have an unknown price (price_known=false); they are not "
                "claimed to fit the budget"
            )
        if plan.budget_scope == "total" and plan.budget_max is not None:
            # one item per slot is what the shopper buys: compare the top picks, not every
            # alternative shown
            tops = [items[0].price for items in picked if items and items[0].price is not None]
            priced_sum = sum(tops)
            if priced_sum > plan.budget_max + 1e-6:
                warnings.append(
                    f"the top pick of each slot adds up to ${priced_sum:.2f}, above the stated "
                    f"total budget of ${plan.budget_max:.2f}"
                )

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
                        average_rating=(
                            round(c.average_rating, 2) if c.average_rating is not None else None
                        ),
                        rating_number=c.rating_number,
                        store=c.store,
                        audience=c.audience,
                        image_url=safe_image_url(c.image_url),
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
                    exclude_keywords=rs.slot.exclude_keywords,
                    budget_max=rs.window.max_price,
                    n_eligible=rs.n_eligible,
                    eligible_rows=rs.eligible_rows,
                    items=out_items,
                )
            )

        timings["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
        meta = self.index.meta
        log.info(
            "request %s: slots=%d planner=%s rerank=%s llm_calls=%d tokens=%d/%d "
            "timings=%s warnings=%d",
            request_id,
            len(slots_out),
            planner_used,
            rerank_used,
            usage.calls,
            usage.input_tokens,
            usage.output_tokens,
            timings,
            len(warnings),
        )
        if self.settings.log_queries:
            log.info("request %s query=%r", request_id, req.query)
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
                rerank_model=(
                    self.rerank_llm.model
                    if self.rerank_llm and self.rerank_llm is not self.llm
                    else None
                ),
                planner_used=planner_used,
                rerank_used=rerank_used,
                plan_cache_hit=plan_cache_hit,
                calls=usage.calls,
                failed_calls=usage.failed_calls,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
            timings=timings,
        )
