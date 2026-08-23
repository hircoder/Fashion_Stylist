import asyncio

import pytest

from stylist.config import Settings
from stylist.llm import FakeLLM, LLMTransportError
from stylist.planner import PlannerOutput
from stylist.reranker import SlotRerankOutput
from stylist.schemas import RecommendRequest
from stylist.service import RecommendationService


def _settings(**env):
    return Settings.from_env({"EMBEDDER": "hash", **env})


def _plan_out(*slot_specs, **kw):
    slots = [{"name": n, "search_query": q, "keywords": kws} for n, q, kws in slot_specs]
    base = {"intent": "t", "audience": None, "budget_scope": "unknown", "slots": slots}
    base.update(kw)
    return PlannerOutput.model_validate(base)


def _rerank_out(text="ok"):
    return SlotRerankOutput(picks=[], note=text)


def _stylist_llm(plan_out, rerank_out=None):
    """FakeLLM that answers planner calls with plan_out and rerank calls with rerank_out."""

    def handler(system, user, schema):
        if schema is PlannerOutput:
            return plan_out
        return rerank_out or _rerank_out()

    return FakeLLM(handler=handler)


async def test_no_llm_path_returns_fused_results(fixture_index, hash_embedder):
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=None)
    res = await svc.recommend(RecommendRequest(query="snow boots", k=3))
    assert res.llm_info.planner_used == "heuristic" and res.llm_info.rerank_used is False
    assert len(res.slots) == 1 and 0 < len(res.slots[0].items) <= 3
    first = res.slots[0].items[0]
    assert "boot" in first.title.lower()
    assert first.url.endswith(first.parent_asin) and first.rank == 1
    assert first.reason  # deterministic reason
    assert res.index_info.rows == fixture_index.n_rows
    assert set(res.timings) >= {"plan_ms", "retrieve_ms", "total_ms"}


async def test_llm_path_uses_plan_slots_and_rerank_note(fixture_index, hash_embedder):
    llm = _stylist_llm(
        _plan_out(
            ("boots", "snow boots", ["boot"]), ("hat", "winter beanie hat", ["beanie", "hat"])
        ),
        _rerank_out("Stay warm!"),
    )
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=llm)
    res = await svc.recommend(RecommendRequest(query="what to wear in the snow", k=2))
    assert [s.name for s in res.slots] == ["boots", "hat"]
    assert "Stay warm!" in res.note
    assert res.llm_info.planner_used == "llm" and res.llm_info.rerank_used is True
    assert len(llm.calls) == 3  # one planner call + one rerank call per slot
    assert res.plan.source == "llm"


async def test_same_product_group_never_appears_in_two_slots(fixture_index, hash_embedder):
    llm = _stylist_llm(_plan_out(("a", "snow boots", ["boot"]), ("b", "snow boots", ["boot"])))
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=llm)
    res = await svc.recommend(RecommendRequest(query="boots", k=3, rerank=False))
    a = {i.row_id for i in res.slots[0].items}
    b = {i.row_id for i in res.slots[1].items}
    assert a and b and not (a & b)
    assert [i.title.lower() for i in res.slots[0].items] != [
        i.title.lower() for i in res.slots[1].items
    ]


async def test_use_llm_false_never_calls_the_model(fixture_index, hash_embedder):
    llm = _stylist_llm(_plan_out(("x", "y", [])))
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=llm)
    res = await svc.recommend(RecommendRequest(query="sandals", use_llm=False))
    assert llm.calls == [] and res.llm_info.planner_used == "heuristic"


async def test_rerank_false_skips_second_call(fixture_index, hash_embedder):
    llm = _stylist_llm(_plan_out(("x", "sandals", ["sandal"])))
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=llm)
    res = await svc.recommend(RecommendRequest(query="sandals", rerank=False))
    assert len(llm.calls) == 1 and res.llm_info.rerank_used is False


async def test_planner_failure_falls_back_to_heuristic_with_warning(fixture_index, hash_embedder):
    llm = FakeLLM(responses=[LLMTransportError("boom"), _rerank_out()])
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=llm)
    res = await svc.recommend(RecommendRequest(query="snow boots"))
    assert res.llm_info.planner_used == "heuristic"
    assert any("planner" in w.lower() for w in res.warnings)
    assert res.slots[0].items


async def test_plan_cache_avoids_second_planner_call(fixture_index, hash_embedder):
    llm = _stylist_llm(_plan_out(("x", "sandals", ["sandal"])))
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=llm)
    await svc.recommend(RecommendRequest(query="Sandals ", rerank=False))
    await svc.recommend(RecommendRequest(query="  sandals", rerank=False))
    assert len(llm.calls) == 1


async def test_rerank_is_skipped_when_deadline_is_nearly_spent(fixture_index, hash_embedder):
    class SlowLLM(FakeLLM):
        async def complete_json(self, **kw):
            await asyncio.sleep(0.6)
            return await super().complete_json(**kw)

    llm = SlowLLM(handler=lambda s, u, schema: _plan_out(("x", "sandals", ["sandal"])))
    settings = _settings(REQUEST_DEADLINE_S="1.0", PLANNER_BUDGET_S="0.9")
    svc = RecommendationService(fixture_index, hash_embedder, settings, llm=llm)
    res = await svc.recommend(RecommendRequest(query="sandals"))
    assert res.llm_info.planner_used == "llm"
    assert res.llm_info.rerank_used is False
    assert any("deadline" in w.lower() for w in res.warnings)
    assert len(llm.calls) == 1


async def test_planner_timeout_falls_back_within_budget(fixture_index, hash_embedder):
    class StuckLLM(FakeLLM):
        async def complete_json(self, **kw):
            await asyncio.sleep(5)
            return None

    settings = _settings(REQUEST_DEADLINE_S="2.0", PLANNER_BUDGET_S="0.3")
    svc = RecommendationService(fixture_index, hash_embedder, settings, llm=StuckLLM())
    res = await asyncio.wait_for(svc.recommend(RecommendRequest(query="sandals")), timeout=3)
    assert res.llm_info.planner_used == "heuristic"


async def test_strict_price_window_only_returns_priced_items(fixture_index, hash_embedder):
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=None)
    res = await svc.recommend(RecommendRequest(query="swimsuit", max_price=5.0))
    for item in res.slots[0].items:
        assert item.price_known and item.price <= 5.0
    relaxed = await svc.recommend(
        RecommendRequest(query="swimsuit", max_price=5.0, include_unpriced=True)
    )
    assert len(relaxed.slots[0].items) >= len(res.slots[0].items)
    assert any(not i.price_known for i in relaxed.slots[0].items)


async def test_request_validation_rules():
    with pytest.raises(ValueError):
        RecommendRequest(query="   ")
    with pytest.raises(ValueError):
        RecommendRequest(query="x", min_price=50, max_price=10)
    with pytest.raises(ValueError):
        RecommendRequest(query="x", k=0)
    assert RecommendRequest(query="x").include_unpriced is False


async def test_twenty_concurrent_requests_all_succeed(fixture_index, hash_embedder):
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=None)
    reqs = [RecommendRequest(query=q) for q in ["boots", "sandals", "hat", "dress"] * 5]
    results = await asyncio.gather(*(svc.recommend(r) for r in reqs))
    assert len(results) == 20 and all(r.slots for r in results)


async def test_request_rejects_unknown_fields_and_non_finite_prices():
    with pytest.raises(ValueError):
        RecommendRequest(query="x", use_lm=False)  # typo must not be silently ignored
    with pytest.raises(ValueError):
        RecommendRequest(query="x", max_price=float("inf"))


async def test_retrieval_past_the_deadline_raises_request_timeout(fixture_index, hash_embedder):
    import time

    from stylist.service import RequestTimeout

    svc = RecommendationService(
        fixture_index, hash_embedder, _settings(REQUEST_DEADLINE_S="0.5"), llm=None
    )
    real = svc.retriever.retrieve

    def slow(*a, **kw):
        time.sleep(1.2)
        return real(*a, **kw)

    svc.retriever.retrieve = slow
    with pytest.raises(RequestTimeout):
        await svc.recommend(RecommendRequest(query="boots"))


async def test_planner_timeout_path_is_really_exercised(fixture_index, hash_embedder):
    class StuckLLM(FakeLLM):
        async def complete_json(self, **kw):
            await asyncio.sleep(5)
            return None

    settings = _settings(REQUEST_DEADLINE_S="3.0", PLANNER_BUDGET_S="0.6")
    svc = RecommendationService(fixture_index, hash_embedder, settings, llm=StuckLLM())
    t = asyncio.get_event_loop().time()
    res = await svc.recommend(RecommendRequest(query="sandals"))
    elapsed = asyncio.get_event_loop().time() - t
    assert res.llm_info.planner_used == "heuristic"
    assert 0.6 <= elapsed < 2.5
    assert any("TimeoutError" in w for w in res.warnings)


async def test_no_rerank_path_fetches_a_deep_enough_pool(fixture_index, hash_embedder):
    llm = _stylist_llm(_plan_out(("a", "women dress", ["dress"]), ("b", "women dress", ["dress"])))
    svc = RecommendationService(fixture_index, hash_embedder, _settings(), llm=llm)
    res = await svc.recommend(RecommendRequest(query="dresses", k=5, rerank=False))
    assert len(res.slots[0].items) == 5 and len(res.slots[1].items) == 5
