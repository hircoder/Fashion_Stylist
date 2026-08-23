"""The AWS branch pieces: bedrock adapter, semantic plan cache, rerank default."""

import numpy as np
import pytest

from stylist.config import Settings
from stylist.llm import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTruncatedError,
    LLMValidationError,
    usage_scope,
)
from stylist.llm.bedrock_client import BedrockLLM
from stylist.planner import PlannerOutput
from stylist.schemas import RecommendRequest
from stylist.service import RecommendationService


class _Boto:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def converse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _tool_response(payload, stop="tool_use", usage=(100, 20)):
    return {
        "output": {
            "message": {
                "content": [{"toolUse": {"toolUseId": "t1", "name": "emit", "input": payload}}]
            }
        },
        "stopReason": stop,
        "usage": {"inputTokens": usage[0], "outputTokens": usage[1]},
    }


class _ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


async def test_bedrock_forces_one_tool_and_validates_the_input():
    boto = _Boto(
        _tool_response({"intent": "boots", "slots": [{"name": "boots", "search_query": "boots"}]})
    )
    llm = BedrockLLM(model="us.amazon.nova-micro-v1:0", client=boto)
    with usage_scope() as usage:
        out = await llm.complete_json(
            system="s", user="u", schema=PlannerOutput, max_tokens=500, timeout=5
        )
    assert out.slots[0].name == "boots"
    assert boto.kwargs["toolConfig"]["toolChoice"] == {"any": {}}
    assert (
        boto.kwargs["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["title"]
        == "PlannerOutput"
    )
    assert boto.kwargs["inferenceConfig"] == {"maxTokens": 500, "temperature": 0.0}
    assert "performanceConfig" not in boto.kwargs
    assert usage.calls == 1 and usage.input_tokens == 100 and usage.output_tokens == 20


async def test_bedrock_latency_optimized_flag():
    boto = _Boto(_tool_response({"intent": "x", "slots": [{"name": "a", "search_query": "a"}]}))
    llm = BedrockLLM(model="m", client=boto, latency_optimized=True)
    await llm.complete_json(system="s", user="u", schema=PlannerOutput, timeout=5)
    assert boto.kwargs["performanceConfig"] == {"latency": "optimized"}


@pytest.mark.parametrize(
    "code,exc",
    [
        ("AccessDeniedException", LLMAuthError),
        ("ThrottlingException", LLMRateLimitError),
        ("ModelTimeoutException", LLMTimeoutError),
        ("ValidationException", Exception),
    ],
)
async def test_bedrock_maps_client_errors(code, exc):
    llm = BedrockLLM(model="m", client=_Boto(error=_ClientError(code)))
    with pytest.raises(exc):
        await llm.complete_json(system="s", user="u", schema=PlannerOutput, timeout=5)


async def test_bedrock_truncation_and_missing_tool_call():
    llm = BedrockLLM(
        model="m", client=_Boto(_tool_response({"intent": "x", "slots": []}, stop="max_tokens"))
    )
    with pytest.raises(LLMTruncatedError):
        await llm.complete_json(system="s", user="u", schema=PlannerOutput, timeout=5)
    llm = BedrockLLM(
        model="m",
        client=_Boto(
            {"output": {"message": {"content": [{"text": "hi"}]}}, "stopReason": "end_turn"}
        ),
    )
    with pytest.raises(LLMValidationError):
        await llm.complete_json(system="s", user="u", schema=PlannerOutput, timeout=5)


def test_bedrock_provider_needs_no_key():
    s = Settings.from_env({"EMBEDDER": "hash", "LLM_PROVIDER": "bedrock"})
    assert s.llm_provider == "bedrock" and s.llm_model == "us.amazon.nova-micro-v1:0"


# ---------------------------------------------------------------------- semantic cache


def _handler_counting(counter):
    def handler(system, user, schema):
        counter.append(1)
        return PlannerOutput(
            intent="boots",
            slots=[{"name": "boots", "search_query": "snow boots", "keywords": ["boot"]}],
        )

    return handler


async def test_semantic_cache_reuses_the_plan_of_a_near_duplicate(fixture_index, hash_embedder):
    from stylist.llm import FakeLLM

    calls = []
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env(
            {"EMBEDDER": "hash", "SEMANTIC_PLAN_CACHE": "1", "SEMANTIC_PLAN_THRESHOLD": "0.95"}
        ),
        llm=FakeLLM(handler=_handler_counting(calls)),
    )
    try:
        r1 = await svc.recommend(RecommendRequest(query="warm snow boots", k=2, rerank=False))
        assert len(calls) == 1 and r1.llm_info.plan_cache_hit is False
        # identical text: the exact cache answers before the semantic one
        r2 = await svc.recommend(RecommendRequest(query="warm snow boots", k=2, rerank=False))
        assert len(calls) == 1 and r2.llm_info.plan_cache_hit is True
        # same words, different casing/punctuation: a different exact key, same vector
        r3 = await svc.recommend(RecommendRequest(query="Warm SNOW boots!", k=2, rerank=False))
        assert len(calls) == 1, "the semantic cache should have answered"
        assert r3.llm_info.plan_cache_hit is True and r3.llm_info.planner_used == "llm"
    finally:
        svc.close()


async def test_semantic_cache_off_by_default(fixture_index, hash_embedder):
    from stylist.llm import FakeLLM

    calls = []
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env({"EMBEDDER": "hash"}),
        llm=FakeLLM(handler=_handler_counting(calls)),
    )
    try:
        await svc.recommend(RecommendRequest(query="warm snow boots", k=2, rerank=False))
        await svc.recommend(RecommendRequest(query="Warm SNOW boots!", k=2, rerank=False))
        assert len(calls) == 2
    finally:
        svc.close()


def test_semantic_ring_buffer_wraps(fixture_index, hash_embedder):
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env({"EMBEDDER": "hash", "SEMANTIC_PLAN_CACHE": "1"}),
        llm=None,
    )
    try:
        svc._sem_capacity = 3
        from stylist.planner import QueryPlan, Slot

        for i in range(5):
            vec = np.zeros(hash_embedder.dim, dtype=np.float32)
            vec[i] = 1.0
            plan = QueryPlan(
                intent=f"p{i}", slots=[Slot(name="a", search_query=f"q{i}")], source="llm"
            )
            svc._semantic_store(vec, f"q{i}", plan)
        assert len(svc._sem_plans) == 3
        assert {e[0].intent for e in svc._sem_plans} == {"p2", "p3", "p4"}
    finally:
        svc.close()


# ---------------------------------------------------------------------- rerank default


async def test_rerank_default_off_is_honoured_unless_the_client_says(fixture_index, hash_embedder):
    from stylist.llm import FakeLLM
    from stylist.reranker import SlotRerankOutput

    def handler(system, user, schema):
        if schema is PlannerOutput:
            return PlannerOutput(
                intent="b", slots=[{"name": "boots", "search_query": "boots", "keywords": ["boot"]}]
            )
        return SlotRerankOutput(picks=[], no_good_match=False, note="")

    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env({"EMBEDDER": "hash", "RERANK_DEFAULT": "0", "PLAN_CACHE_SIZE": "0"}),
        llm=FakeLLM(handler=handler),
    )
    try:
        r = await svc.recommend(RecommendRequest(query="boots", k=2))
        assert r.llm_info.rerank_used is False  # the default said no
        r = await svc.recommend(RecommendRequest(query="boots", k=2, rerank=True))
        assert r.llm_info.rerank_used is True  # the client said yes explicitly
    finally:
        svc.close()


async def test_semantic_hit_is_refused_when_budget_or_audience_differ(fixture_index, hash_embedder):
    from stylist.llm import FakeLLM

    calls = []
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env(
            {"EMBEDDER": "hash", "SEMANTIC_PLAN_CACHE": "1", "SEMANTIC_PLAN_THRESHOLD": "0.5"}
        ),
        llm=FakeLLM(handler=_handler_counting(calls)),
    )
    try:
        await svc.recommend(RecommendRequest(query="snow boots under $50", k=2, rerank=False))
        assert len(calls) == 1
        # near-identical text, different budget: the guard must force a fresh plan
        await svc.recommend(RecommendRequest(query="snow boots under $500", k=2, rerank=False))
        assert len(calls) == 2
        # near-identical text, different audience word: fresh plan again
        await svc.recommend(
            RecommendRequest(query="womens snow boots under $50", k=2, rerank=False)
        )
        assert len(calls) == 3
    finally:
        svc.close()


async def test_shared_planner_call_outlives_the_waiters_budget(fixture_index, hash_embedder):
    seen = {}

    class Recorder:
        provider = "fake"
        model = "fake"

        async def complete_json(self, *, system, user, schema, max_tokens=2000, timeout=30.0):
            seen["timeout"] = timeout
            return PlannerOutput(intent="x", slots=[{"name": "a", "search_query": "boots"}])

    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env(
            {"EMBEDDER": "hash", "PLANNER_BUDGET_S": "0.7", "PLANNER_CALL_TIMEOUT_S": "20"}
        ),
        llm=Recorder(),
    )
    try:
        await svc.recommend(RecommendRequest(query="boots", k=2, rerank=False))
        assert seen["timeout"] == 20.0  # the call runs on its own clock, not the waiter's
    finally:
        svc.close()


async def test_background_planner_failure_is_logged_and_negative_cached(
    fixture_index, hash_embedder, caplog
):
    import asyncio

    from stylist.llm import FakeLLM, LLMTransportError

    class SlowFail(FakeLLM):
        async def complete_json(self, *, system, user, schema, max_tokens=2000, timeout=30.0):
            await asyncio.sleep(0.5)  # longer than the waiter's budget
            raise LLMTransportError("bedrock exploded")

    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env(
            {"EMBEDDER": "hash", "PLANNER_BUDGET_S": "0.1", "PLANNER_FAILURE_TTL_S": "30"}
        ),
        llm=SlowFail(),
    )
    try:
        r = await svc.recommend(RecommendRequest(query="boots", k=2, rerank=False))
        assert r.llm_info.planner_used == "heuristic"
        await asyncio.sleep(0.6)  # let the shared call fail in the background
        assert svc._plan_failed_until  # the failure is remembered
        assert any("planner failed in the background" in rec.message for rec in caplog.records)
    finally:
        svc.close()


# ---------------------------------------------------------------------- round 2 caches


class _CountingEmbedder:
    """Wraps the hash embedder and counts encode_queries calls."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.name = inner.name
        self.dim = inner.dim

    def encode_queries(self, texts):
        self.calls += 1
        return self._inner.encode_queries(texts)

    def encode_docs(self, texts):
        return self._inner.encode_docs(texts)


async def test_warm_requests_skip_the_query_encoder(fixture_index, hash_embedder):
    counting = _CountingEmbedder(hash_embedder)
    svc = RecommendationService(
        fixture_index, counting, Settings.from_env({"EMBEDDER": "hash"}), llm=None
    )
    try:
        await svc.recommend(RecommendRequest(query="snow boots", k=2, use_llm=False))
        first = counting.calls
        assert first >= 1
        await svc.recommend(RecommendRequest(query="snow boots", k=2, use_llm=False))
        assert counting.calls == first  # the slot query vector came from the cache
    finally:
        svc.close()


async def test_response_cache_serves_a_copy_with_a_fresh_request_id(fixture_index, hash_embedder):
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env({"EMBEDDER": "hash", "RESPONSE_CACHE_TTL_S": "60"}),
        llm=None,
    )
    try:
        r1 = await svc.recommend(RecommendRequest(query="snow boots", k=2, use_llm=False))
        r2 = await svc.recommend(RecommendRequest(query="snow boots", k=2, use_llm=False))
        assert r2.request_id != r1.request_id
        assert r2.served_from_cache is True and r1.served_from_cache is False
        assert r2.llm_info.calls == 0
        assert [i.title for s in r2.slots for i in s.items] == [
            i.title for s in r1.slots for i in s.items
        ]
        assert 0.0 <= r2.timings["total_ms"] < 50.0  # real serve time, not the old compute
        # a different body must not collide
        r3 = await svc.recommend(RecommendRequest(query="snow boots", k=3, use_llm=False))
        assert len(r3.slots[0].items) == 3
    finally:
        svc.close()


async def test_response_cache_key_sees_explicitly_set_fields(fixture_index, hash_embedder):
    from stylist.llm import FakeLLM
    from stylist.reranker import SlotRerankOutput

    def handler(system, user, schema):
        if schema is PlannerOutput:
            return PlannerOutput(
                intent="b", slots=[{"name": "boots", "search_query": "boots", "keywords": ["boot"]}]
            )
        return SlotRerankOutput(picks=[], no_good_match=False, note="")

    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env(
            {"EMBEDDER": "hash", "RESPONSE_CACHE_TTL_S": "60", "RERANK_DEFAULT": "0"}
        ),
        llm=FakeLLM(handler=handler),
    )
    try:
        r1 = await svc.recommend(RecommendRequest(query="boots", k=2))
        assert r1.llm_info.rerank_used is False
        r2 = await svc.recommend(RecommendRequest(query="boots", k=2, rerank=True))
        assert r2.llm_info.rerank_used is True  # not served from r1's cache entry
    finally:
        svc.close()


async def test_response_cache_expires(fixture_index, hash_embedder):
    import asyncio

    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env({"EMBEDDER": "hash", "RESPONSE_CACHE_TTL_S": "0.15"}),
        llm=None,
    )
    try:
        r1 = await svc.recommend(RecommendRequest(query="snow boots", k=2, use_llm=False))
        await asyncio.sleep(0.2)
        r2 = await svc.recommend(RecommendRequest(query="snow boots", k=2, use_llm=False))
        assert r2.timings["total_ms"] > 0.0  # recomputed, not the frozen copy
        assert r1.request_id != r2.request_id
    finally:
        svc.close()


class _SlowCountingLLM:
    """Planner that outlives any small wait budget, counting real calls."""

    provider = "fake"
    model = "fake"

    def __init__(self, delay=0.3):
        self.delay = delay
        self.calls = 0

    async def complete_json(self, *, system, user, schema, max_tokens=2000, timeout=30.0):
        import asyncio

        self.calls += 1
        await asyncio.sleep(self.delay)
        return PlannerOutput(
            intent="boots",
            slots=[{"name": "boots", "search_query": "snow boots", "keywords": ["boot"]}],
        )


async def test_background_plan_lands_in_semantic_cache_for_paraphrases(
    fixture_index, hash_embedder
):
    import asyncio

    llm = _SlowCountingLLM(delay=0.3)
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env(
            {
                "EMBEDDER": "hash",
                "PLANNER_BUDGET_S": "0.05",
                "SEMANTIC_PLAN_CACHE": "1",
                "SEMANTIC_PLAN_THRESHOLD": "0.5",
            }
        ),
        llm=llm,
    )
    try:
        r1 = await svc.recommend(RecommendRequest(query="warm snow boots", k=2, rerank=False))
        assert r1.llm_info.planner_used == "heuristic"  # the wait was too short
        await asyncio.sleep(0.5)  # the shared call finishes in the background
        # a PARAPHRASE, not the same text: only the semantic cache can serve it
        r2 = await svc.recommend(
            RecommendRequest(query="warm snow boots for me", k=2, rerank=False)
        )
        assert r2.llm_info.planner_used == "llm"
        assert llm.calls == 1  # no second bedrock call
    finally:
        svc.close()


async def test_degraded_response_is_never_frozen_by_the_response_cache(
    fixture_index, hash_embedder
):
    import asyncio

    llm = _SlowCountingLLM(delay=0.3)
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env(
            {"EMBEDDER": "hash", "PLANNER_BUDGET_S": "0.05", "RESPONSE_CACHE_TTL_S": "60"}
        ),
        llm=llm,
    )
    try:
        r1 = await svc.recommend(RecommendRequest(query="snow boots", k=2, rerank=False))
        assert r1.llm_info.planner_used == "heuristic" and r1.warnings
        await asyncio.sleep(0.5)  # background plan lands in the exact cache
        r2 = await svc.recommend(RecommendRequest(query="snow boots", k=2, rerank=False))
        # the fallback answer must not have been pinned for the ttl
        assert r2.served_from_cache is False
        assert r2.llm_info.planner_used == "llm"
        r3 = await svc.recommend(RecommendRequest(query="snow boots", k=2, rerank=False))
        assert r3.served_from_cache is True  # the GOOD answer is the one that got cached
        assert r3.llm_info.planner_used == "llm"
    finally:
        svc.close()


async def test_planner_admission_is_bounded(fixture_index, hash_embedder):
    import asyncio

    from stylist.service import MAX_INFLIGHT_PLANS

    llm = _SlowCountingLLM(delay=0.2)
    svc = RecommendationService(
        fixture_index,
        hash_embedder,
        Settings.from_env({"EMBEDDER": "hash", "PLANNER_BUDGET_S": "0.05"}),
        llm=llm,
    )
    try:
        for i in range(MAX_INFLIGHT_PLANS):
            svc._plan_inflight[("q", str(i))] = asyncio.get_running_loop().create_future()
        r = await svc.recommend(RecommendRequest(query="snow boots", k=2, rerank=False))
        assert r.llm_info.planner_used == "heuristic"
        assert any("planner queue full" in w for w in r.warnings)
        assert llm.calls == 0  # nothing new was admitted
    finally:
        for fut in list(svc._plan_inflight.values()):
            fut.cancel()
        svc._plan_inflight.clear()
        svc.close()
