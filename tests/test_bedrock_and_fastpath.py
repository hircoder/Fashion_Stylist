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
