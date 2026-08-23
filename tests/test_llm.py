import anthropic
import openai
import pytest
from pydantic import BaseModel

from stylist.config import Settings
from stylist.llm import (
    FakeLLM,
    LLMAuthError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMTransportError,
    LLMTruncatedError,
    LLMValidationError,
    make_llm_client,
)
from stylist.llm.anthropic_client import AnthropicLLM
from stylist.llm.openai_client import OpenAILLM


class Answer(BaseModel):
    colour: str
    count: int


async def test_fake_llm_returns_validated_model_and_records_call():
    llm = FakeLLM(responses=[{"colour": "red", "count": 2}])
    out = await llm.complete_json(system="sys", user="usr", schema=Answer)
    assert out == Answer(colour="red", count=2)
    assert llm.calls[0]["system"] == "sys" and llm.calls[0]["schema"] is Answer


async def test_fake_llm_raises_configured_exception():
    llm = FakeLLM(responses=[LLMRateLimitError("slow down")])
    with pytest.raises(LLMRateLimitError):
        await llm.complete_json(system="s", user="u", schema=Answer)


async def test_fake_llm_invalid_dict_is_a_validation_error():
    llm = FakeLLM(responses=[{"colour": "red"}])
    with pytest.raises(LLMValidationError):
        await llm.complete_json(system="s", user="u", schema=Answer)


def test_make_llm_client_none_when_no_provider():
    assert make_llm_client(Settings.from_env({})) is None


def test_make_llm_client_builds_provider_adapters():
    a = make_llm_client(Settings.from_env({"ANTHROPIC_API_KEY": "k"}))
    assert isinstance(a, AnthropicLLM) and a.model == "claude-opus-5"
    o = make_llm_client(Settings.from_env({"OPENAI_API_KEY": "k", "LLM_MODEL": "gpt-5-mini"}))
    assert isinstance(o, OpenAILLM) and o.model == "gpt-5-mini"


# ---- anthropic adapter contract (SDK client replaced by a recorder) ----


class _Resp:
    def __init__(self, stop_reason="end_turn", parsed=None):
        self.stop_reason = stop_reason
        self.parsed_output = parsed


class _AnthropicRecorder:
    def __init__(self, result=None, raises=None):
        self.kwargs = None
        self._result = result
        self._raises = raises
        self.messages = self

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        if self._raises:
            raise self._raises
        return self._result


async def test_anthropic_adapter_sends_schema_and_returns_parsed_output():
    rec = _AnthropicRecorder(result=_Resp(parsed=Answer(colour="blue", count=1)))
    llm = AnthropicLLM(api_key="k", model="claude-opus-5", client=rec)
    out = await llm.complete_json(system="S", user="U", schema=Answer, max_tokens=321, timeout=9)
    assert out == Answer(colour="blue", count=1)
    assert rec.kwargs["model"] == "claude-opus-5"
    assert rec.kwargs["system"] == "S"
    assert rec.kwargs["messages"] == [{"role": "user", "content": "U"}]
    assert rec.kwargs["output_format"] is Answer
    assert rec.kwargs["max_tokens"] == 321
    assert rec.kwargs["timeout"] == 9


async def test_anthropic_adapter_maps_refusal_and_truncation():
    llm = AnthropicLLM(api_key="k", model="m", client=_AnthropicRecorder(result=_Resp("refusal")))
    with pytest.raises(LLMRefusalError):
        await llm.complete_json(system="S", user="U", schema=Answer)
    rec = _AnthropicRecorder(result=_Resp("max_tokens"))
    llm = AnthropicLLM(api_key="k", model="m", client=rec)
    with pytest.raises(LLMTruncatedError):
        await llm.complete_json(system="S", user="U", schema=Answer)


async def test_anthropic_adapter_unparsed_output_is_validation_error():
    llm = AnthropicLLM(api_key="k", model="m", client=_AnthropicRecorder(result=_Resp(parsed=None)))
    with pytest.raises(LLMValidationError):
        await llm.complete_json(system="S", user="U", schema=Answer)


def _sdk_exc(cls):
    # SDK errors need an http response to construct; isinstance is all the adapter checks
    return cls.__new__(cls)


@pytest.mark.parametrize(
    "sdk_exc,mapped",
    [
        (anthropic.AuthenticationError, LLMAuthError),
        (anthropic.RateLimitError, LLMRateLimitError),
        (anthropic.APITimeoutError, LLMTimeoutError),
        (anthropic.APIConnectionError, LLMTransportError),
        (anthropic.InternalServerError, LLMTransportError),
    ],
)
async def test_anthropic_adapter_maps_sdk_errors(sdk_exc, mapped):
    llm = AnthropicLLM(api_key="k", model="m", client=_AnthropicRecorder(raises=_sdk_exc(sdk_exc)))
    with pytest.raises(mapped):
        await llm.complete_json(system="S", user="U", schema=Answer)


# ---- openai adapter contract ----


class _Msg:
    def __init__(self, parsed=None, refusal=None):
        self.parsed = parsed
        self.refusal = refusal


class _Choice:
    def __init__(self, parsed=None, refusal=None, finish_reason="stop"):
        self.message = _Msg(parsed, refusal)
        self.finish_reason = finish_reason


class _OAResp:
    def __init__(self, choice):
        self.choices = [choice]


class _OpenAIRecorder:
    def __init__(self, result=None, raises=None):
        self.kwargs = None
        self._result = result
        self._raises = raises
        self.chat = self
        self.completions = self

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        if self._raises:
            raise self._raises
        return self._result


async def test_openai_adapter_sends_schema_and_returns_parsed_output():
    rec = _OpenAIRecorder(result=_OAResp(_Choice(parsed=Answer(colour="green", count=3))))
    llm = OpenAILLM(api_key="k", model="gpt-5-mini", client=rec)
    out = await llm.complete_json(system="S", user="U", schema=Answer, max_tokens=222, timeout=7)
    assert out == Answer(colour="green", count=3)
    assert rec.kwargs["model"] == "gpt-5-mini"
    assert rec.kwargs["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    assert rec.kwargs["response_format"] is Answer
    assert rec.kwargs["max_completion_tokens"] == 222
    assert rec.kwargs["timeout"] == 7


async def test_openai_adapter_maps_refusal_and_length():
    rec = _OpenAIRecorder(result=_OAResp(_Choice(refusal="no")))
    with pytest.raises(LLMRefusalError):
        await OpenAILLM("k", "m", client=rec).complete_json(system="S", user="U", schema=Answer)
    rec = _OpenAIRecorder(result=_OAResp(_Choice(finish_reason="length")))
    with pytest.raises(LLMTruncatedError):
        await OpenAILLM("k", "m", client=rec).complete_json(system="S", user="U", schema=Answer)


@pytest.mark.parametrize(
    "sdk_exc,mapped",
    [
        (openai.AuthenticationError, LLMAuthError),
        (openai.RateLimitError, LLMRateLimitError),
        (openai.APITimeoutError, LLMTimeoutError),
        (openai.APIConnectionError, LLMTransportError),
        (openai.InternalServerError, LLMTransportError),
    ],
)
async def test_openai_adapter_maps_sdk_errors(sdk_exc, mapped):
    llm = OpenAILLM("k", "m", client=_OpenAIRecorder(raises=_sdk_exc(sdk_exc)))
    with pytest.raises(mapped):
        await llm.complete_json(system="S", user="U", schema=Answer)


@pytest.mark.live
async def test_live_provider_roundtrip():
    settings = Settings.from_env()
    llm = make_llm_client(settings)
    if llm is None:
        pytest.skip("no LLM configured")
    out = await llm.complete_json(
        system="Answer with JSON only.",
        user="The sky colour and the number of wheels on a bicycle.",
        schema=Answer,
        max_tokens=4000,
        timeout=60,
    )
    assert out.count == 2


async def test_openai_adapter_maps_sdk_parse_exceptions():
    # the sdk raises these from parse() before any response object exists
    for exc_cls, mapped in [
        (openai.LengthFinishReasonError, LLMTruncatedError),
        (openai.ContentFilterFinishReasonError, LLMRefusalError),
    ]:
        llm = OpenAILLM("k", "m", client=_OpenAIRecorder(raises=_sdk_exc(exc_cls)))
        with pytest.raises(mapped):
            await llm.complete_json(system="S", user="U", schema=Answer)


async def test_adapters_map_any_other_exception_to_an_llm_error():
    from stylist.llm import LLMError

    for llm in (
        OpenAILLM("k", "m", client=_OpenAIRecorder(raises=KeyError("weird"))),
        AnthropicLLM(api_key="k", model="m", client=_AnthropicRecorder(raises=ValueError("odd"))),
    ):
        with pytest.raises(LLMError):
            await llm.complete_json(system="S", user="U", schema=Answer)
