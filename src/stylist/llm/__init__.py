"""One small async protocol for "give me a JSON object that matches this pydantic model".

Two real adapters (Anthropic, OpenAI) and a FakeLLM for tests. Every provider error is
mapped onto the LLMError family so the pipeline can decide what to do (retry, skip the
stage, fall back to the heuristic planner) without knowing which SDK is underneath.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from stylist.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for anything that goes wrong talking to the model."""


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRefusalError(LLMError):
    pass


class LLMTruncatedError(LLMError):
    pass


class LLMValidationError(LLMError):
    """The model answered, but not with something that fits the schema."""


class LLMTransportError(LLMError):
    """Connection problems, 5xx, anything else we can only retry or give up on."""


@dataclass
class Usage:
    """Token counts for one request (plan + every rerank call), filled in by the adapters."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)


_usage_var: contextvars.ContextVar[Usage | None] = contextvars.ContextVar("llm_usage", default=None)


@contextlib.contextmanager
def usage_scope() -> Iterator[Usage]:
    """Collect the usage of every LLM call made inside the block (and in tasks spawned from
    it: context variables propagate into asyncio tasks)."""
    usage = Usage()
    token = _usage_var.set(usage)
    try:
        yield usage
    finally:
        _usage_var.reset(token)


def record_usage(input_tokens: int | None, output_tokens: int | None) -> None:
    usage = _usage_var.get()
    if usage is not None:
        usage.add(input_tokens, output_tokens)


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 2000,
        timeout: float = 30.0,
    ) -> T: ...


class FakeLLM:
    """Scripted client for tests: each call pops the next response (model, dict or exception)."""

    provider = "fake"
    model = "fake"

    def __init__(
        self,
        responses: list[Any] | None = None,
        handler: Callable[[str, str, type[BaseModel]], Any] | None = None,
        usage: tuple[int, int] = (0, 0),
    ):
        self._responses = list(responses or [])
        self._handler = handler
        self._usage = usage
        self.calls: list[dict[str, Any]] = []

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 2000,
        timeout: float = 30.0,
    ) -> T:
        self.calls.append(
            {"system": system, "user": user, "schema": schema, "max_tokens": max_tokens}
        )
        if self._handler is not None:
            item = self._handler(system, user, schema)
        elif self._responses:
            item = self._responses.pop(0)
        else:
            raise LLMTransportError("FakeLLM has no scripted response left")
        if isinstance(item, Exception):
            raise item
        record_usage(*self._usage)
        if isinstance(item, BaseModel):
            return item  # type: ignore[return-value]
        try:
            return schema.model_validate(item)
        except ValidationError as exc:
            raise LLMValidationError(str(exc)) from exc


class ThrottledLLM:
    """Wraps any client with a global concurrency cap (one semaphore for all requests)."""

    def __init__(self, inner: LLMClient, semaphore: asyncio.Semaphore):
        self._inner = inner
        self._sem = semaphore
        self.provider = inner.provider
        self.model = inner.model

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 2000,
        timeout: float = 30.0,
    ) -> T:
        async with self._sem:
            return await self._inner.complete_json(
                system=system, user=user, schema=schema, max_tokens=max_tokens, timeout=timeout
            )


def make_llm_client(settings: Settings) -> LLMClient | None:
    """Build the configured provider adapter, or None when the service runs without an LLM."""
    if settings.llm_provider == "none" or not settings.llm_model:
        return None
    if settings.llm_provider == "anthropic":
        from stylist.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM(
            api_key=settings.anthropic_api_key or "",
            model=settings.llm_model,
            base_url=settings.anthropic_base_url,
            effort=settings.llm_effort,
        )
    if settings.llm_provider == "openai":
        from stylist.llm.openai_client import OpenAILLM

        return OpenAILLM(
            api_key=settings.openai_api_key or "",
            model=settings.llm_model,
            base_url=settings.openai_base_url,
            effort=settings.llm_effort,
        )
    raise ValueError(f"unknown provider {settings.llm_provider}")
