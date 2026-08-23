"""OpenAI adapter: chat completions with a pydantic `response_format`.

Works against api.openai.com and against OpenAI-compatible endpoints that implement
`/chat/completions` with json_schema response formats (set OPENAI_BASE_URL).
"""

from __future__ import annotations

from typing import Any, TypeVar

import openai
from pydantic import BaseModel

from stylist.llm import (
    LLMAuthError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMTransportError,
    LLMTruncatedError,
    LLMValidationError,
)

T = TypeVar("T", bound=BaseModel)

_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAILLM:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        effort: str | None = "low",
        client: Any | None = None,
    ):
        self.model = model
        self.effort = effort
        self._client = client or openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _is_reasoning_model(self) -> bool:
        return self.model.lower().startswith(_REASONING_PREFIXES)

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 2000,
        timeout: float = 30.0,
    ) -> T:
        kwargs: dict[str, Any] = {}
        if self.effort and self._is_reasoning_model():
            kwargs["reasoning_effort"] = self.effort
        try:
            resp = await self._client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=schema,
                max_completion_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthError(str(exc)) from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            raise LLMTransportError(str(exc)) from exc

        if not getattr(resp, "choices", None):
            raise LLMValidationError("empty choices in response")
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMRefusalError(str(choice.message.refusal))
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMTruncatedError("output hit max_completion_tokens")
        parsed = getattr(choice.message, "parsed", None)
        if parsed is None:
            raise LLMValidationError("no parsed output in response")
        return parsed
