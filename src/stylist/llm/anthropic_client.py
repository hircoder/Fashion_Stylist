"""Anthropic adapter: Messages API with structured output (`messages.parse`)."""

from __future__ import annotations

from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from stylist.llm import (
    LLMAuthError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMTransportError,
    LLMTruncatedError,
    LLMValidationError,
    record_usage,
)

T = TypeVar("T", bound=BaseModel)


class AnthropicLLM:
    provider = "anthropic"

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
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)

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
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            resp = await self._client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                timeout=timeout,
                **kwargs,
            )
        except anthropic.AuthenticationError as exc:
            raise LLMAuthError(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            raise LLMTransportError(str(exc)) from exc
        except ValidationError as exc:
            raise LLMValidationError(str(exc)) from exc
        except Exception as exc:  # anything else the sdk throws: still an llm failure
            raise LLMTransportError(f"{type(exc).__name__}: {exc}") from exc

        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            raise LLMRefusalError("model refused the request")
        if stop == "max_tokens":
            raise LLMTruncatedError("output hit max_tokens")
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            raise LLMValidationError("no parsed output in response")
        usage = getattr(resp, "usage", None)
        record_usage(getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
        return parsed
