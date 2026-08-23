"""Bedrock adapter: the same complete_json contract over the Converse API.

Structured output is one forced tool whose input schema is the pydantic model's json
schema; every model on Bedrock that does tool use (Nova, Claude, Mistral) then returns
the object as the tool input, no free-text json parsing anywhere. boto3 is sync, so the
call runs in a thread and the caller's timeout wraps it.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from stylist.llm import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransportError,
    LLMTruncatedError,
    LLMValidationError,
    record_attempt,
    record_failure,
    record_usage,
)

T = TypeVar("T", bound=BaseModel)

# ClientError codes worth telling apart; everything else is transport
_AUTH = {"AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException"}
_RATE = {"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"}
_TIMEOUT = {"ModelTimeoutException"}


class BedrockLLM:
    provider = "bedrock"

    def __init__(
        self,
        model: str,
        region: str | None = None,
        effort: str | None = None,  # accepted for interface parity, bedrock has no knob
        latency_optimized: bool = False,
        client: Any = None,
    ):
        self.model = model
        self.effort = effort
        self._latency_optimized = latency_optimized
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(read_timeout=60, connect_timeout=5, retries={"max_attempts": 1}),
            )
        self._client = client

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 2000,
        timeout: float = 30.0,
    ) -> T:
        record_attempt()
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._converse, system, user, schema, max_tokens),
                timeout=timeout,
            )
        except TimeoutError as exc:
            record_failure()
            raise LLMTimeoutError(f"bedrock call exceeded {timeout}s") from exc
        except LLMTimeoutError:
            record_failure()
            raise
        except (LLMAuthError, LLMRateLimitError, LLMTruncatedError, LLMValidationError):
            record_failure()
            raise
        except LLMTransportError:
            record_failure()
            raise

    def _converse(self, system: str, user: str, schema: type[T], max_tokens: int) -> T:
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "emit",
                            "description": "Return the structured answer.",
                            "inputSchema": {"json": schema.model_json_schema()},
                        }
                    }
                ],
                # "any" = the model must call a tool; with one tool that is our schema.
                # ("tool" pinning is not supported by every bedrock model family.)
                "toolChoice": {"any": {}},
            },
        }
        if self._latency_optimized:
            kwargs["performanceConfig"] = {"latency": "optimized"}
        try:
            resp = self._client.converse(**kwargs)
        except Exception as exc:  # botocore ClientError and friends
            code = ""
            if hasattr(exc, "response"):
                code = ((getattr(exc, "response", None) or {}).get("Error") or {}).get("Code", "")
            name = type(exc).__name__
            if code in _AUTH or name == "NoCredentialsError":
                raise LLMAuthError(f"{code or name}") from exc
            if code in _RATE:
                raise LLMRateLimitError(f"{code}") from exc
            if code in _TIMEOUT or "Timeout" in name:
                raise LLMTimeoutError(f"{code or name}") from exc
            raise LLMTransportError(f"{code or name}: {exc}") from exc

        usage = resp.get("usage") or {}
        record_usage(usage.get("inputTokens"), usage.get("outputTokens"))
        if resp.get("stopReason") == "max_tokens":
            raise LLMTruncatedError("output hit maxTokens")
        content = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        tool_use = next((c["toolUse"] for c in content if "toolUse" in c), None)
        if tool_use is None:
            raise LLMValidationError("model returned no tool call")
        try:
            return schema.model_validate(tool_use.get("input"))
        except ValidationError as exc:
            raise LLMValidationError(str(exc)) from exc
