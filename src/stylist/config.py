"""Runtime settings, read from environment variables (and a local .env if present).

Kept as a plain frozen dataclass on purpose: no pydantic-settings, nothing magic,
every knob is listed here once with its default so it is easy to see what can be tuned.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROVIDERS = ("anthropic", "openai", "bedrock", "none")
# the anthropic default is the model every number in docs/ was measured with; set
# LLM_MODEL to move (claude-sonnet-5, claude-opus-5, ...) and re-run the evaluation
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5-mini",
    "bedrock": "us.amazon.nova-micro-v1:0",  # the fast planner; see docs/aws-latency.md
}
# commit of BAAI/bge-small-en-v1.5 the indexes are built with; a build and a runtime must agree
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


class ConfigError(ValueError):
    """Raised when the environment describes a configuration that cannot work."""


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key} must be true or false, got {raw!r}")


def _get_str(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


@dataclass(frozen=True)
class Settings:
    # paths
    data_dir: Path
    raw_path: Path
    processed_path: Path
    index_dir: Path

    # embeddings
    embedding_model: str
    embedding_revision: str | None
    embedder: str  # "sentence-transformers" or "hash" (offline, tests only)
    embed_device: str | None
    max_seq_length: int

    # llm
    llm_provider: str
    llm_model: str | None
    llm_rerank_model: str | None  # a cheaper model for the per-slot rerank calls, else llm_model
    bedrock_region: str | None  # region for LLM_PROVIDER=bedrock (default: the boto3 chain)
    bedrock_latency_optimized: bool  # pass performanceConfig latency=optimized on converse
    semantic_plan_cache: bool  # reuse the plan of the nearest cached query (cosine)
    semantic_plan_threshold: float
    rerank_default: bool  # what rerank does when the request does not say
    response_cache_ttl_s: float  # 0 = off; identical requests within the ttl share a response
    anthropic_api_key: str | None
    anthropic_base_url: str | None
    openai_api_key: str | None
    openai_base_url: str | None
    llm_effort: str | None  # low | medium | high for reasoning models, None = provider default

    # deadlines (seconds)
    request_deadline_s: float
    planner_budget_s: float  # how long a request WAITS for the plan
    planner_call_timeout_s: float  # how long the shared planner call itself may run
    rerank_budget_s: float

    # ranking knobs
    channels: tuple[str, ...]  # which retrieval channels fuse: ("dense", "bm25")
    top_n_per_channel: int
    rrf_k: int
    keyword_boost: float  # in units of RRF(rank 1) = 1/(rrf_k+1)
    quality_weight: float  # same units
    rerank_candidates: int
    retrieval_concurrency: int
    plan_cache_size: int

    # optional prebuilt index download (Railway style deployments)
    index_url: str | None
    index_sha256: str | None
    index_max_bytes: int
    index_allow_file_url: bool  # file:// urls are for tests and local dev only
    index_allow_private_url: bool  # loopback / private / link-local INDEX_URL hosts

    # operational limits for a public deployment
    llm_concurrency: int  # llm calls in flight across all requests
    planner_failure_ttl_s: (
        float  # do not retry the llm planner for a query this soon after a failure
    )
    cors_allow_origins: tuple[str, ...]  # empty = same origin only
    rate_limit_per_minute: int  # per client ip
    max_inflight_requests: int
    max_body_bytes: int
    log_queries: bool
    startup_fail_fast: bool
    trust_proxy_headers: bool  # read the client ip from x-forwarded-for (behind a proxy only)

    log_level: str

    def __repr__(self) -> str:  # never leak keys through a repr in a log or traceback
        hidden = {"anthropic_api_key", "openai_api_key"}
        fields = ", ".join(
            f"{k}={'***' if k in hidden and v else v!r}" for k, v in self.__dict__.items()
        )
        return f"Settings({fields})"

    @property
    def embedding_name(self) -> str:
        """Name recorded in index meta for the configured embedder (must match at load)."""
        return "hash" if self.embedder == "hash" else self.embedding_model

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from a mapping (defaults to os.environ merged over .env)."""
        if env is None:
            merged: dict[str, str] = {
                k: v for k, v in dotenv_values(".env").items() if v is not None
            }
            merged.update(os.environ)
            env = merged

        data_dir = Path(_get_str(env, "DATA_DIR", "data") or "data")
        raw_path = Path(
            _get_str(env, "RAW_PATH", str(data_dir / "raw" / "meta_Amazon_Fashion.jsonl.gz")) or ""
        )
        processed_path = Path(
            _get_str(env, "PROCESSED_PATH", str(data_dir / "processed" / "catalog.parquet")) or ""
        )
        index_dir = Path(_get_str(env, "INDEX_DIR", str(data_dir / "index")) or "")

        anthropic_key = _get_str(env, "ANTHROPIC_API_KEY")
        openai_key = _get_str(env, "OPENAI_API_KEY")
        provider = (_get_str(env, "LLM_PROVIDER") or "").lower()
        if provider == "":
            # documented precedence when nothing explicit is set
            if anthropic_key:
                provider = "anthropic"
            elif openai_key:
                provider = "openai"
            else:
                provider = "none"
        if provider not in PROVIDERS:
            raise ConfigError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
        if provider == "anthropic" and not anthropic_key:
            raise ConfigError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        if provider == "openai" and not openai_key:
            raise ConfigError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")

        model = _get_str(env, "LLM_MODEL")
        if provider == "none":
            model = None
        elif model is None:
            model = DEFAULT_MODELS[provider]

        embedder = (_get_str(env, "EMBEDDER", "sentence-transformers") or "").lower()
        if embedder not in ("sentence-transformers", "hash"):
            raise ConfigError("EMBEDDER must be 'sentence-transformers' or 'hash'")

        settings = cls(
            data_dir=data_dir,
            raw_path=raw_path,
            processed_path=processed_path,
            index_dir=index_dir,
            embedding_model=_get_str(env, "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") or "",
            embedding_revision=_get_str(
                env,
                "EMBEDDING_REVISION",
                # the pin belongs to the default model only; a different model must bring
                # its own revision (or none) rather than inherit a foreign commit hash
                DEFAULT_EMBEDDING_REVISION
                if (_get_str(env, "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") or "")
                == "BAAI/bge-small-en-v1.5"
                else None,
            ),
            embedder=embedder,
            embed_device=_get_str(env, "EMBED_DEVICE"),
            max_seq_length=_get_int(env, "MAX_SEQ_LENGTH", 256),
            llm_provider=provider,
            llm_model=model,
            llm_rerank_model=_get_str(env, "LLM_RERANK_MODEL") if model else None,
            bedrock_region=_get_str(env, "BEDROCK_REGION"),
            bedrock_latency_optimized=_get_bool(env, "BEDROCK_LATENCY_OPTIMIZED", False),
            semantic_plan_cache=_get_bool(env, "SEMANTIC_PLAN_CACHE", False),
            semantic_plan_threshold=_get_float(env, "SEMANTIC_PLAN_THRESHOLD", 0.92),
            rerank_default=_get_bool(env, "RERANK_DEFAULT", True),
            response_cache_ttl_s=_get_float(env, "RESPONSE_CACHE_TTL_S", 0.0),
            anthropic_api_key=anthropic_key,
            anthropic_base_url=_get_str(env, "ANTHROPIC_BASE_URL"),
            openai_api_key=openai_key,
            openai_base_url=_get_str(env, "OPENAI_BASE_URL"),
            llm_effort=_get_str(env, "LLM_EFFORT", "low"),
            request_deadline_s=_get_float(env, "REQUEST_DEADLINE_S", 40.0),
            planner_budget_s=_get_float(env, "PLANNER_BUDGET_S", 15.0),
            planner_call_timeout_s=_get_float(env, "PLANNER_CALL_TIMEOUT_S", 20.0),
            rerank_budget_s=_get_float(env, "RERANK_BUDGET_S", 20.0),
            channels=tuple(
                c.strip().lower()
                for c in (_get_str(env, "CHANNELS", "dense,bm25") or "").split(",")
                if c.strip()
            ),
            top_n_per_channel=_get_int(env, "TOP_N_PER_CHANNEL", 100),
            rrf_k=_get_int(env, "RRF_K", 60),
            keyword_boost=_get_float(env, "KEYWORD_BOOST", 0.5),
            quality_weight=_get_float(env, "QUALITY_WEIGHT", 0.1),
            rerank_candidates=_get_int(env, "RERANK_CANDIDATES", 10),
            retrieval_concurrency=_get_int(env, "RETRIEVAL_CONCURRENCY", 4),
            plan_cache_size=_get_int(env, "PLAN_CACHE_SIZE", 256),
            index_url=_get_str(env, "INDEX_URL"),
            index_sha256=_get_str(env, "INDEX_SHA256"),
            index_max_bytes=_get_int(env, "INDEX_MAX_BYTES", 4 * 1024**3),
            index_allow_file_url=_get_bool(env, "INDEX_ALLOW_FILE_URL", False),
            index_allow_private_url=_get_bool(env, "INDEX_ALLOW_PRIVATE_URL", False),
            llm_concurrency=_get_int(env, "LLM_CONCURRENCY", 8),
            planner_failure_ttl_s=_get_float(env, "PLANNER_FAILURE_TTL_S", 30.0),
            cors_allow_origins=tuple(
                o.strip()
                for o in (_get_str(env, "CORS_ALLOW_ORIGINS", "") or "").split(",")
                if o.strip()
            ),
            rate_limit_per_minute=_get_int(env, "RATE_LIMIT_PER_MINUTE", 60),
            max_inflight_requests=_get_int(env, "MAX_INFLIGHT_REQUESTS", 16),
            max_body_bytes=_get_int(env, "MAX_BODY_BYTES", 16 * 1024),
            log_queries=_get_bool(env, "LOG_QUERIES", False),
            startup_fail_fast=_get_bool(env, "STARTUP_FAIL_FAST", False),
            trust_proxy_headers=_get_bool(env, "TRUST_PROXY_HEADERS", False),
            log_level=(_get_str(env, "LOG_LEVEL", "INFO") or "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Range checks; a bad knob should fail at startup, not divide by zero at request time."""
        positive = {
            "request_deadline_s": self.request_deadline_s,
            "planner_budget_s": self.planner_budget_s,
            "planner_call_timeout_s": self.planner_call_timeout_s,
            "rerank_budget_s": self.rerank_budget_s,
            "top_n_per_channel": self.top_n_per_channel,
            "rrf_k": self.rrf_k,
            "rerank_candidates": self.rerank_candidates,
            "retrieval_concurrency": self.retrieval_concurrency,
            "max_seq_length": self.max_seq_length,
            "index_max_bytes": self.index_max_bytes,
            "llm_concurrency": self.llm_concurrency,
            "max_body_bytes": self.max_body_bytes,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name.upper()} must be a positive finite number, got {value}")
        for name, value in {
            "keyword_boost": self.keyword_boost,
            "quality_weight": self.quality_weight,
            "plan_cache_size": self.plan_cache_size,
            "planner_failure_ttl_s": self.planner_failure_ttl_s,
            "rate_limit_per_minute": self.rate_limit_per_minute,  # 0 = off
            "response_cache_ttl_s": self.response_cache_ttl_s,  # 0 = off
            "max_inflight_requests": self.max_inflight_requests,  # 0 = no cap
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"{name.upper()} must be finite and >= 0, got {value}")
        if not (0.5 <= self.semantic_plan_threshold <= 1.0):
            raise ConfigError(
                "SEMANTIC_PLAN_THRESHOLD must be between 0.5 and 1.0, "
                f"got {self.semantic_plan_threshold}"
            )
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigError(
                f"LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL, got {self.log_level}"
            )
        if not self.channels or any(c not in ("dense", "bm25") for c in self.channels):
            raise ConfigError("CHANNELS must be a comma list drawn from: dense, bm25")
        if self.llm_effort and self.llm_effort not in ("low", "medium", "high"):
            raise ConfigError("LLM_EFFORT must be low, medium or high")


def configure_logging(level: str) -> None:
    """One logging setup shared by the CLI and the API factory (idempotent)."""
    import logging

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
