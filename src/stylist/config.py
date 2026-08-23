"""Runtime settings, read from environment variables (and a local .env if present).

Kept as a plain frozen dataclass on purpose: no pydantic-settings, nothing magic,
every knob is listed here once with its default so it is easy to see what can be tuned.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROVIDERS = ("anthropic", "openai", "none")
DEFAULT_MODELS = {"anthropic": "claude-opus-5", "openai": "gpt-5-mini"}


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
    anthropic_api_key: str | None
    anthropic_base_url: str | None
    openai_api_key: str | None
    openai_base_url: str | None
    llm_effort: str | None  # low | medium | high for reasoning models, None = provider default

    # deadlines (seconds)
    request_deadline_s: float
    planner_budget_s: float
    rerank_budget_s: float

    # ranking knobs
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

    log_level: str

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

        return cls(
            data_dir=data_dir,
            raw_path=raw_path,
            processed_path=processed_path,
            index_dir=index_dir,
            embedding_model=_get_str(env, "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") or "",
            embedding_revision=_get_str(env, "EMBEDDING_REVISION"),
            embedder=embedder,
            embed_device=_get_str(env, "EMBED_DEVICE"),
            max_seq_length=_get_int(env, "MAX_SEQ_LENGTH", 256),
            llm_provider=provider,
            llm_model=model,
            anthropic_api_key=anthropic_key,
            anthropic_base_url=_get_str(env, "ANTHROPIC_BASE_URL"),
            openai_api_key=openai_key,
            openai_base_url=_get_str(env, "OPENAI_BASE_URL"),
            llm_effort=_get_str(env, "LLM_EFFORT", "low"),
            request_deadline_s=_get_float(env, "REQUEST_DEADLINE_S", 25.0),
            planner_budget_s=_get_float(env, "PLANNER_BUDGET_S", 10.0),
            rerank_budget_s=_get_float(env, "RERANK_BUDGET_S", 12.0),
            top_n_per_channel=_get_int(env, "TOP_N_PER_CHANNEL", 100),
            rrf_k=_get_int(env, "RRF_K", 60),
            keyword_boost=_get_float(env, "KEYWORD_BOOST", 0.5),
            quality_weight=_get_float(env, "QUALITY_WEIGHT", 0.1),
            rerank_candidates=_get_int(env, "RERANK_CANDIDATES", 15),
            retrieval_concurrency=_get_int(env, "RETRIEVAL_CONCURRENCY", 4),
            plan_cache_size=_get_int(env, "PLAN_CACHE_SIZE", 256),
            index_url=_get_str(env, "INDEX_URL"),
            index_sha256=_get_str(env, "INDEX_SHA256"),
            index_max_bytes=_get_int(env, "INDEX_MAX_BYTES", 4 * 1024**3),
            log_level=(_get_str(env, "LOG_LEVEL", "INFO") or "INFO").upper(),
        )
