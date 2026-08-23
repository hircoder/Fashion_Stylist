"""Embedding backends behind one tiny protocol.

`SentenceTransformerEmbedder` is the real thing (bge-small by default). `HashEmbedder`
is a deterministic bag-of-hashed-ngrams vectorizer that needs no model download, it
exists so the test-suite and a fully offline demo can run in seconds. It is lexical, so
do not expect semantic matches from it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from stylist.config import Settings

# bge-style models want an instruction in front of the *query* only
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int
    query_prefix: str

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class HashEmbedder:
    """Deterministic hashed unigram+bigram vectors. For tests and offline smoke runs."""

    _token_rx = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256):
        self.name = "hash"
        self.dim = dim
        self.query_prefix = ""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = self._token_rx.findall(text.lower())
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:], strict=False)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=4).digest(), "little")
            sign = 1.0 if (h >> 31) & 1 else -1.0
            v[h % self.dim] += sign
        return v

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.stack([self._vec(t) for t in texts]))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_docs(texts)


class SentenceTransformerEmbedder:
    """sentence-transformers wrapper with normalized float32 output."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        revision: str | None = None,
        device: str | None = None,
        max_seq_length: int = 256,
    ):
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self.name = model_name
        self.revision = revision
        self._model = SentenceTransformer(model_name, revision=revision, device=device)
        self._model.max_seq_length = max_seq_length
        self.max_seq_length = max_seq_length
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension  # sentence-transformers < 6
        )
        self.dim = int(getter())
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_docs([self.query_prefix + t for t in texts], batch_size=32)

    def count_tokens_over(self, texts: list[str], limit: int) -> int:
        """How many texts would be truncated at `limit` tokens (build-time diagnostic)."""
        tok = self._model.tokenizer
        return sum(1 for t in texts if len(tok(t, add_special_tokens=True)["input_ids"]) > limit)


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        revision=settings.embedding_revision,
        device=settings.embed_device,
        max_seq_length=settings.max_seq_length,
    )
