"""Build and load the search index.

One directory holds everything the service needs at query time, so a deployment
only ships this folder:

    embeddings.npy   float16 (n, dim), L2 normalised, loaded as float32
    row_ids.npy      int64 (n,)  catalog row_id for each index row
    catalog.parquet  the indexed rows only (serving columns, same order)
    bm25/            bm25s index over the same doc text
    meta.json        versions, counts, checksums, how the subset was picked

`SearchIndex.load` refuses to serve anything whose counts, row order or checksums do
not line up. Silent misalignment between an embedding row and a product row is the
worst kind of bug in a retrieval system, so we pay the hashing cost at startup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

from stylist.catalog import PIPELINE_VERSION, load_catalog_subset
from stylist.embeddings import Embedder

log = logging.getLogger(__name__)

SERVING_COLUMNS = [
    "row_id",
    "parent_asin",
    "title",
    "average_rating",
    "rating_number",
    "price",
    "store",
    "features",
    "description",
    "department",
    "brand",
    "material",
    "style",
    "color",
    "pattern",
    "age_range",
    "image_url",
    "audience",
    "group_key",
]
CHECKSUMMED = ("embeddings.npy", "row_ids.npy", "catalog.parquet")


class IndexValidationError(RuntimeError):
    """The index directory is missing, incomplete, or does not line up with itself."""


@dataclass
class IndexMeta:
    pipeline_version: str
    embedding_model: str
    embedding_revision: str | None
    dim: int
    n_rows: int
    sampling: str
    limit: int | None
    seed: int
    source_catalog: str
    row_ids_sha256: str
    checksums: dict[str, str]
    versions: dict[str, str]
    built_at: str
    build_seconds: float
    truncated_docs_in_sample: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> IndexMeta:
        data = json.loads(text)
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords="en", return_ids=False, show_progress=False)


def build_index(
    catalog_path: Path,
    index_dir: Path,
    embedder: Embedder,
    limit: int | None = 100_000,
    sampling: str = "popular",
    seed: int = 42,
    batch_size: int = 128,
) -> IndexMeta:
    """Select rows from the catalog, embed them, build BM25, write the index directory."""
    t0 = time.time()
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    df = load_catalog_subset(catalog_path, limit=limit, sampling=sampling, seed=seed)
    texts = df["doc_text"].fillna("").astype(str).tolist()
    n = len(df)
    log.info("indexing %d rows (sampling=%s, limit=%s)", n, sampling, limit)

    truncated = None
    counter = getattr(embedder, "count_tokens_over", None)
    if callable(counter) and n:
        sample = texts[:: max(1, n // 5000)][:5000]
        truncated = int(counter(sample, getattr(embedder, "max_seq_length", 256)))
        log.info("%d of %d sampled docs exceed the token limit (truncated)", truncated, len(sample))

    emb = np.zeros((n, embedder.dim), dtype=np.float16)
    step = max(batch_size * 8, 1024)
    for start in range(0, n, step):
        emb[start : start + step] = embedder.encode_docs(texts[start : start + step], batch_size)
        if start and start % (step * 20) == 0:
            log.info("embedded %d / %d", start, n)
    np.save(index_dir / "embeddings.npy", emb)

    row_ids = df["row_id"].to_numpy(dtype=np.int64)
    np.save(index_dir / "row_ids.npy", row_ids)

    bm = bm25s.BM25()
    bm.index(_tokenize(texts), show_progress=False)
    bm.save(index_dir / "bm25", show_progress=False)

    serving = df[SERVING_COLUMNS].reset_index(drop=True)
    serving.to_parquet(index_dir / "catalog.parquet", index=False)

    import sentence_transformers

    meta = IndexMeta(
        pipeline_version=PIPELINE_VERSION,
        embedding_model=embedder.name,
        embedding_revision=getattr(embedder, "revision", None),
        dim=int(embedder.dim),
        n_rows=n,
        sampling=sampling,
        limit=limit,
        seed=seed,
        source_catalog=str(catalog_path),
        row_ids_sha256=_sha256_bytes(row_ids.tobytes()),
        checksums={name: sha256_file(index_dir / name) for name in CHECKSUMMED},
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "bm25s": bm25s.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        build_seconds=round(time.time() - t0, 1),
        truncated_docs_in_sample=truncated,
    )
    (index_dir / "meta.json").write_text(meta.to_json())
    log.info("index built in %.1fs", meta.build_seconds)
    return meta


class SearchIndex:
    """In-memory dense + lexical index over one catalog subset."""

    def __init__(
        self,
        meta: IndexMeta,
        catalog: pd.DataFrame,
        embeddings: np.ndarray,
        row_ids: np.ndarray,
        bm25: bm25s.BM25,
    ):
        self.meta = meta
        self.catalog = catalog
        self.embeddings = embeddings
        self.row_ids = row_ids
        self._bm25 = bm25
        self.n_rows = int(embeddings.shape[0])

    @classmethod
    def load(cls, index_dir: Path, expected_model: str | None = None) -> SearchIndex:
        index_dir = Path(index_dir)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            raise IndexValidationError(f"no index at {index_dir} (meta.json missing)")
        meta = IndexMeta.from_json(meta_path.read_text())
        if meta.pipeline_version != PIPELINE_VERSION:
            raise IndexValidationError(
                f"index pipeline_version {meta.pipeline_version} != code "
                f"{PIPELINE_VERSION}, rebuild the index"
            )
        if expected_model and meta.embedding_model != expected_model:
            raise IndexValidationError(
                f"index embedding model {meta.embedding_model!r} != configured {expected_model!r}"
            )
        for name in CHECKSUMMED:
            path = index_dir / name
            if not path.exists():
                raise IndexValidationError(f"{name} missing from {index_dir}")
            got = sha256_file(path)
            if got != meta.checksums.get(name):
                raise IndexValidationError(f"checksum mismatch for {name}")

        embeddings = np.load(index_dir / "embeddings.npy").astype(np.float32)
        row_ids = np.load(index_dir / "row_ids.npy")
        catalog = pd.read_parquet(index_dir / "catalog.parquet")
        bm = bm25s.BM25.load(index_dir / "bm25", show_progress=False)

        n = embeddings.shape[0]
        if not (n == len(row_ids) == len(catalog) == meta.n_rows):
            raise IndexValidationError(
                f"row count mismatch: embeddings={n} row_ids={len(row_ids)} "
                f"catalog={len(catalog)} meta={meta.n_rows}"
            )
        if embeddings.shape[1] != meta.dim:
            raise IndexValidationError("embedding dim does not match meta")
        if _sha256_bytes(row_ids.tobytes()) != meta.row_ids_sha256:
            raise IndexValidationError("row_ids hash mismatch")
        if not np.array_equal(catalog["row_id"].to_numpy(dtype=np.int64), row_ids):
            raise IndexValidationError("catalog row order does not match row_ids")
        bm_docs = bm.scores.get("num_docs") if isinstance(bm.scores, dict) else None
        if bm_docs is not None and int(bm_docs) != n:
            raise IndexValidationError(f"bm25 doc count {bm_docs} != {n}")
        return cls(meta, catalog, embeddings, row_ids, bm)

    def dense_scores(self, query_vecs: np.ndarray) -> np.ndarray:
        """Cosine scores for every index row, all queries in one matmul. Shape (q, n)."""
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        return (q @ self.embeddings.T).astype(np.float32)

    def bm25_scores(self, query: str) -> np.ndarray:
        """BM25 score for every index row. Shape (n,). Zeros when no token matches."""
        toks = _tokenize([query])[0]
        if not toks:
            return np.zeros(self.n_rows, dtype=np.float32)
        return np.asarray(self._bm25.get_scores(toks), dtype=np.float32)
