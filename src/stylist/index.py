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

import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover - windows
    fcntl = None  # type: ignore[assignment]

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
]
CHECKSUMMED = ("embeddings.npy", "row_ids.npy", "catalog.parquet")
BM25_DIR = "bm25"


def index_files(index_dir: Path) -> list[str]:
    """Every artifact that gets a checksum: the three big files plus all bm25 files."""
    names = list(CHECKSUMMED)
    bm25_dir = index_dir / BM25_DIR
    if bm25_dir.is_dir():
        names += sorted(f"{BM25_DIR}/{p.name}" for p in bm25_dir.iterdir() if p.is_file())
    return names


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
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("meta.json is not an object")
            known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
            return cls(**known)
        except (ValueError, TypeError) as e:
            raise IndexValidationError(f"meta.json is unreadable: {e}") from e


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _scratch_dir(final_dir: Path) -> Path:
    """A fresh, uniquely named build directory next to the target (two builds into the
    same target never touch each other's files)."""
    return Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.building-", dir=final_dir.parent))


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
    final_dir = Path(index_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    # build next to the target, swap in at the end: a crash never leaves a half index
    index_dir = _scratch_dir(final_dir)
    try:
        meta = _build_into(
            index_dir, Path(catalog_path), embedder, limit, sampling, seed, batch_size, t0
        )
    except BaseException:
        shutil.rmtree(index_dir, ignore_errors=True)
        raise
    _swap_in(index_dir, final_dir)
    log.info("index built in %.1fs", meta.build_seconds)
    return meta


def _park_name(final_dir: Path) -> Path:
    return final_dir.parent / f".{final_dir.name}.old-{os.getpid()}-{int(time.time() * 1000)}"


@contextlib.contextmanager
def _swap_lock(final_dir: Path):
    """The same lock file the artifact installer uses, so two builders (or a builder and
    an installer) never rename over each other."""
    lock = final_dir.parent / f".{final_dir.name}.lock"
    with open(lock, "a") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _swap_in(built: Path, final_dir: Path) -> None:
    """Replace `final_dir` with `built` under the directory lock. POSIX cannot swap two
    non-empty directories in one step: the old index is parked under a unique name for
    the instant between the two renames; a crash there is reported by `SearchIndex.load`
    (it names the parked copy), and the parked copy is removed on success."""
    with _swap_lock(final_dir):
        if final_dir.exists():
            old_dir = _park_name(final_dir)
            os.replace(final_dir, old_dir)
            os.replace(built, final_dir)
            shutil.rmtree(old_dir, ignore_errors=True)
        else:
            os.replace(built, final_dir)


def _build_into(
    index_dir: Path,
    catalog_path: Path,
    embedder: Embedder,
    limit: int | None,
    sampling: str,
    seed: int,
    batch_size: int,
    t0: float,
) -> IndexMeta:
    df = load_catalog_subset(catalog_path, limit=limit, sampling=sampling, seed=seed)
    texts = df["doc_text"].fillna("").astype(str).tolist()
    n = len(df)
    log.info("indexing %d rows (sampling=%s, limit=%s)", n, sampling, limit)

    truncated = None
    counter = getattr(embedder, "count_tokens_over", None)
    if callable(counter) and n:
        stride = max(1, -(-n // 5000))  # ceil: an even spread over the whole catalog
        sample = texts[::stride][:5000]
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

    meta = IndexMeta(
        pipeline_version=PIPELINE_VERSION,
        embedding_model=embedder.name,
        embedding_revision=getattr(embedder, "revision", None),
        dim=int(embedder.dim),
        n_rows=n,
        sampling=sampling,
        limit=limit,
        seed=seed,
        source_catalog=Path(catalog_path).name,  # the name only, never a local path
        row_ids_sha256=_sha256_bytes(row_ids.tobytes()),
        checksums={name: sha256_file(index_dir / name) for name in index_files(index_dir)},
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "bm25s": bm25s.__version__,
            "sentence_transformers": _dist_version("sentence-transformers"),
            "stylist": _dist_version("fashion-stylist"),
        },
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        build_seconds=round(time.time() - t0, 1),
        truncated_docs_in_sample=truncated,
    )
    (index_dir / "meta.json").write_text(meta.to_json())
    return meta


def verify_checksums(index_dir: Path, meta: IndexMeta) -> None:
    """Every file listed in meta must exist and match; extra/missing bm25 files fail too."""
    expected = set(meta.checksums)
    present = set(index_files(index_dir))
    if not any(name.startswith(f"{BM25_DIR}/") for name in expected):
        raise IndexValidationError("meta.json lists no bm25 files, the index is incomplete")
    if expected != present:
        raise IndexValidationError(
            f"index files do not match meta.json (missing {sorted(expected - present)}, "
            f"unexpected {sorted(present - expected)})"
        )
    for name in sorted(expected):
        if sha256_file(index_dir / name) != meta.checksums[name]:
            raise IndexValidationError(f"checksum mismatch for {name}")


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
            parked = sorted(index_dir.parent.glob(f".{index_dir.name}.old*"))
            if parked and not index_dir.exists():
                raise IndexValidationError(
                    f"no index at {index_dir}, but {parked[-1]} exists: an index rebuild was "
                    f"interrupted between its two renames. Rename it back or rebuild."
                )
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
        verify_checksums(index_dir, meta)

        embeddings = _load_array(index_dir / "embeddings.npy")
        row_ids = _load_array(index_dir / "row_ids.npy")
        if embeddings.ndim != 2:
            raise IndexValidationError("embeddings.npy is not a 2-d array")
        if embeddings.dtype.kind != "f":
            raise IndexValidationError("embeddings.npy is not a float array")
        embeddings = embeddings.astype(np.float32)
        if embeddings.size == 0:
            raise IndexValidationError("embeddings.npy is empty")
        if not np.isfinite(embeddings).all():
            raise IndexValidationError("embeddings.npy contains non-finite values")
        if row_ids.ndim != 1 or row_ids.dtype.kind not in "iu":
            raise IndexValidationError("row_ids.npy is not a 1-d integer array")
        row_ids = row_ids.astype(np.int64, copy=False)
        try:
            catalog = pd.read_parquet(index_dir / "catalog.parquet")
        except Exception as e:  # pyarrow raises several unrelated types
            raise IndexValidationError(f"catalog.parquet is unreadable: {e}") from e
        try:
            bm = bm25s.BM25.load(index_dir / "bm25", show_progress=False)
        except Exception as e:
            raise IndexValidationError(f"bm25 index is unreadable: {e}") from e

        n = embeddings.shape[0]
        if not (n == len(row_ids) == len(catalog) == meta.n_rows):
            raise IndexValidationError(
                f"row count mismatch: embeddings={n} row_ids={len(row_ids)} "
                f"catalog={len(catalog)} meta={meta.n_rows}"
            )
        if embeddings.shape[1] != meta.dim:
            raise IndexValidationError("embedding dim does not match meta")
        missing = [c for c in SERVING_COLUMNS if c not in catalog.columns]
        if missing:
            raise IndexValidationError(f"catalog.parquet lacks serving columns: {missing}")
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
        scores = np.asarray(self._bm25.get_scores(toks), dtype=np.float32)
        if scores.shape != (self.n_rows,):
            raise IndexValidationError(f"bm25 returned {scores.shape}, expected ({self.n_rows},)")
        return scores


def _load_array(path: Path) -> np.ndarray:
    """np.load without pickles; any failure becomes an IndexValidationError naming the file."""
    try:
        return np.load(path, allow_pickle=False)
    except (ValueError, OSError, EOFError) as e:
        raise IndexValidationError(f"{path.name} is unreadable: {e}") from e
