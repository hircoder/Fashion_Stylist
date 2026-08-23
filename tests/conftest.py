from pathlib import Path

import pandas as pd
import pytest

from stylist.catalog import ingest
from stylist.embeddings import HashEmbedder

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "sample_500.jsonl.gz"


@pytest.fixture(scope="session")
def fixture_catalog(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("catalog") / "catalog.parquet"
    ingest(FIXTURE_RAW, out)
    return out


@pytest.fixture(scope="session")
def fixture_catalog_df(fixture_catalog) -> pd.DataFrame:
    return pd.read_parquet(fixture_catalog)


@pytest.fixture(scope="session")
def hash_embedder() -> HashEmbedder:
    return HashEmbedder(dim=256)


@pytest.fixture(scope="session")
def fixture_index(tmp_path_factory, fixture_catalog, hash_embedder):
    from stylist.index import SearchIndex, build_index

    index_dir = tmp_path_factory.mktemp("index") / "index"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=None, sampling="all")
    return SearchIndex.load(index_dir)


@pytest.fixture(scope="session")
def index_tar(tmp_path_factory, fixture_catalog, hash_embedder):
    """A 40 row index packed the way a deployment ships it, plus its sha256."""
    import hashlib
    import tarfile

    from stylist.index import build_index

    root = tmp_path_factory.mktemp("src")
    build_index(fixture_catalog, root / "index", hash_embedder, limit=40, sampling="popular")
    tar_path = root / "index.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(root / "index", arcname="index")
    return tar_path, hashlib.sha256(tar_path.read_bytes()).hexdigest()
