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
