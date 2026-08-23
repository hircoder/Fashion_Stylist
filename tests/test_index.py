import json

import numpy as np
import pytest

from stylist.index import IndexValidationError, SearchIndex, build_index


@pytest.fixture(scope="module")
def built_index(tmp_path_factory, fixture_catalog, hash_embedder):
    index_dir = tmp_path_factory.mktemp("index") / "index"
    meta = build_index(fixture_catalog, index_dir, hash_embedder, limit=None, sampling="all")
    return index_dir, meta


def test_build_writes_all_artifacts_and_meta(built_index, fixture_catalog_df):
    index_dir, meta = built_index
    for name in ("embeddings.npy", "row_ids.npy", "catalog.parquet", "meta.json"):
        assert (index_dir / name).exists(), name
    assert (index_dir / "bm25").is_dir()
    assert meta.n_rows == len(fixture_catalog_df)
    assert meta.embedding_model == "hash"
    saved = json.loads((index_dir / "meta.json").read_text())
    assert saved["n_rows"] == meta.n_rows
    assert {"embeddings.npy", "row_ids.npy", "catalog.parquet"} <= set(saved["checksums"])
    assert any(name.startswith("bm25/") for name in saved["checksums"])


def test_build_with_popular_limit_keeps_row_order(tmp_path, fixture_catalog, hash_embedder):
    index_dir = tmp_path / "idx"
    meta = build_index(fixture_catalog, index_dir, hash_embedder, limit=50, sampling="popular")
    idx = SearchIndex.load(index_dir)
    assert meta.n_rows == 50 and idx.n_rows == 50
    assert list(idx.row_ids) == sorted(idx.row_ids)
    assert list(idx.catalog["row_id"]) == list(idx.row_ids)
    assert idx.meta.sampling == "popular" and idx.meta.limit == 50


def test_loaded_index_scores_have_expected_shapes(built_index, hash_embedder):
    index_dir, _ = built_index
    idx = SearchIndex.load(index_dir)
    q = hash_embedder.encode_queries(["snow boots", "beach sandals"])
    dense = idx.dense_scores(q)
    assert dense.shape == (2, idx.n_rows) and dense.dtype == np.float32
    lex = idx.bm25_scores("snow boots")
    assert lex.shape == (idx.n_rows,)


def test_bm25_ranks_a_boot_listing_first(built_index):
    index_dir, _ = built_index
    idx = SearchIndex.load(index_dir)
    scores = idx.bm25_scores("snow boots")
    top_title = idx.catalog.iloc[int(np.argmax(scores))]["title"].lower()
    assert "boot" in top_title


def test_dense_scores_find_lexical_match_with_hash_embedder(built_index, hash_embedder):
    index_dir, _ = built_index
    idx = SearchIndex.load(index_dir)
    scores = idx.dense_scores(hash_embedder.encode_queries(["swim trunks"]))[0]
    top_title = idx.catalog.iloc[int(np.argmax(scores))]["title"].lower()
    assert "swim" in top_title or "trunks" in top_title


def test_load_rejects_model_mismatch(built_index):
    index_dir, _ = built_index
    with pytest.raises(IndexValidationError, match="embedding model"):
        SearchIndex.load(index_dir, expected_model="BAAI/bge-small-en-v1.5")


def test_load_rejects_tampered_embeddings(tmp_path, fixture_catalog, hash_embedder):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=30, sampling="popular")
    emb = np.load(index_dir / "embeddings.npy")
    np.save(index_dir / "embeddings.npy", emb[:-1])
    with pytest.raises(IndexValidationError):
        SearchIndex.load(index_dir)


def test_load_rejects_row_id_misalignment(tmp_path, fixture_catalog, hash_embedder):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=30, sampling="popular")
    ids = np.load(index_dir / "row_ids.npy")
    ids[0], ids[1] = ids[1], ids[0]
    np.save(index_dir / "row_ids.npy", ids)
    with pytest.raises(IndexValidationError):
        SearchIndex.load(index_dir)


def test_load_rejects_missing_directory(tmp_path):
    with pytest.raises(IndexValidationError):
        SearchIndex.load(tmp_path / "nope")


def test_load_rejects_tampered_bm25(tmp_path, fixture_catalog, hash_embedder):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=30, sampling="popular")
    vocab = index_dir / "bm25" / "vocab.index.json"
    vocab.write_text(vocab.read_text() + "\n")  # any byte change must be caught
    with pytest.raises(IndexValidationError, match="bm25"):
        SearchIndex.load(index_dir)


def test_build_replaces_an_existing_index_atomically(tmp_path, fixture_catalog, hash_embedder):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=30, sampling="popular")
    (index_dir / "stale.txt").write_text("old file that must not survive a rebuild")
    build_index(fixture_catalog, index_dir, hash_embedder, limit=20, sampling="popular")
    assert SearchIndex.load(index_dir).n_rows == 20
    assert not (index_dir / "stale.txt").exists()
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name != ".idx.lock")
    assert leftovers == ["idx"]  # no temp dirs left behind (the swap lock file stays)
