"""Regression tests for the audit round on catalog / index / retrieval / planner."""

import gzip
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from stylist.catalog import (
    build_doc_text,
    derive_audience,
    ingest,
    load_catalog_subset,
    normalize_record,
    parse_price,
    select_rows,
)
from stylist.config import Settings
from stylist.index import IndexValidationError, SearchIndex, build_index
from stylist.planner import PlannerOutput, Slot, SlotWindow, normalize_plan, parse_budget
from stylist.retrieval import Candidate, Retriever, bayes_rating
from stylist.schemas import product_url


def _raw(i, title, rating_number=0, price=None, department=None, **extra):
    rec = {
        "parent_asin": f"B{i:09d}",
        "title": title,
        "average_rating": 4.0,
        "rating_number": rating_number,
        "price": price,
        "store": "Store",
        "features": ["f"],
        "description": [],
        "images": [{"variant": "MAIN", "large": "https://x/1.jpg"}],
        "details": {"Department": department} if department else {},
    }
    rec.update(extra)
    return rec


def _write_raw(path, records, bad_lines=()):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        for b in bad_lines:
            f.write(b + "\n")


# ----------------------------------------------------------------------------- catalog


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("12,99", (1299.0, "string")),  # commas are thousands separators in this (US) dataset
        ("$1,000", (1000.0, "string")),
    ],
)
def test_parse_price_treats_comma_as_thousands_only(raw, expected):
    assert parse_price(raw) == expected


def test_department_babydoll_is_not_baby():
    assert derive_audience("Lace Babydoll Chemise", "babydoll") == "unknown"
    assert derive_audience("x", "unisex-baby") == "baby"


def test_store_is_trimmed_and_only_strings_survive():
    assert normalize_record(_raw(1, "t", store="  Nike  "), 1)["store"] == "Nike"
    assert normalize_record(_raw(1, "t", store=5), 1)["store"] is None
    assert len(normalize_record(_raw(1, "t", store="s" * 300), 1)["store"]) == 80


def test_missing_rating_is_null_not_zero():
    row = normalize_record({"title": "x", "rating_number": 3, "parent_asin": "A"}, row_id=1)
    assert row["average_rating"] is None
    row = normalize_record({"title": "x", "average_rating": float("nan")}, row_id=1)
    assert row["average_rating"] is None
    row = normalize_record({"title": "x", "average_rating": 4.5}, row_id=1)
    assert row["average_rating"] == 4.5


def test_doc_text_is_built_from_the_normalized_row():
    raw = _raw(1, "  Cozy   Sweater ", department="womens", store="  Acme ")
    raw["features"] = ["  Machine   Wash  "]
    raw["details"]["Material"] = " Cotton "
    row = normalize_record(raw, 1)
    assert row["doc_text"] == build_doc_text(row)
    assert row["doc_text"].startswith("Cozy Sweater | Machine Wash | womens, Cotton | Acme")


def test_ingest_skips_and_counts_bad_lines(tmp_path):
    raw = tmp_path / "raw.jsonl.gz"
    _write_raw(raw, [_raw(0, "a"), _raw(1, "b")], bad_lines=["{not json", "[1,2]"])
    stats = ingest(raw, tmp_path / "c.parquet")
    assert stats.rows == 2 and stats.bad_lines == 2
    assert stats.as_dict()["bad_lines"] == 2


def test_ingest_is_atomic_and_refuses_an_empty_catalog(tmp_path):
    raw = tmp_path / "raw.jsonl.gz"
    out = tmp_path / "c.parquet"
    out.write_bytes(b"previous good file")
    _write_raw(raw, [])
    with pytest.raises(ValueError, match="no rows"):
        ingest(raw, out)
    assert out.read_bytes() == b"previous good file"
    assert [p.name for p in tmp_path.iterdir()] == sorted(["raw.jsonl.gz", "c.parquet"]) or set(
        p.name for p in tmp_path.iterdir()
    ) == {"raw.jsonl.gz", "c.parquet"}  # no temp file left behind


def test_load_catalog_subset_reads_row_groups_without_the_whole_table(tmp_path):
    raw = tmp_path / "raw.jsonl.gz"
    _write_raw(raw, [_raw(i, f"item {i}", rating_number=(i * 7) % 11) for i in range(23)])
    out = tmp_path / "c.parquet"
    ingest(raw, out, chunk_size=5)  # several row groups
    assert pq.ParquetFile(out).num_row_groups > 1
    df = pd.read_parquet(out)
    for limit, sampling in ((7, "popular"), (9, "random"), (None, "all")):
        got = load_catalog_subset(out, limit=limit, sampling=sampling)
        want = select_rows(df, limit=limit, sampling=sampling)
        assert list(got["row_id"]) == list(want["row_id"])
        assert list(got["title"]) == list(want["title"])


# ----------------------------------------------------------------------------- schemas


@pytest.mark.parametrize("asin", ["B0ABC12345", "0123456789"])
def test_product_url_for_a_well_formed_asin(asin):
    assert product_url(asin) == f"https://www.amazon.com/dp/{asin}"


@pytest.mark.parametrize("asin", ["", "bad id", "B0ABC1234", "B0ABC12345/../x", "b0abc12345"])
def test_product_url_is_none_for_anything_else(asin):
    assert product_url(asin) is None


# ----------------------------------------------------------------------------- index


def _resign(index_dir):
    """Recompute meta checksums after a deliberate tamper so the load reaches the
    semantic checks (the checksum test would otherwise stop it earlier)."""
    from stylist.index import IndexMeta, _sha256_bytes, index_files, sha256_file

    meta = IndexMeta.from_json((index_dir / "meta.json").read_text())
    meta.checksums = {n: sha256_file(index_dir / n) for n in index_files(index_dir)}
    try:
        meta.row_ids_sha256 = _sha256_bytes(np.load(index_dir / "row_ids.npy").tobytes())
    except ValueError:
        pass  # deliberately unloadable row_ids (the loader itself must reject them)
    (index_dir / "meta.json").write_text(meta.to_json())


def test_load_rejects_non_finite_or_misshaped_embeddings(tmp_path, fixture_catalog, hash_embedder):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=20, sampling="popular")
    emb = np.load(index_dir / "embeddings.npy")
    bad = emb.copy()
    bad[0, 0] = np.nan
    np.save(index_dir / "embeddings.npy", bad)
    _resign(index_dir)
    with pytest.raises(IndexValidationError, match="finite"):
        SearchIndex.load(index_dir)
    np.save(index_dir / "embeddings.npy", emb.reshape(-1))
    _resign(index_dir)
    with pytest.raises(IndexValidationError, match="2-d"):
        SearchIndex.load(index_dir)


def test_load_wraps_unreadable_artifacts_in_a_validation_error(
    tmp_path, fixture_catalog, hash_embedder
):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=20, sampling="popular")
    (index_dir / "embeddings.npy").write_bytes(b"not a numpy file at all")
    _resign(index_dir)
    with pytest.raises(IndexValidationError, match="embeddings.npy"):
        SearchIndex.load(index_dir)
    (index_dir / "meta.json").write_text("{not json")
    with pytest.raises(IndexValidationError, match="meta.json"):
        SearchIndex.load(index_dir)


def test_load_refuses_pickled_arrays(tmp_path, fixture_catalog, hash_embedder):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=20, sampling="popular")
    arr = np.empty(20, dtype=object)
    arr[:] = ["x"] * 20
    np.save(index_dir / "row_ids.npy", arr, allow_pickle=True)
    _resign(index_dir)
    with pytest.raises(IndexValidationError, match="row_ids.npy"):
        SearchIndex.load(index_dir)


def test_meta_records_the_catalog_basename_and_library_versions(
    tmp_path, fixture_catalog, hash_embedder
):
    meta = build_index(
        fixture_catalog, tmp_path / "idx", hash_embedder, limit=10, sampling="popular"
    )
    assert "/" not in meta.source_catalog and meta.source_catalog == fixture_catalog.name
    assert {"python", "numpy", "pandas", "bm25s", "stylist"} <= set(meta.versions)


def test_failed_build_leaves_no_scratch_dir_and_keeps_the_old_index(
    tmp_path, fixture_catalog, hash_embedder
):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=10, sampling="popular")

    class Boom(type(hash_embedder)):
        def encode_docs(self, texts, batch_size=64):
            raise RuntimeError("gpu fell over")

    with pytest.raises(RuntimeError, match="gpu fell over"):
        build_index(fixture_catalog, index_dir, Boom(dim=256), limit=10, sampling="popular")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["idx"]
    assert SearchIndex.load(index_dir).n_rows == 10


def test_two_builds_into_the_same_target_do_not_share_a_scratch_dir(
    tmp_path, fixture_catalog, hash_embedder
):
    """The scratch dir name must be unique per build (a second build must not rmtree the
    first build's half-written files)."""
    from stylist import index as index_mod

    names = []
    real_mkdir = index_mod._scratch_dir

    def spy(final_dir):
        d = real_mkdir(final_dir)
        names.append(d.name)
        return d

    index_mod._scratch_dir = spy
    try:
        build_index(fixture_catalog, tmp_path / "idx", hash_embedder, limit=5, sampling="popular")
        build_index(fixture_catalog, tmp_path / "idx", hash_embedder, limit=5, sampling="popular")
    finally:
        index_mod._scratch_dir = real_mkdir
    assert len(names) == 2 and names[0] != names[1]


def test_load_explains_an_interrupted_swap(tmp_path, fixture_catalog, hash_embedder):
    index_dir = tmp_path / "idx"
    build_index(fixture_catalog, index_dir, hash_embedder, limit=10, sampling="popular")
    index_dir.rename(tmp_path / ".idx.old")  # what a crash between the two renames leaves
    with pytest.raises(IndexValidationError, match="interrupted"):
        SearchIndex.load(index_dir)


# ----------------------------------------------------------------------------- retrieval


def test_bayes_rating_without_a_rating_is_the_prior():
    assert bayes_rating(None, 0, m=20, prior=4.2) == pytest.approx(4.2)
    assert bayes_rating(None, 50, m=20, prior=4.2) == pytest.approx(4.2)
    assert bayes_rating(5.0, 0, m=20, prior=4.2) == pytest.approx(4.2)


def _retriever(fixture_index, hash_embedder, **env):
    return Retriever(fixture_index, hash_embedder, Settings.from_env({"EMBEDDER": "hash", **env}))


def test_hydrate_keeps_a_missing_rating_as_none(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    old = fixture_index.catalog.loc[0, "average_rating"]
    fixture_index.catalog.loc[0, "average_rating"] = float("nan")
    try:
        c = r._hydrate(Candidate(0, 0, 0.0))
        assert c.average_rating is None
    finally:
        fixture_index.catalog.loc[0, "average_rating"] = old
        r._group_keys.clear()


def test_group_key_is_memoised_per_row(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    c = r._hydrate(Candidate(3, 0, 0.0))
    assert r._group_keys[3] == c.group_key
    again = r._hydrate(Candidate(3, 0, 0.0))
    assert again.group_key == c.group_key


def test_slot_candidates_report_eligible_rows_separately(fixture_index, hash_embedder):
    from stylist.planner import QueryPlan

    r = _retriever(fixture_index, hash_embedder)
    plan = QueryPlan(
        intent="t",
        slots=[Slot(name="b", search_query="boots", keywords=["boot"])],
        source="heuristic",
    )
    [res] = r.retrieve(plan, [SlotWindow(None, 5.0, None, False)], n_candidates=10, k=4)
    assert res.eligible_rows == int((fixture_index.catalog["price"] <= 5.0).sum())
    assert res.n_eligible <= 10 and res.n_eligible <= res.eligible_rows


def test_exact_audience_match_outranks_unknown_audience_on_a_tie(fixture_index, hash_embedder):
    from stylist.planner import QueryPlan

    r = _retriever(fixture_index, hash_embedder)
    cat = fixture_index.catalog
    # two rows with the same text: the later one (worse tie-break) gets the requested audience
    i0, i1 = 0, 1
    saved = cat.loc[[i0, i1], ["title", "audience"]].copy()
    emb_saved = fixture_index.embeddings[[i0, i1]].copy()
    try:
        for i in (i0, i1):
            cat.loc[i, "title"] = "zzq test item"
        cat.loc[i0, "audience"] = "unknown"
        cat.loc[i1, "audience"] = "women"
        fixture_index._column_cache.clear()
        v = hash_embedder.encode_docs(["zzq test item"])[0]
        fixture_index.embeddings[i0] = v
        fixture_index.embeddings[i1] = v
        r._group_keys.clear()
        plan = QueryPlan(
            intent="t", slots=[Slot(name="x", search_query="zzq test item")], source="heuristic"
        )
        [res] = r.retrieve(plan, [SlotWindow(None, None, "women", False)], n_candidates=10, k=4)
        top_two = [c.idx for c in res.candidates[:2]]
        assert top_two[0] == i1, top_two
    finally:
        cat.loc[[i0, i1], ["title", "audience"]] = saved.to_numpy()
        fixture_index._column_cache.clear()
        fixture_index.embeddings[[i0, i1]] = emb_saved
        r._group_keys.clear()


# ----------------------------------------------------------------------------- planner


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a coat under $1,000", (None, 1000.0)),
        ("between $1,200 and $2,000", (1200.0, 2000.0)),
        ("at least 1,500 dollars", (1500.0, None)),
    ],
)
def test_parse_budget_handles_thousands_separators(text, expected):
    assert parse_budget(text) == expected


def test_normalize_sanitizes_free_text_fields():
    out = PlannerOutput(
        intent="  beach \n\n outfit \x00 for   a trip " + "x" * 400,
        occasion="wedding\tguest\n" + "y" * 100,
        season="summer\r\n",
        slots=[{"name": " top \n", "search_query": "linen\n\nshirt  "}],
    )
    plan = normalize_plan(out, "q")
    assert "\n" not in plan.intent and "\x00" not in plan.intent and len(plan.intent) <= 200
    assert plan.occasion == ("wedding guest " + "y" * 100)[:60].strip()
    assert plan.season == "summer"
    assert plan.slots[0].name == "top" and plan.slots[0].search_query == "linen shirt"


def test_normalize_drops_absurd_budgets():
    out = PlannerOutput(
        budget_max=5e12, budget_scope="per_item", slots=[{"name": "a", "search_query": "a"}]
    )
    plan = normalize_plan(out, "q")
    assert plan.budget_max is None and plan.slots[0].budget_max is None
    assert any("budget" in w for w in plan.warnings)


@pytest.mark.parametrize(
    "allocs", [[1, 1, 1], [90, 5, 5], [0, 0, 100], [33.333, 33.333, 33.334], [0.01, 200, 0.01]]
)
def test_total_budget_invariants_hold_for_any_allocation(allocs):
    out = PlannerOutput(
        budget_max=100.0,
        budget_scope="total",
        slots=[
            {"name": f"s{i}", "search_query": f"q{i}", "budget_max": a}
            for i, a in enumerate(allocs)
        ],
    )
    plan = normalize_plan(out, "q")
    per = [s.budget_max for s in plan.slots]
    assert all(p is not None and p >= 10.0 for p in per)  # 10% floor of 100
    assert sum(per) <= 100.0 + 1e-6
