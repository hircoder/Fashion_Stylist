import numpy as np
import pandas as pd

from stylist.config import Settings
from stylist.planner import QueryPlan, Slot, SlotWindow
from stylist.retrieval import (
    Retriever,
    bayes_rating,
    diversify_by_group,
    eligibility_masks,
    keyword_matches,
    rrf_fuse,
)


class _Idx:
    """Minimal stand-in for SearchIndex: just the catalog columns masks look at."""

    def __init__(self, rows):
        self.catalog = pd.DataFrame(rows)
        self.n_rows = len(rows)


def _rows():
    return [
        {"audience": "women", "price": 20.0},
        {"audience": "men", "price": 20.0},
        {"audience": "unisex", "price": None},
        {"audience": "unknown", "price": 500.0},
        {"audience": "women", "price": None},
    ]


def test_masks_without_any_window_accept_everything():
    elig, pool = eligibility_masks(_Idx(_rows()), SlotWindow(None, None, None, False))
    assert elig.all() and not pool.any()


def test_masks_audience_lets_unisex_and_unknown_through():
    elig, _ = eligibility_masks(_Idx(_rows()), SlotWindow(None, None, "women", False))
    assert elig.tolist() == [True, False, True, True, True]


def test_masks_price_window_is_strict_and_unpriced_goes_to_pool_only_when_allowed():
    elig, pool = eligibility_masks(_Idx(_rows()), SlotWindow(None, 50.0, None, False))
    assert elig.tolist() == [True, True, False, False, False]
    assert not pool.any()
    elig, pool = eligibility_masks(_Idx(_rows()), SlotWindow(None, 50.0, None, True))
    assert elig.tolist() == [True, True, False, False, False]
    assert pool.tolist() == [False, False, True, False, True]


def test_masks_min_price_excludes_cheap_known_prices():
    elig, _ = eligibility_masks(_Idx(_rows()), SlotWindow(100.0, None, None, False))
    assert elig.tolist() == [False, False, False, True, False]


def test_rrf_prefers_items_ranked_by_both_channels_and_respects_mask():
    dense = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
    bm25 = np.array([0.0, 5.0, 4.0, 9.0], dtype=np.float32)
    mask = np.array([True, True, True, False])
    fused = rrf_fuse(dense, bm25, mask, top_n=10, k=60)
    order = [c.idx for c in fused]
    assert order[0] == 1  # top-ish in both
    assert 3 not in order  # masked out even though bm25 loves it
    by_idx = {c.idx: c for c in fused}
    assert by_idx[0].bm25_rank is None and by_idx[0].dense_rank == 0
    assert by_idx[1].score > by_idx[0].score


def test_rrf_skips_zero_bm25_scores():
    dense = np.array([0.5, 0.4], dtype=np.float32)
    bm25 = np.zeros(2, dtype=np.float32)
    fused = rrf_fuse(dense, bm25, np.array([True, True]), top_n=10, k=60)
    assert all(c.bm25_rank is None for c in fused)


def test_keyword_matches_are_word_bounded_and_allow_plural():
    assert keyword_matches("Women's Flat Sandals Brown", ["sandal", "boot"]) == ["sandal"]
    assert keyword_matches("Sandalwood Candle", ["sandal"]) == []
    assert keyword_matches("Mens Flip Flops", ["flip flop"]) == ["flip flop"]


def test_bayes_rating_pulls_low_count_items_toward_the_prior():
    strong = bayes_rating(4.8, 500, m=20, prior=4.0)
    weak = bayes_rating(5.0, 1, m=20, prior=4.0)
    assert strong > weak
    assert abs(weak - 4.05) < 0.01


def test_diversify_keeps_highest_scored_row_per_group():
    from stylist.retrieval import Candidate

    def cand(idx, score, group):
        return Candidate(idx=idx, row_id=idx, score=score, group_key=group, title=f"t{idx}")

    out = diversify_by_group([cand(1, 0.9, "a"), cand(2, 0.8, "a"), cand(3, 0.7, "b")], seen=set())
    assert [c.idx for c in out] == [1, 3]


def _plan(*slots):
    return QueryPlan(intent="t", slots=list(slots), source="heuristic")


def _retriever(fixture_index, hash_embedder, **env):
    return Retriever(fixture_index, hash_embedder, Settings.from_env({"EMBEDDER": "hash", **env}))


def test_retriever_returns_boot_listings_for_a_boots_slot(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    plan = _plan(Slot(name="boots", search_query="snow boots", keywords=["boot", "boots"]))
    [res] = r.retrieve(plan, [SlotWindow(None, None, None, False)], n_candidates=10, k=4)
    titles = [c.title.lower() for c in res.candidates]
    assert titles and "boot" in titles[0]
    assert res.candidates[0].matched_keywords
    assert res.n_eligible == len(res.candidates)


def test_retriever_applies_audience_window(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    plan = _plan(Slot(name="x", search_query="sandals", keywords=["sandal"]))
    [res] = r.retrieve(plan, [SlotWindow(None, None, "men", False)], n_candidates=20, k=4)
    assert all(c.audience in ("men", "unisex", "unknown") for c in res.candidates)


def test_retriever_backfills_unpriced_when_allowed(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    plan = _plan(Slot(name="x", search_query="swimsuit", keywords=["swimsuit"]))
    strict = r.retrieve(plan, [SlotWindow(None, 5.0, None, False)], n_candidates=10, k=4)[0]
    assert all(c.price is not None and c.price <= 5.0 for c in strict.candidates)
    relaxed = r.retrieve(plan, [SlotWindow(None, 5.0, None, True)], n_candidates=10, k=4)[0]
    assert len(relaxed.candidates) >= len(strict.candidates)
    backfilled = [c for c in relaxed.candidates if not c.in_window]
    assert backfilled and all(c.price is None for c in backfilled)
    assert len(backfilled) > 4 - relaxed.n_eligible  # a real pool, not just k - eligible
    in_window = [c.row_id for c in relaxed.candidates if c.in_window]
    assert in_window == [c.row_id for c in strict.candidates]  # same priced items, same order
    assert any("price" in w for w in relaxed.warnings)


def test_retriever_batches_all_slots_in_one_call(fixture_index, hash_embedder, monkeypatch):
    r = _retriever(fixture_index, hash_embedder)
    calls = []
    orig = fixture_index.dense_scores

    def spy(q):
        calls.append(q.shape)
        return orig(q)

    monkeypatch.setattr(fixture_index, "dense_scores", spy)
    plan = _plan(Slot(name="a", search_query="boots"), Slot(name="b", search_query="hat"))
    out = r.retrieve(plan, [SlotWindow(None, None, None, False)] * 2, n_candidates=5, k=3)
    assert len(out) == 2 and calls == [(2, hash_embedder.dim)]


def test_rrf_disabled_channel_contributes_nothing():
    dense = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    bm25 = np.array([1.0, 5.0, 4.0], dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    only_bm25 = rrf_fuse(None, bm25, mask, top_n=10, k=60)
    assert [c.idx for c in only_bm25] == [1, 2, 0]
    assert all(c.dense_rank is None for c in only_bm25)
    only_dense = rrf_fuse(dense, None, mask, top_n=10, k=60)
    assert [c.idx for c in only_dense] == [0, 1, 2]


def test_channels_setting_disables_dense(fixture_index, hash_embedder, monkeypatch):
    r = _retriever(fixture_index, hash_embedder, CHANNELS="bm25")
    calls = []
    monkeypatch.setattr(fixture_index, "dense_scores", lambda q: calls.append(1))
    plan = _plan(Slot(name="boots", search_query="snow boots", keywords=["boot"]))
    [res] = r.retrieve(plan, [SlotWindow(None, None, None, False)], n_candidates=5, k=3)
    assert calls == [] and res.candidates and all(c.dense_rank is None for c in res.candidates)


def test_retriever_expands_depth_when_top_n_is_full_of_variants(fixture_index, hash_embedder):
    # with a tiny per-channel top-n, variants of one product can fill the whole window;
    # the retriever must keep digging until it has k distinct groups
    r = _retriever(fixture_index, hash_embedder, TOP_N_PER_CHANNEL="2")
    plan = _plan(Slot(name="x", search_query="women dress", keywords=["dress"]))
    [res] = r.retrieve(plan, [SlotWindow(None, None, None, False)], n_candidates=6, k=6)
    assert len(res.candidates) == 6
    assert len({c.group_key for c in res.candidates}) == 6


def test_hydrate_turns_nan_rating_into_zero(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    fixture_index.catalog.loc[0, "average_rating"] = float("nan")
    try:
        c = r._hydrate(__import__("stylist.retrieval", fromlist=["Candidate"]).Candidate(0, 0, 0.0))
        assert c.average_rating == 0.0
    finally:
        fixture_index.catalog.loc[0, "average_rating"] = 4.0


def test_pool_items_are_merged_by_score_with_a_small_in_window_bonus(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    plan = _plan(Slot(name="x", search_query="swimsuit", keywords=["swimsuit"]))
    [res] = r.retrieve(plan, [SlotWindow(None, 5.0, None, True)], n_candidates=10, k=4)
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)  # one order, not two partitions
    assert any(not c.in_window for c in res.candidates)
    assert any(c.in_window for c in res.candidates)
