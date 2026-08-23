"""Second council round: reranker admissibility, retrieval ordering, api limits, index checks."""

import pytest

from stylist.config import Settings
from stylist.planner import QueryPlan, Slot, SlotWindow
from stylist.reranker import SlotRerankOutput, apply_slot_rerank
from stylist.retrieval import Candidate, Retriever, SlotCandidates, type_match


def _cand(i, title, typed, matched=None, rating=4.0):
    c = Candidate(
        idx=i, row_id=i, score=1.0 - i / 100, title=title, average_rating=rating, rating_number=10
    )
    c.type_match = typed
    c.matched_keywords = list(matched or (["boot"] if typed else []))
    return c


def _sc(cands, keywords=("boot",)):
    slot = Slot(name="boots", search_query="boots", keywords=list(keywords))
    return SlotCandidates(
        slot, SlotWindow(None, None, None, False), cands, len(cands), [], eligible_rows=len(cands)
    )


def test_off_type_picks_are_rejected_when_enough_typed_candidates_exist():
    cands = [
        _cand(1, "Snow Boots", True),
        _cand(2, "Ball Pump", False),
        _cand(3, "Rain Boots", True),
        _cand(4, "Hiking Boots", True),
    ]
    out = SlotRerankOutput(
        picks=[
            {"row_id": 2, "reason": "nice", "evidence": ["title"]},
            {"row_id": 1, "reason": "warm", "evidence": ["title"]},
        ]
    )
    ranked, warnings = apply_slot_rerank(out, _sc(cands), k=2)
    assert [c.row_id for c in ranked.ordered][:2] == [1, 3]
    assert any("2" in w and "not the product type" in w for w in warnings)


def test_off_type_pick_is_kept_when_nothing_typed_exists():
    cands = [_cand(1, "Ball Pump", False), _cand(2, "Air Mattress", False)]
    out = SlotRerankOutput(picks=[{"row_id": 2, "reason": "closest", "evidence": ["title"]}])
    ranked, _ = apply_slot_rerank(out, _sc(cands), k=2)
    assert [c.row_id for c in ranked.ordered] == [2]


def test_no_good_match_with_picks_keeps_the_picks_and_warns():
    cands = [_cand(1, "Snow Boots", True), _cand(2, "Rain Boots", True)]
    out = SlotRerankOutput(
        picks=[{"row_id": 2, "reason": "r", "evidence": ["title"]}], no_good_match=True
    )
    ranked, warnings = apply_slot_rerank(out, _sc(cands), k=2)
    assert ranked.ordered[0].row_id == 2
    assert any("no_good_match" in w for w in warnings)


def test_reasons_are_link_stripped_word_capped_and_evidence_checked():
    cands = [_cand(1, "Snow Boots", True, rating=None)]
    long_reason = "visit http://evil.example now " + "word " * 40
    out = SlotRerankOutput(
        picks=[{"row_id": 1, "reason": long_reason, "evidence": ["title", "rating", "price"]}]
    )
    ranked, _ = apply_slot_rerank(out, _sc(cands), k=1)
    reason = ranked.reasons[1]
    assert "http" not in reason.reason and "evil" not in reason.reason
    assert len(reason.reason.split()) <= 20
    assert reason.evidence == ["title"]  # no rating, no price on this candidate


@pytest.mark.parametrize(
    "title,keywords,expected",
    [
        ("Shoe Insoles Arch Support", ["shoe"], False),
        ("Jacket Hanger Set", ["jacket"], False),
        ("Boot Laces 2 Pack", ["boot", "boots"], False),
        ("Shoe Laces Flat", ["laces", "shoe laces"], True),  # the slot asks for laces
        ("Leather Boots", ["boot"], True),
    ],
)
def test_accessory_veto_applies_to_exact_keywords_too(title, keywords, expected):
    assert type_match(title, keywords) is expected


def _retriever(fixture_index, hash_embedder):
    return Retriever(fixture_index, hash_embedder, Settings.from_env({"EMBEDDER": "hash"}))


def test_type_gate_threshold_is_k_not_candidate_depth(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    plan = QueryPlan(
        intent="t",
        slots=[Slot(name="x", search_query="boots", keywords=["boot", "boots"])],
        source="llm",
    )
    [res] = r.retrieve(plan, [SlotWindow(None, None, None, False)], n_candidates=40, k=3)
    typed = [c for c in res.candidates if c.type_match]
    assert len(typed) >= 3 and all(
        c.type_match for c in res.candidates
    )  # never padded with off-type rows


def test_brand_order_is_typed_own_then_two_untyped_own_then_typed_others(
    fixture_index, hash_embedder
):
    cat = fixture_index.catalog
    r = _retriever(fixture_index, hash_embedder)
    rows = [i for i in cat.index if "boot" not in str(cat.loc[i, "title"]).lower()][:8]
    saved = cat.loc[rows, "store"].copy()
    try:
        cat.loc[rows, "store"] = "Zebrabrand"
        getattr(fixture_index, "_column_cache", {}).clear()
        plan = QueryPlan(
            intent="t",
            brand="zebrabrand",
            slots=[Slot(name="boots", search_query="boots", keywords=["boot", "boots"])],
            source="llm",
        )
        [res] = r.retrieve(plan, [SlotWindow(None, None, None, False)], n_candidates=10, k=4)
        own = [c for c in res.candidates if c.store == "Zebrabrand"]
        assert len(own) <= 2
        others = [c for c in res.candidates if c.store != "Zebrabrand"]
        assert others and all(c.type_match for c in others)
        first_other = next(i for i, c in enumerate(res.candidates) if c.store != "Zebrabrand")
        assert first_other <= 2
    finally:
        cat.loc[rows, "store"] = saved
        getattr(fixture_index, "_column_cache", {}).clear()
        r._brand_masks.clear()


def test_features_null_cell_is_an_empty_list(fixture_index, hash_embedder):
    r = _retriever(fixture_index, hash_embedder)
    cat = fixture_index.catalog
    saved = cat.at[0, "features"]
    cat.at[0, "features"] = float("nan")
    try:
        assert r._hydrate(Candidate(0, 0, 0.0)).features == []
    finally:
        cat.at[0, "features"] = saved
