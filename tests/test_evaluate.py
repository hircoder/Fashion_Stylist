import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import _match_slot_rule, _passes  # noqa: E402


def test_slot_rules_match_on_whole_words_not_substrings():
    rules = {"hat": {"any": ["hat"]}, "boots": {"any": ["boot"]}}
    assert _match_slot_rule("sun hat", rules) is rules["hat"]
    assert _match_slot_rule("what to wear", rules) is None
    assert _match_slot_rule("hiking boots", rules) is rules["boots"]


def test_passes_checks_any_and_none_words():
    rule = {"any": ["sandal", "flip flop"], "none": ["sock"]}
    assert _passes("Women's Flat Sandals", rule)
    assert not _passes("Sandal Socks", rule)
    assert not _passes("Running Shoes", rule)


def test_passes_is_word_bounded_with_plurals():
    rule = {"any": ["hat", "ear"], "none": ["sock"]}
    assert _passes("Wide Brim Sun Hat", rule)
    assert _passes("Knit Hats for Women", rule)
    assert not _passes("What to Wear Chat Shirt", rule)  # 'hat' inside 'what' / 'chat'
    assert not _passes("Pearl Earrings Wearable", rule)  # 'ear' inside 'pearl' / 'wear'
    assert not _passes("Hat Socks", rule)  # none wins, plural


def test_passes_matches_multi_word_terms_as_phrases():
    rule = {"any": ["bow tie", "flip flop"]}
    assert _passes("Men's Pre-Tied Bow Ties", rule)
    assert _passes("Flip-Flops for the beach", rule)
    assert not _passes("Elbow Tiered Dress", rule)


def test_score_query_reports_slot_recall_and_unmapped_slots():
    from evaluate import score_query

    rules = {"hat": {"any": ["hat"]}, "boots": {"any": ["boot"]}}
    returned = [("sun hat", ["Straw Hat", "Bucket Hat"]), ("mystery", ["Wool Socks"])]
    s = score_query(rules, returned)
    assert s["expected_slots"] == 2 and s["slots_found"] == 1  # boots never came back
    assert s["unmapped_slots"] == 1
    assert s["items"] == 3 and s["match"] == 2
    assert s["mapped_items"] == 2 and s["mapped_match"] == 2
    assert s["success"] is False  # a returned slot with no matching item


def test_bootstrap_ci_brackets_the_mean():
    from evaluate import bootstrap_ci

    lo, hi = bootstrap_ci([0.5, 1.0, 0.75, 1.0, 0.25], n=500, seed=1)
    assert lo <= 0.7 <= hi and lo < hi
