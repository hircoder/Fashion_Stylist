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
