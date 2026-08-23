import pytest

from stylist.llm import FakeLLM, LLMError, LLMValidationError
from stylist.planner import (
    HeuristicPlanner,
    LLMPlanner,
    PlannerOutput,
    QueryPlan,
    Slot,
    content_words,
    merge_constraints,
    normalize_plan,
    parse_budget,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("sandals under $50", (None, 50.0)),
        ("boots below 80 dollars", (None, 80.0)),
        ("a dress for less than $30", (None, 30.0)),
        ("jacket, max 120", (None, 120.0)),
        ("up to 60 usd", (None, 60.0)),
        ("no more than $25", (None, 25.0)),
        ("something between 20 and 40 dollars", (20.0, 40.0)),
        ("shoes $20-40", (20.0, 40.0)),
        ("shoes $20 - $45", (20.0, 45.0)),
        ("at least $100 watch", (100.0, None)),
        ("over 200 dollars", (200.0, None)),
        ("around $30", (21.0, 39.0)),
        ("blue shirt size 12", (None, None)),
        ("", (None, None)),
    ],
)
def test_parse_budget(text, expected):
    assert parse_budget(text) == expected


def test_content_words_drops_filler_and_caps_at_six():
    words = content_words("I need an outfit to go to the beach this summer with my friends")
    assert "beach" in words and "summer" in words
    assert "need" not in words and "outfit" not in words
    assert len(words) <= 6
    assert len(set(words)) == len(words)


def test_heuristic_plan_is_always_valid_even_for_a_long_query():
    query = ("warm waterproof boots for hiking in snow " * 15)[:500]
    plan = HeuristicPlanner().plan(query)
    assert isinstance(plan, QueryPlan)
    assert plan.source == "heuristic"
    assert len(plan.slots) == 1 and plan.slots[0].search_query == query
    assert len(plan.slots[0].keywords) <= 6
    QueryPlan.model_validate(plan.model_dump())  # round trip validates


def test_heuristic_plan_picks_budget_and_audience_from_text():
    plan = HeuristicPlanner().plan("men's running shoes under $80")
    assert plan.audience == "men"
    assert plan.budget_max == 80.0 and plan.budget_scope == "per_item"
    assert plan.slots[0].budget_max == 80.0


def _out(**kw):
    base = {
        "intent": "beach outfit",
        "audience": "women",
        "occasion": "beach",
        "season": "summer",
        "budget_min": None,
        "budget_max": None,
        "budget_scope": "unknown",
        "style_keywords": [],
        "slots": [
            {
                "name": "swimsuit",
                "search_query": "women's one piece swimsuit",
                "keywords": ["swimsuit"],
            },
            {"name": "sandals", "search_query": "women's flat sandals", "keywords": ["sandals"]},
        ],
    }
    base.update(kw)
    return PlannerOutput.model_validate(base)


def test_normalize_caps_slots_at_five_and_drops_empty_queries():
    slots = [{"name": f"s{i}", "search_query": f"q{i}", "keywords": []} for i in range(7)]
    slots.insert(0, {"name": "bad", "search_query": "   ", "keywords": []})
    plan = normalize_plan(_out(slots=slots), "q")
    assert [s.name for s in plan.slots] == ["s0", "s1", "s2", "s3", "s4"]
    assert any("slot" in w for w in plan.warnings)


def test_normalize_falls_back_to_raw_query_when_no_usable_slot():
    plan = normalize_plan(
        _out(slots=[{"name": "x", "search_query": "", "keywords": []}]), "red hat"
    )
    assert len(plan.slots) == 1 and plan.slots[0].search_query == "red hat"


def test_normalize_cleans_keywords():
    out = _out(
        slots=[
            {
                "name": "s",
                "search_query": "q",
                "keywords": [" Sandal ", "sandal", "FLIP FLOP", "", "a", "b", "c", "d", "e"],
            }
        ]
    )
    plan = normalize_plan(out, "q")
    kws = plan.slots[0].keywords
    assert kws[:2] == ["sandal", "flip flop"]
    assert len(kws) == 6 and len(set(kws)) == 6


def test_normalize_per_item_budget_is_copied_to_every_slot():
    plan = normalize_plan(_out(budget_max=40.0, budget_scope="per_item"), "q")
    assert [s.budget_max for s in plan.slots] == [40.0, 40.0]


def test_normalize_total_budget_keeps_valid_allocation():
    out = _out(
        budget_max=100.0,
        budget_scope="total",
        slots=[
            {"name": "a", "search_query": "a", "keywords": [], "budget_max": 70.0},
            {"name": "b", "search_query": "b", "keywords": [], "budget_max": 30.0},
        ],
    )
    plan = normalize_plan(out, "q")
    assert [s.budget_max for s in plan.slots] == [70.0, 30.0]
    assert plan.warnings == []


def test_normalize_total_budget_scales_down_overspent_allocation():
    out = _out(
        budget_max=100.0,
        budget_scope="total",
        slots=[
            {"name": "a", "search_query": "a", "keywords": [], "budget_max": 150.0},
            {"name": "b", "search_query": "b", "keywords": [], "budget_max": 50.0},
        ],
    )
    plan = normalize_plan(out, "q")
    assert [s.budget_max for s in plan.slots] == [75.0, 25.0]
    assert any("scaled" in w for w in plan.warnings)


def test_normalize_total_budget_even_split_when_allocation_missing():
    plan = normalize_plan(_out(budget_max=100.0, budget_scope="total"), "q")
    assert [s.budget_max for s in plan.slots] == [50.0, 50.0]
    assert any("even" in w for w in plan.warnings)


def test_normalize_drops_inverted_budget_min():
    plan = normalize_plan(_out(budget_min=90.0, budget_max=40.0, budget_scope="per_item"), "q")
    assert plan.budget_min is None and plan.budget_max == 40.0
    assert plan.warnings


async def test_llm_planner_returns_llm_sourced_plan():
    llm = FakeLLM(responses=[_out()])
    plan = await LLMPlanner(llm).plan("I need an outfit for the beach")
    assert plan.source == "llm" and len(plan.slots) == 2
    assert "beach" in llm.calls[0]["user"]


async def test_llm_planner_propagates_llm_errors():
    llm = FakeLLM(responses=[LLMValidationError("bad json")])
    with pytest.raises(LLMError):
        await LLMPlanner(llm).plan("anything")


def _plan(**kw):
    base = dict(
        intent="x",
        audience="women",
        budget_min=None,
        budget_max=None,
        budget_scope="unknown",
        slots=[Slot(name="a", search_query="a"), Slot(name="b", search_query="b")],
        source="llm",
    )
    base.update(kw)
    return QueryPlan(**base)


def test_merge_uses_plan_values_when_request_is_silent():
    plan = _plan(budget_max=40.0, budget_scope="per_item")
    plan = normalize_plan(PlannerOutput.model_validate(plan.model_dump()), "q")
    windows, warnings = merge_constraints(plan)
    assert [w.max_price for w in windows] == [40.0, 40.0]
    assert all(w.audience == "women" for w in windows)
    assert all(w.include_unpriced is True for w in windows)  # inferred budget: soft
    assert warnings == []


def test_merge_request_fields_override_plan_fields():
    plan = _plan(budget_max=40.0, budget_scope="per_item", audience="women")
    plan = normalize_plan(PlannerOutput.model_validate(plan.model_dump()), "q")
    windows, _ = merge_constraints(plan, max_price=25.0, audience="men", include_unpriced=True)
    assert [w.max_price for w in windows] == [25.0, 25.0]
    assert all(w.audience == "men" and w.include_unpriced for w in windows)


def test_merge_drops_inferred_bound_that_conflicts_with_request():
    plan = _plan(budget_min=100.0, budget_max=None, budget_scope="per_item")
    plan = normalize_plan(PlannerOutput.model_validate(plan.model_dump()), "q")
    windows, warnings = merge_constraints(plan, max_price=50.0)
    assert windows[0].min_price is None and windows[0].max_price == 50.0
    assert warnings


def test_normalize_makes_slot_names_unique():
    out = _out(
        slots=[
            {"name": "top", "search_query": "a", "keywords": []},
            {"name": "Top", "search_query": "b", "keywords": []},
            {"name": "top", "search_query": "c", "keywords": []},
        ]
    )
    plan = normalize_plan(out, "q")
    assert [s.name for s in plan.slots] == ["top", "top 2", "top 3"]


def test_normalize_drops_non_finite_budgets():
    plan = normalize_plan(_out(budget_max=float("inf"), budget_scope="per_item"), "q")
    assert plan.budget_max is None and plan.budget_scope == "unknown"


def test_merge_unpriced_policy_is_strict_for_explicit_bounds_and_relaxed_for_inferred():
    inferred = _plan(budget_max=40.0, budget_scope="per_item")
    inferred = normalize_plan(PlannerOutput.model_validate(inferred.model_dump()), "q")
    windows, _ = merge_constraints(inferred)  # bound came from the planner
    assert all(w.include_unpriced for w in windows)
    windows, _ = merge_constraints(inferred, max_price=40.0)  # same bound, now explicit
    assert not any(w.include_unpriced for w in windows)
    windows, _ = merge_constraints(inferred, max_price=40.0, include_unpriced=True)
    assert all(w.include_unpriced for w in windows)
    windows, _ = merge_constraints(inferred, include_unpriced=False)
    assert not any(w.include_unpriced for w in windows)


def test_normalize_total_budget_floors_tiny_allocations():
    out = _out(
        budget_max=200.0,
        budget_scope="total",
        slots=[
            {"name": "pants", "search_query": "a", "keywords": [], "budget_max": 120.0},
            {"name": "shirt", "search_query": "b", "keywords": [], "budget_max": 70.0},
            {"name": "blazer", "search_query": "c", "keywords": [], "budget_max": 10.0},
        ],
    )
    plan = normalize_plan(out, "q")
    allocs = [s.budget_max for s in plan.slots]
    assert min(allocs) >= 20.0  # 10% of the total
    assert sum(allocs) <= 200.0 + 1e-6
    assert any("floor" in w for w in plan.warnings)


def test_normalize_keeps_cleaned_exclude_keywords():
    out = _out(
        slots=[
            {
                "name": "swimsuit",
                "search_query": "women's one piece swimsuit",
                "keywords": ["swimsuit"],
                "exclude_keywords": [" Cover Up", "cover up", "coverup", "cover-up", "x", "y"],
            }
        ]
    )
    plan = normalize_plan(out, "q")
    assert plan.slots[0].exclude_keywords == ["cover up", "coverup", "cover-up", "x"]
    assert HeuristicPlanner().plan("swimsuit").slots[0].exclude_keywords == []
