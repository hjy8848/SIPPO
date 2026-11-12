import pytest

from qa_manager.ROSA.planner_router import (
    classify_causal_comparison_question,
    is_exact_causal_plan,
    normalize_route_mode,
    normalize_specialist_route_scope,
    plan_topology,
)


@pytest.mark.parametrize(
    "question",
    [
        "Where was the older candidate, Ada Lovelace or Grace Hopper, born?",
        "What was the occupation of the younger candidate, Ada or Grace?",
        "Who directed the earlier-released work, Film A or Film B?",
        "What was the genre of the work whose release year was closer to 1950, A or B?",
        "What was the nationality of the candidate whose birth year was closer to 1900, A or B?",
    ],
)
def test_causal_question_router_accepts_dataset_grammar(question):
    decision = classify_causal_comparison_question(question)
    assert decision["selected"] is True
    assert decision["reason"] == "causal_comparison_question"
    assert decision["reasoning_mode"] in {"direct", "anchor_distance"}


@pytest.mark.parametrize(
    "question",
    [
        "Which film has the director who was born later, Film A or Film B?",
        "Who was born later, Ada Lovelace or Grace Hopper?",
        "Who directed more films, Director A or Director B?",
        "Where was Ada Lovelace born?",
        "What is the occupation of Ada Lovelace or Grace Hopper?",
    ],
)
def test_causal_question_router_rejects_selection_only_or_noncausal_questions(question):
    assert not classify_causal_comparison_question(question)["selected"]


def test_exact_causal_plan_requires_search_search_reason_search():
    exact = [
        ("search", "1", "A birth date"),
        ("search", "2", "B birth date"),
        ("reason", "3", "Compare #1# and #2#"),
        ("search", "4", "#3# occupation"),
    ]
    assert plan_topology(exact) == ("search", "search", "reason", "search")
    assert is_exact_causal_plan(exact)
    assert not is_exact_causal_plan(exact[:-1])
    assert not is_exact_causal_plan([*exact, ("reason", "5", "Summarize #4#")])


def test_route_mode_validation_is_strict():
    assert normalize_route_mode("") == "all"
    assert normalize_route_mode("CAUSAL_EXACT") == "causal_exact"
    assert normalize_route_mode("ANCHOR_EXACT") == "anchor_exact"
    with pytest.raises(ValueError, match="Unsupported PLANNER_ROUTE_MODE"):
        normalize_route_mode("learned_from_formal")


def test_specialist_route_scope_is_strictly_normalized():
    assert normalize_specialist_route_scope(" WORKFLOW ") == "workflow"
    assert normalize_specialist_route_scope("") == "planner"
    with pytest.raises(ValueError, match="Unsupported SPECIALIST_ROUTE_SCOPE"):
        normalize_specialist_route_scope("all_roles")
