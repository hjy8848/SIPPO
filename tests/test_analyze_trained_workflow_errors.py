from scripts.analyze_trained_workflow_errors import (
    classify_formal_row,
    classify_train_row,
    parse_plan_types,
)


def metadata():
    return {
        "winner": "Alice",
        "counterfactual_entity": "Bob",
        "counterfactual_answer": "London",
        "blueprint": {"selected_candidate": "Alice"},
        "comparison_evidence": {
            "candidate_a": {"answer": "1900"},
            "candidate_b": {"answer": "1950"},
        },
        "reasoning": {"operation": "earlier"},
    }


def row(nodes, *, em=0, error=""):
    return {
        "question": "Where was the older candidate born?",
        "gold_answer": ["Paris"],
        "pred_answer": "" if not em else "Paris",
        "em": em,
        "f1": em,
        "raw_trace": {
            "plan_output": (
                '<plan><search id="1">a</search><search id="2">b</search>'
                '<reason id="3">#1# #2#</reason><search id="4">#3# born</search></plan>'
            ),
            "nodes_execution": nodes,
            "execution_error": error,
        },
    }


def complete_prefix():
    return [
        {"id": "1", "type": "search", "conclusion": "Paris"},
        {"id": "2", "type": "search", "conclusion": "London"},
        {"id": "3", "type": "reason", "conclusion": "London"},
    ]


def comparison_prefix():
    return [
        {"id": "1", "type": "search", "conclusion": "1900"},
        {"id": "2", "type": "search", "conclusion": "1950"},
        {"id": "3", "type": "reason", "conclusion": "Alice"},
    ]


def test_train_classifier_isolates_identity_transfer_failure():
    nodes = comparison_prefix() + [
        {
            "id": "4",
            "type": "search",
            "query": "#3# place of birth",
            "executed_query": "the older candidate place of birth",
            "search_context": "",
            "conclusion": "",
        }
    ]
    result = classify_train_row(row(nodes), metadata())
    assert result["comparison_evidence_complete"] is True
    assert result["reason_selects_winner"] is True
    assert result["correct_reason_consumed"] is True
    assert result["identity_binding_correct"] is False
    assert result["category"] == "reason_to_search_identity_transfer_failure"


def test_train_classifier_isolates_extraction_and_aggregation():
    search = {
        "id": "4",
        "type": "search",
        "query": "#3# place of birth",
        "executed_query": "Alice place of birth",
        "search_context": "Alice was born in Paris",
        "conclusion": "unknown",
    }
    assert classify_train_row(row(comparison_prefix() + [search]), metadata())["category"] == (
        "answer_extraction_failure"
    )
    search["conclusion"] = "Paris"
    assert classify_train_row(row(comparison_prefix() + [search]), metadata())["category"] == (
        "final_aggregation_failure"
    )


def test_train_classifier_detects_semantically_invalid_shortcut():
    result = classify_train_row(row(complete_prefix()), metadata())
    assert result["comparison_evidence_complete"] is False
    assert result["semantic_shortcut"] is True
    assert result["category"] == "semantic_plan_compares_target_values"


def test_formal_classifier_uses_observable_stage_order():
    base = row(
        [
            {
                "id": "1",
                "type": "search",
                "search_context": "The answer is Paris",
                "conclusion": "unknown",
            }
        ]
    )
    result = classify_formal_row(base, "toy")
    assert result["category"] == "answer_extraction_failure"
    base["raw_trace"]["nodes_execution"][0]["conclusion"] = "Paris"
    assert classify_formal_row(base, "toy")["category"] == "final_sink_aggregation_failure"


def test_plan_parser_accepts_cpv2_typed_attributes():
    value = row([])
    value["raw_trace"]["plan_output"] = (
        '<plan sink="#4#" answer_type="location">'
        '<search id="1" subject="Alice" relation="birth_year" out="year" '
        'candidate="Alice">When was Alice born?</search>'
        '<search id="2" subject="Bob" relation="birth_year" out="year" '
        'candidate="Bob">When was Bob born?</search>'
        '<reason id="3" op="earlier" inputs="#1#,#2#" return="candidate" '
        'out="entity">Select the earlier candidate.</reason>'
        '<search id="4" subject="#3#" relation="birth_place" out="location">'
        'Where was #3# born?</search>'
        '</plan>'
    )
    assert parse_plan_types(value) == ["search", "search", "reason", "search"]
