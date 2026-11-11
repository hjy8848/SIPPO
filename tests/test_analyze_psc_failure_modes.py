import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_psc_failure_modes", ROOT / "scripts" / "analyze_psc_failure_modes.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def metadata():
    return {
        "sample_id": "sample-1",
        "question": "Where was the older candidate born?",
        "reasoning_mode": "direct",
        "operation": "earlier",
        "relation": "place of birth",
        "winner": "Ada Example",
        "candidate_a": "Ada Example",
        "candidate_b": "Bob Example",
    }


def test_dependency_and_reason_search_detection():
    nodes = MODULE.parse_plan(
        '<plan><search id="1">A date</search><search id="2">B date</search>'
        '<reason id="3">Compare #1# and #2#</reason>'
        '<search id="4">#3# birthplace</search></plan>'
    )
    assert MODULE.reason_search_edges(nodes) == [("3", "4")]
    assert MODULE.dependency_violations(nodes) == []
    broken = MODULE.parse_plan(
        '<plan><search id="1">#2# fact</search><reason id="2">#2# compare</reason></plan>'
    )
    assert MODULE.dependency_violations(broken) == ["future:1->2", "self:2"]

    executed = [
        {"id": "3", "type": "reason", "conclusion": "Ada Example"},
        {"id": "4", "type": "search", "query": "#3# birthplace"},
    ]
    assert MODULE.executed_reason_search_edges(executed) == [("3", "4")]


def test_extract_record_requires_edge_nodes_to_execute():
    row = {
        "sample_index": "sample-1",
        "question": metadata()["question"],
        "em": 1,
        "f1": 1,
        "raw_trace": {
            "plan_output": (
                '<plan><search id="1">A date</search><search id="2">B date</search>'
                '<reason id="3">Compare #1# and #2#</reason>'
                '<search id="4">#3# birthplace</search></plan>'
            ),
            "nodes_execution": [
                {"id": "1", "type": "search"},
                {"id": "2", "type": "search"},
                {"id": "3", "type": "reason", "conclusion": "Ada Example"},
                {
                    "id": "4", "type": "search", "query": "#3# birthplace",
                    "resolved_query": "Ada Example birthplace",
                },
            ],
        },
    }
    record = MODULE.extract_record(row, metadata())
    assert record["executed_reason_to_search"] is True
    assert record["executed_search_after_reason"] is True
    assert record["executed_exact_topology"] is True
    assert record["reason_selects_winner"] is True
    assert record["reason_mentions_candidate"] is True
    assert record["downstream_query_contains_winner"] is True
    assert record["downstream_query_mentions_candidate"] is True

    row["raw_trace"]["nodes_execution"] = row["raw_trace"]["nodes_execution"][:3]
    truncated = MODULE.extract_record(row, metadata())
    assert truncated["planned_reason_to_search"] is True
    assert truncated["executed_reason_to_search"] is False
    assert truncated["executed_search_after_reason"] is False


def test_paired_summary_and_case_categories():
    left = {
        "a": {**metadata(), "em": 0.0, "f1": 0.0, "executed_reason_to_search": False},
        "b": {**metadata(), "sample_id": "b", "em": 1.0, "f1": 1.0, "executed_reason_to_search": False},
    }
    right = {
        "a": {
            **left["a"], "em": 1.0, "f1": 1.0, "executed_reason_to_search": True,
            "reason_selects_winner": True, "downstream_query_contains_winner": True,
            "execution_error_text": "", "downstream_queries": "Ada Example birthplace",
            "reason_conclusions": "Ada Example", "plan_output": "candidate plan",
        },
        "b": {
            **left["b"], "em": 0.0, "f1": 0.0, "executed_reason_to_search": False,
            "reason_selects_winner": False, "downstream_query_contains_winner": False,
            "execution_error_text": "", "downstream_queries": "", "reason_conclusions": "",
            "plan_output": "candidate plan",
        },
    }
    left["a"]["plan_output"] = left["b"]["plan_output"] = "reference plan"
    report = MODULE.paired_summary(left, right, ["a", "b"], samples=100, seed=3, label="test")
    assert report["em"]["delta"] == 0.0
    assert report["mcnemar"]["candidate_only"] == 1
    assert report["mcnemar"]["reference_only"] == 1
    categories = {case["category"] for case in MODULE.select_cases(left, right)}
    assert categories == {
        "reason_to_search_candidate_only_correct",
        "no_reason_to_search_reference_only_correct",
    }
