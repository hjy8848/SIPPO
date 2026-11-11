import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "exp" / "analyze_reason_search_eval.py"
SPEC = importlib.util.spec_from_file_location("analyze_reason_search_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def eval_row(question, em):
    return {
        "question": question,
        "em": em,
        "f1": em,
        "raw_trace": {
            "plan_output": (
                '<plan><search id="1">A</search><search id="2">B</search>'
                '<reason id="3">Compare #1# and #2#</reason>'
                '<search id="4">Find #3# occupation</search></plan>'
            ),
            "nodes_execution": [
                {"type": "search"},
                {"type": "search"},
                {"type": "reason"},
                {"type": "search"},
            ],
        },
        "search_steps": 3,
        "retrieval_calls": 3,
    }


def metadata(question, mode, relation):
    return {
        "question": question,
        "trajectory_id": question,
        "reasoning": {"mode": mode},
        "blueprint": {"downstream": [{"relation": relation}]},
    }


def test_metadata_join_and_grouped_metrics():
    rows = [eval_row("Question A?", 1), eval_row("Question B?", 0)]
    metadata_rows = [
        metadata("Question A?", "direct", "occupation"),
        metadata("Question B?", "anchor_distance", "director"),
    ]
    summary, records = MODULE.analyze(rows, metadata_rows)
    assert summary["metadata_joined"] == 2
    assert summary["em"] == 0.5
    assert summary["by_reasoning_mode"]["direct"]["em"] == 1.0
    assert summary["by_reasoning_mode"]["anchor_distance"]["em"] == 0.0
    assert summary["by_relation"]["occupation"]["samples"] == 1
    assert records[0]["trajectory_id"] == "Question A?"


def test_metadata_join_fails_when_evaluation_is_incomplete():
    rows = [eval_row("Question A?", 1)]
    metadata_rows = [
        metadata("Question A?", "direct", "occupation"),
        metadata("Question B?", "direct", "occupation"),
    ]
    with pytest.raises(ValueError, match="matched 1/2"):
        MODULE.analyze(rows, metadata_rows)


def test_metadata_join_rejects_duplicate_questions():
    rows = [eval_row("Question A?", 1)]
    metadata_rows = [
        metadata("Question A?", "direct", "occupation"),
        metadata("Question A!", "direct", "occupation"),
    ]
    with pytest.raises(ValueError, match="Duplicate normalized metadata question"):
        MODULE.analyze(rows, metadata_rows)
