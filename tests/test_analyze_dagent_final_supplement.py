import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_dagent_final_supplement.py"
SPEC = importlib.util.spec_from_file_location("dagent_final_supplement", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def trace_row(index: int, question: str, answer: str, em: float, *, context: str) -> dict:
    return {
        "sample_index": index,
        "question": question,
        "gold_answer": [answer],
        "pred_answer": answer if em else "wrong",
        "em": em,
        "f1": em,
        "error": None,
        "pipeline_error": None,
        "retrieval_calls": 1,
        "raw_trace": {
            "plan_warnings": [],
            "nodes_execution": [
                {
                    "type": "search",
                    "query": "Where was #1# born?",
                    "resolved_query": "Where was Ada born?",
                    "search_context": context,
                }
            ],
        },
    }


def test_coverage_and_runtime_are_derived_from_recorded_execution():
    rows = {
        0: trace_row(0, "Question A", "Paris", 1.0, context="Paris is in France."),
        1: trace_row(1, "Question B", "London", 0.0, context="No answer here."),
    }

    coverage = MODULE.coverage_analysis(rows)
    runtime = MODULE.runtime_analysis({"toy": rows})

    assert coverage["gold_string_coverage"] == pytest.approx(0.5)
    assert coverage["em_when_present"] == pytest.approx(1.0)
    assert coverage["em_when_absent"] == pytest.approx(0.0)
    assert runtime["trace_dependency_resolution_rate"] == pytest.approx(1.0)
    assert runtime["warning_free_plan_rate"] == pytest.approx(1.0)
    assert runtime["avg_executed_nodes"] == pytest.approx(1.0)


def test_analyzer_runtime_quality_uses_project_strict_validity(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "trajectory_quality.json").write_text(
        json.dumps(
            [
                {
                    "valid_plan": 0.75,
                    "dep_resolved": 1.0,
                    "unresolved_id": 0.01,
                    "num_samples": 4,
                }
            ]
        ),
        encoding="utf-8",
    )
    report = MODULE.load_analyzer_runtime_quality(tmp_path, 4)
    assert report["analyzer_strict_valid_plan_rate"] == pytest.approx(0.75)
    assert report["analyzer_dependency_resolution_rate"] == pytest.approx(1.0)


def test_official_type_slice_reports_paired_effect_and_mcnemar(tmp_path):
    official = [
        {"question": "Which film -- A or B?", "type": "bridge_comparison"},
        {"question": "Who directed C?", "type": "compositional"},
    ]
    raw_path = tmp_path / "dev.json"
    raw_path.write_text(json.dumps(official), encoding="utf-8")
    reference = {
        0: trace_row(0, "Which film A or B?", "A", 0.0, context="A"),
        1: trace_row(1, "Who directed C?", "D", 1.0, context="D"),
    }
    candidate = {
        0: trace_row(0, "Which film A or B?", "A", 1.0, context="A"),
        1: trace_row(1, "Who directed C?", "D", 1.0, context="D"),
    }

    report = MODULE.paired_type_analysis(
        reference,
        candidate,
        MODULE.load_2wiki_types(raw_path),
        bootstrap_samples=100,
        seed=7,
    )

    bridge = report["bridge_comparison"]
    assert bridge["em"]["delta"] == pytest.approx(1.0)
    assert bridge["mcnemar"]["candidate_only"] == 1
    assert bridge["mcnemar"]["reference_only"] == 0
    assert bridge["mcnemar"]["pvalue"] == pytest.approx(1.0)
    assert bridge["mcnemar"]["bonferroni_pvalue"] == pytest.approx(1.0)


def test_question_gold_alignment_supports_different_sample_indices():
    reference = {
        "toy": {
            10: trace_row(10, "Question A", "A", 1.0, context="A"),
            20: trace_row(20, "Question B", "B", 0.0, context="B"),
        }
    }
    candidate = {
        "toy": {
            1: trace_row(1, "Question B", "B", 1.0, context="B"),
            2: trace_row(2, "Question A", "A", 1.0, context="A"),
        }
    }

    aligned_reference, aligned_candidate = MODULE.validate_alignment(
        reference, candidate, "question_gold"
    )

    assert list(aligned_reference["toy"]) == list(aligned_candidate["toy"])
    assert [row["question"] for row in aligned_candidate["toy"].values()] == [
        "Question A",
        "Question B",
    ]


def test_question_gold_alignment_rejects_duplicate_keys():
    duplicate = trace_row(2, "Question A", "A", 1.0, context="A")
    reference = {
        "toy": {
            1: trace_row(1, "Question A", "A", 1.0, context="A"),
            2: duplicate,
        }
    }

    with pytest.raises(RuntimeError, match="Duplicate question_gold"):
        MODULE.validate_alignment(reference, reference, "question_gold")


def test_yes_no_answers_are_excluded_from_lexical_coverage():
    rows = {
        0: trace_row(0, "Binary", "yes", 1.0, context="yes"),
        1: trace_row(1, "Entity", "Paris", 1.0, context="Paris"),
    }
    report = MODULE.coverage_analysis(rows)
    assert report["non_binary_samples"] == 1
    assert report["gold_string_coverage"] == pytest.approx(1.0)
