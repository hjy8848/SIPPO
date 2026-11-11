import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_trained_workflow_stages.py"
SPEC = importlib.util.spec_from_file_location("compare_trained_workflow_stages", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _rows(specs):
    return {
        ("train1000", MODULE.normalize(question)): {
            "question": question,
            "em": str(em),
            "category": category,
            "semantic_shortcut": str(shortcut),
            "canonical_plan": str(canonical),
            "plan_topology": topology,
        }
        for question, em, category, shortcut, canonical, topology in specs
    }


def test_compare_split_tracks_introduced_and_resolved_defects():
    sft = _rows(
        [
            ("q1", 0, "planner_missing_reason", False, False, "search"),
            ("q2", 1, "correct", True, False, "search-reason"),
        ]
    )
    pure = _rows(
        [
            ("q1", 1, "correct", False, True, "search-search-reason-search"),
            ("q2", 0, "semantic_plan_compares_target_values", True, False, "search-reason"),
        ]
    )
    psc = _rows(
        [
            ("q1", 0, "semantic_plan_compares_target_values", True, False, "search-reason"),
            ("q2", 1, "correct", False, True, "search-search-reason-search"),
        ]
    )

    report, cases = MODULE.compare_split(sft, pure, psc, MODULE.TRAIN_DEFECTS)

    assert report["em_transitions"]["sft_to_pure"]["improved"] == 1
    assert report["em_transitions"]["sft_to_pure"]["regressed"] == 1
    assert report["defects"]["semantic_shortcut"]["psc_introduced_vs_pure"] == 1
    assert report["defects"]["semantic_shortcut"]["psc_resolved_vs_pure"] == 1
    assert report["defects"]["missing_reason"]["pure_resolved_vs_sft"] == 1
    assert len(cases) == 2


def test_load_cases_rejects_duplicate_questions(tmp_path):
    path = tmp_path / "cases.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "em", "category"])
        writer.writeheader()
        writer.writerow({"question": "Same?", "em": 0, "category": "x"})
        writer.writerow({"question": "same", "em": 1, "category": "correct"})

    with pytest.raises(ValueError, match="duplicate"):
        MODULE.load_cases(path, formal=False)


def test_node_budget_stable_subset_filters_any_over_budget_stage():
    sft = _rows(
        [
            ("q1", 0, "correct", False, True, "search-reason"),
            ("q2", 0, "correct", False, True, "search-search"),
        ]
    )
    pure = _rows(
        [
            ("q1", 0, "correct", False, True, "search-reason"),
            ("q2", 0, "correct", False, True, "search-search-reason"),
        ]
    )
    psc = _rows(
        [
            ("q1", 0, "correct", False, True, "search-reason"),
            ("q2", 0, "correct", False, True, "search-search-reason-search"),
        ]
    )

    stable = MODULE.node_budget_stable_subset(
        {"sft": sft, "pure": pure, "psc": psc}, max_nodes=3
    )

    assert set(stable["sft"]) == {("train1000", MODULE.normalize("q1"))}
