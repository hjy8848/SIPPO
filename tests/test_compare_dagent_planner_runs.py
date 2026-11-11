from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_dagent_planner_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_dagent_planner_runs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_run(label: str, values: list[tuple[bool, bool, int]]):
    rows = {}
    for index, (valid, reason_search, node_count) in enumerate(values):
        row = {
            "dataset": "toy",
            "sample_index": index,
            "question": f"question {index}",
            "valid": valid,
            "reason_search": reason_search,
            "node_count": node_count,
        }
        rows[MODULE.row_key(row)] = row
    return MODULE.PlannerRun(label=label, path=Path(f"{label}.jsonl"), rows=rows)


def test_compare_runs_reports_paired_mechanism_deltas():
    reference = make_run("reference", [(False, True, 7), (True, False, 2)])
    candidate = make_run("candidate", [(True, True, 5), (True, True, 4)])

    report, paired = MODULE.compare_runs(
        [reference, candidate],
        focus_label="candidate",
        bootstrap_samples=100,
        seed=7,
    )

    metrics = report["comparisons"][0]["metrics"]
    assert metrics["valid"]["delta"] == pytest.approx(0.5)
    assert metrics["reason_search"]["delta"] == pytest.approx(0.5)
    assert metrics["over_five_nodes"]["delta"] == pytest.approx(-0.5)
    assert metrics["node_count"]["delta"] == pytest.approx(0.0)
    assert len(paired) == 2


def test_validate_alignment_rejects_question_drift():
    reference = make_run("reference", [(True, False, 2)])
    candidate = make_run("candidate", [(True, False, 2)])
    candidate.rows["toy:0"]["question"] = "different question"

    with pytest.raises(RuntimeError, match="question mismatch"):
        MODULE.validate_alignment([reference, candidate])
