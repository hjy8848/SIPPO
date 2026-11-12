from pathlib import Path
import importlib.util
import json
import sys

import pytest


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "compare_dagent_reasoner_runs.py"
SPEC = importlib.util.spec_from_file_location("reasoner_paired_comparison", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(pair_id: str, *, chosen: bool, exposure: str = "held_out") -> dict:
    return {
        "pair_id": pair_id,
        "group_id": f"group-{pair_id}",
        "operation": "earlier",
        "variant": "factual_condition",
        "split": "test",
        "causal_sft_exposure": exposure,
        "format_valid": True,
        "chosen_exact": chosen,
        "semantic_label": "chosen" if chosen else "rejected",
    }


def write_run(root: Path, rows: list[dict]) -> None:
    root.mkdir()
    (root / "generations.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
    )


def test_compare_runs_pairs_by_pair_id_and_reports_ci(tmp_path):
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    write_run(reference_root, [row("b", chosen=False), row("a", chosen=False)])
    write_run(candidate_root, [row("a", chosen=True), row("b", chosen=True)])
    runs = [
        MODULE.load_run("reference", reference_root, expected_total=2),
        MODULE.load_run("candidate", candidate_root, expected_total=2),
    ]

    report, paired_rows = MODULE.compare_runs(
        runs,
        exposure="held_out",
        expected_filtered=2,
        bootstrap_samples=200,
        seed=7,
    )

    semantic = report["comparisons"][0]["metrics"]["semantic_chosen"]
    assert semantic["delta"] == pytest.approx(1.0)
    assert semantic["bootstrap_95ci"] == pytest.approx([1.0, 1.0])
    assert (semantic["wins"], semantic["losses"], semantic["ties"]) == (2, 0, 0)
    assert [item["pair_id"] for item in paired_rows] == ["a", "b"]


def test_compare_runs_filters_actual_exposure(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    write_run(
        left_root,
        [row("held", chosen=False), row("seen", chosen=False, exposure="seen")],
    )
    write_run(
        right_root,
        [row("seen", chosen=True, exposure="seen"), row("held", chosen=True)],
    )
    runs = [
        MODULE.load_run("left", left_root, expected_total=2),
        MODULE.load_run("right", right_root, expected_total=2),
    ]

    report, paired_rows = MODULE.compare_runs(
        runs, exposure="held_out", expected_filtered=1, bootstrap_samples=20
    )

    assert report["samples"] == 1
    assert [item["pair_id"] for item in paired_rows] == ["held"]


def test_validate_alignment_rejects_metadata_mismatch(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left = row("a", chosen=False)
    right = row("a", chosen=True)
    right["operation"] = "later"
    write_run(left_root, [left])
    write_run(right_root, [right])
    runs = [
        MODULE.load_run("left", left_root, expected_total=1),
        MODULE.load_run("right", right_root, expected_total=1),
    ]

    with pytest.raises(RuntimeError, match="operation mismatch"):
        MODULE.compare_runs(
            runs, exposure="held_out", expected_filtered=1, bootstrap_samples=20
        )
