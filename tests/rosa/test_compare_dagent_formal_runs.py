from pathlib import Path
import importlib.util
import json
import sys

import pytest


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "compare_dagent_formal_runs.py"
SPEC = importlib.util.spec_from_file_location("formal_paired_comparison", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_run(root: Path, name: str, rows: list[dict]) -> None:
    trajectories = root / "trajectories"
    trajectories.mkdir(parents=True)
    (trajectories / f"full_logs_{root.name}_{name}.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )


def row(index: int, question: str, em: float, f1: float) -> dict:
    return {
        "sample_index": index,
        "question": question,
        "gold_answer": [f"answer-{index}"],
        "pred_answer": f"prediction-{index}",
        "em": em,
        "f1": f1,
        "error": None,
        "pipeline_error": None,
    }


def test_compare_runs_is_paired_by_sample_index_and_reports_ci(tmp_path):
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    write_run(
        reference_root,
        "toy",
        [row(1, "Question B", 0.0, 0.25), row(0, "Question A", 0.0, 0.0)],
    )
    write_run(
        candidate_root,
        "toy",
        [row(0, "Question A", 1.0, 1.0), row(1, "Question B", 1.0, 0.75)],
    )
    datasets = {"toy": 2}
    runs = [
        MODULE.load_run("reference", reference_root, datasets),
        MODULE.load_run("candidate", candidate_root, datasets),
    ]

    report, paired_rows = MODULE.compare_runs(
        runs, datasets, bootstrap_samples=200, seed=7
    )

    summary = report["comparisons"][0]["datasets"]["toy"]
    assert summary["em"]["delta"] == pytest.approx(1.0)
    assert summary["em"]["bootstrap_95ci"] == pytest.approx([1.0, 1.0])
    assert summary["em"]["mcnemar"]["candidate_only"] == 2
    assert summary["em"]["mcnemar"]["reference_only"] == 0
    assert summary["em"]["mcnemar"]["pvalue"] == pytest.approx(0.5)
    assert report["comparisons"][0]["macro"]["em"]["mcnemar"] == {
        "candidate_only": 2,
        "reference_only": 0,
        "pvalue": pytest.approx(0.5),
    }
    assert (summary["em"]["wins"], summary["em"]["losses"], summary["em"]["ties"]) == (2, 0, 0)
    assert summary["f1"]["delta"] == pytest.approx(0.75)
    assert [item["sample_index"] for item in paired_rows] == [0, 1]


def test_validate_alignment_rejects_question_mismatch(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    write_run(left_root, "toy", [row(0, "Original", 0.0, 0.0)])
    write_run(right_root, "toy", [row(0, "Changed", 0.0, 0.0)])
    datasets = {"toy": 1}
    runs = [
        MODULE.load_run("left", left_root, datasets),
        MODULE.load_run("right", right_root, datasets),
    ]

    with pytest.raises(RuntimeError, match="Question mismatch"):
        MODULE.compare_runs(runs, datasets, bootstrap_samples=10)


def test_question_gold_alignment_supports_different_sample_indices(tmp_path):
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_rows = [row(10, "Question A", 0.0, 0.0), row(20, "Question B", 1.0, 0.5)]
    candidate_rows = [row(200, "Question B", 1.0, 1.0), row(100, "Question A", 1.0, 1.0)]
    candidate_rows[0]["gold_answer"] = ["answer-20"]
    candidate_rows[1]["gold_answer"] = ["answer-10"]
    write_run(reference_root, "toy", reference_rows)
    write_run(candidate_root, "toy", candidate_rows)
    datasets = {"toy": 2}
    runs = [
        MODULE.load_run("reference", reference_root, datasets),
        MODULE.load_run("candidate", candidate_root, datasets),
    ]

    report, paired_rows = MODULE.compare_runs(
        runs,
        datasets,
        bootstrap_samples=20,
        alignment_key="question_gold",
    )

    assert report["alignment_key"] == "question_gold"
    assert report["comparisons"][0]["datasets"]["toy"]["em"]["delta"] == pytest.approx(0.5)
    assert {(item["sample_index"], item["candidate_sample_index"]) for item in paired_rows} == {
        (10, 100),
        (20, 200),
    }


def test_question_gold_alignment_rejects_duplicate_keys(tmp_path):
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    duplicate = [row(0, "Question A", 0.0, 0.0), row(1, "Question A", 1.0, 1.0)]
    duplicate[1]["gold_answer"] = duplicate[0]["gold_answer"]
    write_run(reference_root, "toy", duplicate)
    write_run(candidate_root, "toy", [row(2, "Question A", 1.0, 1.0), row(3, "Question B", 0.0, 0.0)])
    datasets = {"toy": 2}
    runs = [
        MODULE.load_run("reference", reference_root, datasets),
        MODULE.load_run("candidate", candidate_root, datasets),
    ]

    with pytest.raises(RuntimeError, match="Duplicate question_gold"):
        MODULE.compare_runs(
            runs,
            datasets,
            bootstrap_samples=10,
            alignment_key="question_gold",
        )


def test_load_result_rows_rejects_duplicates_and_pipeline_errors(tmp_path):
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        json.dumps([row(0, "A", 0.0, 0.0), row(0, "A", 1.0, 1.0)]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Duplicate sample_index"):
        MODULE.load_result_rows(duplicate_path, 2)

    error_path = tmp_path / "error.json"
    error_row = row(0, "A", 0.0, 0.0)
    error_row["pipeline_error"] = "retriever timeout"
    error_path.write_text(json.dumps([error_row]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Evaluation error"):
        MODULE.load_result_rows(error_path, 1)
