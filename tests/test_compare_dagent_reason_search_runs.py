from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_dagent_reason_search_runs import (
    compare,
    load_metadata,
    load_run,
    validate_alignment,
    write_outputs,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def make_run(path: Path, values: list[tuple[str, float, float]]) -> None:
    trajectories = path / "trajectories"
    trajectories.mkdir(parents=True)
    payload = [
        {
            "sample_index": sample_id,
            "question": f"Question {sample_id}",
            "gold_answer": [f"answer-{sample_id}"],
            "em": em,
            "f1": f1,
            "error": None,
            "pipeline_error": None,
        }
        for sample_id, em, f1 in values
    ]
    (trajectories / "full_logs_test.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_paired_comparison_supports_string_ids_and_source_dependence(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_rows = [
        {
            "trajectory_id": f"sample-{index}",
            "question": f"Question sample-{index}",
            "source": {"parent_trajectory_ids": [f"parent-{index}", "shared-parent"]},
        }
        for index in range(4)
    ]
    write_jsonl(metadata_path, metadata_rows)
    make_run(
        tmp_path / "reference",
        [(f"sample-{index}", 0.0, 0.1 * index) for index in range(4)],
    )
    make_run(
        tmp_path / "candidate",
        [(f"sample-{index}", 1.0, 0.2 + 0.1 * index) for index in range(4)],
    )

    metadata = load_metadata(metadata_path)
    reference = load_run("reference", tmp_path / "reference", set(metadata))
    candidate = load_run("candidate", tmp_path / "candidate", set(metadata))
    validate_alignment([reference, candidate], metadata)
    report, rows = compare(reference, candidate, metadata, samples=1000, seed=42)

    assert report["samples"] == 4
    assert report["unique_parent_trajectories"] == 5
    assert report["metrics"]["em"]["delta"] == 1.0
    assert report["metrics"]["f1"]["delta"] == pytest.approx(0.2)
    assert len(report["metrics"]["em"]["source_node_bayesian_95_interval"]) == 2
    assert len(rows) == 4
    assert set(report["subgroups"]) == {"all"}

    output_dir = tmp_path / "comparison"
    write_outputs(output_dir, report, rows)
    assert (output_dir / "paired_comparison.json").is_file()
    assert (output_dir / "paired_questions.csv").is_file()
    assert (output_dir / "paired_comparison.md").is_file()


def test_paired_comparison_reports_routing_controls(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    write_jsonl(
        metadata_path,
        [
            {
                "trajectory_id": f"sample-{index}",
                "question": f"Question sample-{index}",
                "source": {"parent_trajectory_ids": [f"parent-{index}", "shared"]},
            }
            for index in range(3)
        ],
    )
    make_run(tmp_path / "reference", [(f"sample-{i}", 0.0, 0.0) for i in range(3)])
    make_run(tmp_path / "candidate", [(f"sample-{i}", 1.0, 1.0) for i in range(3)])
    candidate_path = tmp_path / "candidate" / "trajectories" / "full_logs_test.json"
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload[0]["raw_trace"] = {"planner_routing": {"question_selected": True, "specialist_accepted": True}}
    payload[1]["raw_trace"] = {"planner_routing": {"question_selected": True, "specialist_accepted": False}}
    payload[2]["raw_trace"] = {"planner_routing": {"question_selected": False, "specialist_accepted": False}}
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    metadata = load_metadata(metadata_path)
    reference = load_run("reference", tmp_path / "reference", set(metadata))
    candidate = load_run("candidate", tmp_path / "candidate", set(metadata))
    report, rows = compare(reference, candidate, metadata, samples=100, seed=42)

    assert report["subgroups"]["question_selected"]["samples"] == 2
    assert report["subgroups"]["specialist_accepted"]["samples"] == 1
    assert report["subgroups"]["selected_fallback"]["samples"] == 1
    assert report["subgroups"]["unselected_control"]["samples"] == 1
    assert {row["routing_group"] for row in rows} == {
        "specialist_accepted", "selected_fallback", "unselected_control"
    }


def test_alignment_rejects_question_mismatch(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    write_jsonl(
        metadata_path,
        [
            {
                "trajectory_id": "sample-a",
                "question": "Question sample-a",
                "source": {"parent_trajectory_ids": ["parent-a", "parent-b"]},
            }
        ],
    )
    make_run(tmp_path / "reference", [("sample-a", 0.0, 0.0)])
    make_run(tmp_path / "candidate", [("sample-a", 1.0, 1.0)])
    payload_path = tmp_path / "candidate" / "trajectories" / "full_logs_test.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload[0]["question"] = "Different question"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    metadata = load_metadata(metadata_path)
    reference = load_run("reference", tmp_path / "reference", set(metadata))
    candidate = load_run("candidate", tmp_path / "candidate", set(metadata))
    with pytest.raises(RuntimeError, match="Question mismatch"):
        validate_alignment([reference, candidate], metadata)
