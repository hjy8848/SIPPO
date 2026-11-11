#!/usr/bin/env python3
"""Compare paired DAGENT Reason-to-Search evaluations with source-aware uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METRICS = ("em", "f1")


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    rows: dict[str, dict[str, Any]]


def parse_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--run must use LABEL=PATH: {value}")
    label, raw_path = (part.strip() for part in value.split("=", 1))
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(f"--run must use LABEL=PATH: {value}")
    return label, Path(raw_path)


def locate_result_file(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted((path / "trajectories").glob("full_logs_*.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one full_logs JSON under {path}, found {len(matches)}"
        )
    return matches[0]


def is_error(value: Any) -> bool:
    return value not in (None, "", False)


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("trajectory_id", "")).strip()
            parents = row.get("source", {}).get("parent_trajectory_ids")
            if not sample_id or not isinstance(parents, list) or len(parents) != 2:
                raise RuntimeError(f"Invalid metadata row at {path}:{line_number}")
            normalized_parents = [str(parent).strip() for parent in parents]
            if any(not parent for parent in normalized_parents):
                raise RuntimeError(f"Empty parent trajectory at {path}:{line_number}")
            if sample_id in rows:
                raise RuntimeError(f"Duplicate metadata sample_id={sample_id}")
            rows[sample_id] = {
                "question": str(row.get("question", "")).strip(),
                "parents": normalized_parents,
            }
    if not rows:
        raise RuntimeError(f"No metadata rows in {path}")
    return rows


def load_run(label: str, path: Path, expected_ids: set[str]) -> Run:
    result_path = locate_result_file(path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {result_path}")
    rows: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(payload):
        sample_id = str(row.get("sample_index", "")).strip()
        if not sample_id or sample_id in rows:
            raise RuntimeError(f"Invalid sample_index at {result_path}:{position}")
        if is_error(row.get("error")) or is_error(row.get("pipeline_error")):
            raise RuntimeError(f"Evaluation error for {label}/{sample_id}")
        for metric in METRICS:
            value = float(row.get(metric, float("nan")))
            if not np.isfinite(value):
                raise RuntimeError(f"Invalid {metric} for {label}/{sample_id}")
        rows[sample_id] = row
    if set(rows) != expected_ids:
        missing = sorted(expected_ids - set(rows))[:10]
        extra = sorted(set(rows) - expected_ids)[:10]
        raise RuntimeError(f"Sample mismatch for {label}: missing={missing}, extra={extra}")
    return Run(label=label, path=result_path.resolve(), rows=rows)


def validate_alignment(runs: Sequence[Run], metadata: Mapping[str, Mapping[str, Any]]) -> None:
    if len(runs) != 2:
        raise RuntimeError("Exactly two runs are required")
    if runs[0].label == runs[1].label:
        raise RuntimeError("Run labels must be unique")
    for sample_id, meta in metadata.items():
        left = runs[0].rows[sample_id]
        right = runs[1].rows[sample_id]
        question = str(left.get("question", "")).strip()
        if not question or question != str(right.get("question", "")).strip():
            raise RuntimeError(f"Question mismatch for sample_id={sample_id}")
        if meta["question"] and question != meta["question"]:
            raise RuntimeError(f"Metadata question mismatch for sample_id={sample_id}")
        if left.get("gold_answer") != right.get("gold_answer"):
            raise RuntimeError(f"Gold-answer mismatch for sample_id={sample_id}")


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def paired_bootstrap(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(start + 512, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def source_node_bayesian_interval(
    values: np.ndarray,
    parent_pairs: Sequence[tuple[str, str]],
    samples: int,
    seed: int,
) -> list[float]:
    parent_ids = sorted({parent for pair in parent_pairs for parent in pair})
    parent_index = {parent: index for index, parent in enumerate(parent_ids)}
    edge_indices = np.asarray(
        [(parent_index[left], parent_index[right]) for left, right in parent_pairs],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(start + 512, samples)
        node_weights = rng.exponential(1.0, size=(stop - start, len(parent_ids)))
        edge_weights = (
            node_weights[:, edge_indices[:, 0]] * node_weights[:, edge_indices[:, 1]]
        )
        means[start:stop] = (edge_weights @ values) / edge_weights.sum(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def compare(
    reference: Run,
    candidate: Run,
    metadata: Mapping[str, Mapping[str, Any]],
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_ids = sorted(metadata)
    paired_rows: list[dict[str, Any]] = []

    def summarize(group_ids: Sequence[str], group: str) -> dict[str, Any]:
        parent_pairs = [tuple(metadata[sample_id]["parents"]) for sample_id in group_ids]
        metric_report: dict[str, Any] = {}
        for metric in METRICS:
            reference_values = np.asarray(
                [float(reference.rows[sample_id][metric]) for sample_id in group_ids]
            )
            candidate_values = np.asarray(
                [float(candidate.rows[sample_id][metric]) for sample_id in group_ids]
            )
            differences = candidate_values - reference_values
            metric_report[metric] = {
                "reference": float(reference_values.mean()),
                "candidate": float(candidate_values.mean()),
                "delta": float(differences.mean()),
                "paired_bootstrap_95ci": paired_bootstrap(
                    differences, samples, stable_seed(seed, group, metric, "paired")
                ),
                "source_node_bayesian_95_interval": source_node_bayesian_interval(
                    differences,
                    parent_pairs,
                    samples,
                    stable_seed(seed, group, metric, "source-node"),
                ),
                "wins": int((differences > 0).sum()),
                "losses": int((differences < 0).sum()),
                "ties": int((differences == 0).sum()),
            }
        return {
            "samples": len(group_ids),
            "unique_parent_trajectories": len(
                {parent for pair in parent_pairs for parent in pair}
            ),
            "prediction_changed": sum(
                reference.rows[sample_id].get("pred_answer")
                != candidate.rows[sample_id].get("pred_answer")
                for sample_id in group_ids
            ),
            "metrics": metric_report,
        }

    routing: dict[str, Mapping[str, Any]] = {}
    for sample_id in sample_ids:
        value = candidate.rows[sample_id].get("raw_trace", {}).get("planner_routing")
        if isinstance(value, Mapping):
            routing[sample_id] = value
    groups: dict[str, list[str]] = {"all": sample_ids}
    if len(routing) == len(sample_ids):
        groups.update({
            "question_selected": [
                sample_id for sample_id in sample_ids
                if routing[sample_id].get("question_selected") is True
            ],
            "specialist_accepted": [
                sample_id for sample_id in sample_ids
                if routing[sample_id].get("specialist_accepted") is True
            ],
            "selected_fallback": [
                sample_id for sample_id in sample_ids
                if routing[sample_id].get("question_selected") is True
                and routing[sample_id].get("specialist_accepted") is not True
            ],
            "unselected_control": [
                sample_id for sample_id in sample_ids
                if routing[sample_id].get("question_selected") is not True
            ],
        })
    subgroup_report = {
        group: summarize(group_ids, group)
        for group, group_ids in groups.items()
        if group_ids
    }
    for sample_id in sample_ids:
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "question": metadata[sample_id]["question"],
            "parent_trajectory_a": metadata[sample_id]["parents"][0],
            "parent_trajectory_b": metadata[sample_id]["parents"][1],
            "routing_group": (
                "specialist_accepted"
                if routing.get(sample_id, {}).get("specialist_accepted") is True
                else "selected_fallback"
                if routing.get(sample_id, {}).get("question_selected") is True
                else "unselected_control"
                if sample_id in routing
                else "unavailable"
            ),
        }
        for metric in METRICS:
            left = float(reference.rows[sample_id][metric])
            right = float(candidate.rows[sample_id][metric])
            row[f"reference_{metric}"] = left
            row[f"candidate_{metric}"] = right
            row[f"delta_{metric}"] = right - left
        paired_rows.append(row)
    report = {
        "schema_version": 1,
        "reference": reference.label,
        "candidate": candidate.label,
        "samples": len(sample_ids),
        "unique_parent_trajectories": subgroup_report["all"]["unique_parent_trajectories"],
        "bootstrap_samples": samples,
        "seed": seed,
        "runs": {reference.label: str(reference.path), candidate.label: str(candidate.path)},
        "metrics": subgroup_report["all"]["metrics"],
        "subgroups": subgroup_report,
    }
    return report, paired_rows


def write_outputs(output_dir: Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "paired_questions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# DAGENT Reason-to-Search Paired Comparison",
        "",
        "| Metric | Reference | Candidate | Delta | Paired 95% CI | Source-node 95% interval | W/L/T |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, values in report["metrics"].items():
        paired = values["paired_bootstrap_95ci"]
        source = values["source_node_bayesian_95_interval"]
        lines.append(
            f"| {metric} | {values['reference']:.6f} | {values['candidate']:.6f} | "
            f"{values['delta']:+.6f} | [{paired[0]:+.6f}, {paired[1]:+.6f}] | "
            f"[{source[0]:+.6f}, {source[1]:+.6f}] | "
            f"{values['wins']}/{values['losses']}/{values['ties']} |"
        )
    if len(report.get("subgroups", {})) > 1:
        lines.extend([
            "",
            "| Group | Samples | Prediction changed | EM delta | F1 delta |",
            "|---|---:|---:|---:|---:|",
        ])
        for group, values in report["subgroups"].items():
            lines.append(
                f"| {group} | {values['samples']} | {values['prediction_changed']} | "
                f"{values['metrics']['em']['delta']:+.6f} | "
                f"{values['metrics']['f1']['delta']:+.6f} |"
            )
    (output_dir / "paired_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 2 or args.bootstrap_samples <= 0:
        raise SystemExit("Provide exactly two --run values and positive bootstrap samples")
    metadata = load_metadata(args.metadata)
    runs = [load_run(label, path, set(metadata)) for label, path in map(parse_assignment, args.run)]
    validate_alignment(runs, metadata)
    report, rows = compare(
        runs[0], runs[1], metadata, samples=args.bootstrap_samples, seed=args.seed
    )
    write_outputs(args.output_dir, report, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
