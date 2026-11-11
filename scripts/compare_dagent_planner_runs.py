#!/usr/bin/env python3
"""Create strict paired comparisons for frozen DAGENT Planner evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METRICS = ("valid", "reason_search", "over_five_nodes", "node_count")
ALIGNMENT_FIELDS = ("dataset", "sample_index", "question")


@dataclass(frozen=True)
class PlannerRun:
    label: str
    path: Path
    rows: dict[str, dict[str, Any]]


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--run must use LABEL=VALUE: {value}")
    label, raw_path = (part.strip() for part in value.split("=", 1))
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(f"--run must use LABEL=VALUE: {value}")
    return label, raw_path


def locate_generations(path: Path) -> Path:
    candidate = path / "generations.jsonl" if path.is_dir() else path
    if not candidate.is_file():
        raise RuntimeError(f"Missing generations.jsonl: {candidate}")
    return candidate


def row_key(row: Mapping[str, Any]) -> str:
    return f"{row['dataset']}:{int(row['sample_index'])}"


def load_run(label: str, path: Path, expected_total: int) -> PlannerRun:
    generations = locate_generations(path)
    rows: dict[str, dict[str, Any]] = {}
    with generations.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ALIGNMENT_FIELDS:
                if field not in row or str(row[field]).strip() == "":
                    raise RuntimeError(
                        f"Missing {field} at {generations}:{line_number}"
                    )
            if not isinstance(row.get("valid"), bool):
                raise RuntimeError(f"Invalid valid flag at {generations}:{line_number}")
            if not isinstance(row.get("reason_search"), bool):
                raise RuntimeError(
                    f"Invalid reason_search flag at {generations}:{line_number}"
                )
            if not isinstance(row.get("node_count"), int) or row["node_count"] < 0:
                raise RuntimeError(
                    f"Invalid node_count at {generations}:{line_number}"
                )
            key = row_key(row)
            if key in rows:
                raise RuntimeError(f"Duplicate question key={key} in {generations}")
            rows[key] = row
    if len(rows) != expected_total:
        raise RuntimeError(
            f"Unexpected row count in {generations}: {len(rows)}/{expected_total}"
        )
    return PlannerRun(label=label, path=generations.resolve(), rows=rows)


def validate_alignment(runs: Sequence[PlannerRun]) -> None:
    if len(runs) < 2:
        raise RuntimeError("At least two runs are required")
    if len({run.label for run in runs}) != len(runs):
        raise RuntimeError("Run labels must be unique")
    reference = runs[0]
    for run in runs[1:]:
        if set(run.rows) != set(reference.rows):
            missing = sorted(set(reference.rows) - set(run.rows))[:10]
            extra = sorted(set(run.rows) - set(reference.rows))[:10]
            raise RuntimeError(
                f"Question mismatch for {run.label}: missing={missing}, extra={extra}"
            )
        for key, left in reference.rows.items():
            right = run.rows[key]
            for field in ALIGNMENT_FIELDS:
                if left[field] != right[field]:
                    raise RuntimeError(
                        f"{field} mismatch for {key}: "
                        f"{reference.label}={left[field]!r}, {run.label}={right[field]!r}"
                    )


def metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "over_five_nodes":
        return float(row["node_count"] > 5)
    return float(row[metric])


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or samples <= 0:
        raise ValueError("Bootstrap needs non-empty values and positive samples")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk_size = min(512, samples)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def selected_pairs(
    runs: Sequence[PlannerRun], focus_label: str | None
) -> Sequence[tuple[PlannerRun, PlannerRun]]:
    if focus_label is None:
        return list(itertools.combinations(runs, 2))
    matches = [run for run in runs if run.label == focus_label]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown --focus-label: {focus_label}")
    focus = matches[0]
    return [(run, focus) for run in runs if run is not focus]


def compare_runs(
    runs: Sequence[PlannerRun],
    *,
    focus_label: str | None,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_alignment(runs)
    keys = sorted(runs[0].rows)
    comparisons = []
    paired_rows = []
    for reference, candidate in selected_pairs(runs, focus_label):
        metrics = {}
        for metric in METRICS:
            reference_values = [metric_value(reference.rows[key], metric) for key in keys]
            candidate_values = [metric_value(candidate.rows[key], metric) for key in keys]
            differences = [
                candidate_value - reference_value
                for reference_value, candidate_value in zip(
                    reference_values, candidate_values, strict=True
                )
            ]
            metrics[metric] = {
                "reference": statistics.fmean(reference_values),
                "candidate": statistics.fmean(candidate_values),
                "delta": statistics.fmean(differences),
                "bootstrap_95ci": bootstrap_mean_ci(
                    differences,
                    samples=bootstrap_samples,
                    seed=stable_seed(seed, reference.label, candidate.label, metric),
                ),
                "wins": sum(value > 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
        comparisons.append(
            {
                "reference": reference.label,
                "candidate": candidate.label,
                "samples": len(keys),
                "metrics": metrics,
            }
        )
        for key in keys:
            left = reference.rows[key]
            right = candidate.rows[key]
            record: dict[str, Any] = {
                "reference": reference.label,
                "candidate": candidate.label,
                **{field: left[field] for field in ALIGNMENT_FIELDS},
            }
            for metric in METRICS:
                left_value = metric_value(left, metric)
                right_value = metric_value(right, metric)
                record[f"reference_{metric}"] = left_value
                record[f"candidate_{metric}"] = right_value
                record[f"delta_{metric}"] = right_value - left_value
            paired_rows.append(record)
    return (
        {
            "schema_version": 1,
            "samples": len(keys),
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "focus_label": focus_label,
            "runs": {run.label: str(run.path) for run in runs},
            "comparisons": comparisons,
        },
        paired_rows,
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# DAGENT Planner Paired Comparison",
        "",
        "| Reference | Candidate | Metric | Reference | Candidate | Delta | 95% CI | W/L/T |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["comparisons"]:
        for metric, values in comparison["metrics"].items():
            low, high = values["bootstrap_95ci"]
            lines.append(
                f"| {comparison['reference']} | {comparison['candidate']} | {metric} | "
                f"{values['reference']:.6f} | {values['candidate']:.6f} | "
                f"{values['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] | "
                f"{values['wins']}/{values['losses']}/{values['ties']} |"
            )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--focus-label")
    parser.add_argument("--expected-total", type=int, default=1625)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = [
        load_run(label, Path(path), args.expected_total)
        for label, path in (parse_assignment(value) for value in args.run)
    ]
    report, paired_rows = compare_runs(
        runs,
        focus_label=args.focus_label,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "paired_questions.csv", paired_rows)
    (args.output_dir / "paired_comparison.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
