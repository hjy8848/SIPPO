#!/usr/bin/env python3
"""Create strict paired comparisons for canonical DAGENT Reasoner runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METRICS = ("format_valid", "chosen_exact", "semantic_chosen")
ALIGNMENT_FIELDS = (
    "group_id",
    "operation",
    "variant",
    "split",
    "causal_sft_exposure",
)


@dataclass(frozen=True)
class ReasonerRun:
    label: str
    path: Path
    rows: dict[str, dict[str, Any]]


def parse_assignment(value: str, *, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} must use LABEL=VALUE: {value}")
    label, raw_value = value.split("=", 1)
    label = label.strip()
    raw_value = raw_value.strip()
    if not label or not raw_value:
        raise argparse.ArgumentTypeError(f"{option} must use LABEL=VALUE: {value}")
    return label, raw_value


def locate_generations(path: Path) -> Path:
    candidate = path / "generations.jsonl" if path.is_dir() else path
    if not candidate.is_file():
        raise RuntimeError(f"Missing generations.jsonl: {candidate}")
    return candidate


def load_run(label: str, path: Path, expected_total: int) -> ReasonerRun:
    generations = locate_generations(path)
    rows: dict[str, dict[str, Any]] = {}
    with generations.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row.get("pair_id", "")).strip()
            if not pair_id:
                raise RuntimeError(f"Missing pair_id at {generations}:{line_number}")
            if pair_id in rows:
                raise RuntimeError(f"Duplicate pair_id={pair_id} in {generations}")
            if row.get("semantic_label") not in {"chosen", "rejected", "other"}:
                raise RuntimeError(
                    f"Invalid semantic_label for pair_id={pair_id} in {generations}"
                )
            for field in ("format_valid", "chosen_exact"):
                if not isinstance(row.get(field), bool):
                    raise RuntimeError(
                        f"Invalid {field} for pair_id={pair_id} in {generations}"
                    )
            for field in ALIGNMENT_FIELDS:
                if not str(row.get(field, "")).strip():
                    raise RuntimeError(
                        f"Missing {field} for pair_id={pair_id} in {generations}"
                    )
            rows[pair_id] = row
    if len(rows) != expected_total:
        raise RuntimeError(
            f"Unexpected row count in {generations}: {len(rows)}/{expected_total}"
        )
    return ReasonerRun(label=label, path=generations.resolve(), rows=rows)


def validate_alignment(runs: Sequence[ReasonerRun]) -> None:
    if len(runs) < 2:
        raise RuntimeError("At least two runs are required for a paired comparison")
    labels = [run.label for run in runs]
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"Run labels must be unique: {labels}")

    reference = runs[0]
    reference_ids = set(reference.rows)
    for run in runs[1:]:
        if set(run.rows) != reference_ids:
            missing = sorted(reference_ids - set(run.rows))[:10]
            extra = sorted(set(run.rows) - reference_ids)[:10]
            raise RuntimeError(
                f"pair_id mismatch for {run.label}: missing={missing}, extra={extra}"
            )
        for pair_id in reference_ids:
            left = reference.rows[pair_id]
            right = run.rows[pair_id]
            for field in ALIGNMENT_FIELDS:
                if left[field] != right[field]:
                    raise RuntimeError(
                        f"{field} mismatch for pair_id={pair_id}: "
                        f"{reference.label}={left[field]!r}, {run.label}={right[field]!r}"
                    )


def metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "semantic_chosen":
        return float(row["semantic_label"] == "chosen")
    return float(bool(row[metric]))


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise RuntimeError("Cannot bootstrap an empty paired difference")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk_size = min(512, samples)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def win_loss_tie(values: Sequence[float]) -> dict[str, int]:
    return {
        "wins": sum(value > 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "ties": sum(value == 0 for value in values),
    }


def selected_pairs(
    runs: Sequence[ReasonerRun], focus_label: str | None
) -> Iterable[tuple[ReasonerRun, ReasonerRun]]:
    if focus_label is None:
        return itertools.combinations(runs, 2)
    matches = [run for run in runs if run.label == focus_label]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown --focus-label: {focus_label}")
    focus = matches[0]
    return ((run, focus) for run in runs if run is not focus)


def compare_runs(
    runs: Sequence[ReasonerRun],
    *,
    exposure: str,
    expected_filtered: int,
    focus_label: str | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_alignment(runs)
    pair_ids = sorted(
        pair_id
        for pair_id, row in runs[0].rows.items()
        if row["causal_sft_exposure"] == exposure
    )
    if len(pair_ids) != expected_filtered:
        raise RuntimeError(
            f"Unexpected {exposure} count: {len(pair_ids)}/{expected_filtered}"
        )

    comparisons: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for reference, candidate in selected_pairs(runs, focus_label):
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            reference_values = [
                metric_value(reference.rows[pair_id], metric) for pair_id in pair_ids
            ]
            candidate_values = [
                metric_value(candidate.rows[pair_id], metric) for pair_id in pair_ids
            ]
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
                **win_loss_tie(differences),
            }
        comparisons.append(
            {
                "reference": reference.label,
                "candidate": candidate.label,
                "samples": len(pair_ids),
                "metrics": metrics,
            }
        )
        for pair_id in pair_ids:
            metadata = reference.rows[pair_id]
            record: dict[str, Any] = {
                "reference": reference.label,
                "candidate": candidate.label,
                "pair_id": pair_id,
                **{field: metadata[field] for field in ALIGNMENT_FIELDS},
            }
            for metric in METRICS:
                reference_value = metric_value(reference.rows[pair_id], metric)
                candidate_value = metric_value(candidate.rows[pair_id], metric)
                record[f"reference_{metric}"] = reference_value
                record[f"candidate_{metric}"] = candidate_value
                record[f"delta_{metric}"] = candidate_value - reference_value
            paired_rows.append(record)

    return (
        {
            "schema_version": 1,
            "exposure": exposure,
            "samples": len(pair_ids),
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "focus_label": focus_label,
            "runs": {run.label: str(run.path) for run in runs},
            "comparisons": comparisons,
        },
        paired_rows,
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# DAGENT Canonical Reasoner Paired Comparison",
        "",
        f"Exposure: {report['exposure']}; samples: {report['samples']}; "
        f"bootstrap samples: {report['bootstrap_samples']}; seed: {report['seed']}",
        "",
        "| Reference | Candidate | Metric | Reference | Candidate | Delta | 95% CI | W/L/T |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["comparisons"]:
        for metric, summary in comparison["metrics"].items():
            low, high = summary["bootstrap_95ci"]
            lines.append(
                f"| {comparison['reference']} | {comparison['candidate']} | {metric} | "
                f"{summary['reference']:.6f} | {summary['candidate']:.6f} | "
                f"{summary['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] | "
                f"{summary['wins']}/{summary['losses']}/{summary['ties']} |"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run directory or generations.jsonl path",
    )
    parser.add_argument("--focus-label")
    parser.add_argument("--exposure", default="held_out")
    parser.add_argument("--expected-total", type=int, default=1970)
    parser.add_argument("--expected-filtered", type=int, default=1744)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_specs = [parse_assignment(value, option="--run") for value in args.run]
    runs = [
        load_run(label, Path(path), expected_total=args.expected_total)
        for label, path in run_specs
    ]
    report, paired_rows = compare_runs(
        runs,
        exposure=args.exposure,
        expected_filtered=args.expected_filtered,
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
