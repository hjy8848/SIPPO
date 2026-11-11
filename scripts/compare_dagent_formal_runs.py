#!/usr/bin/env python3
"""Create strict paired comparisons for frozen DAGENT formal-v2 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_DATASETS = {
    "2wikimultihopqa": 500,
    "hotpotqa": 500,
    "musique": 500,
    "bamboogle": 125,
}
METRICS = ("em", "f1")
ALIGNMENT_KEYS = ("sample_index", "question_gold")


@dataclass(frozen=True)
class FormalRun:
    label: str
    root: Path
    rows: dict[str, dict[int, dict[str, Any]]]


def parse_assignment(value: str, *, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} must use LABEL=VALUE: {value}")
    label, raw_value = value.split("=", 1)
    label = label.strip()
    raw_value = raw_value.strip()
    if not label or not raw_value:
        raise argparse.ArgumentTypeError(f"{option} must use LABEL=VALUE: {value}")
    return label, raw_value


def parse_datasets(values: Sequence[str] | None) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_DATASETS)
    datasets: dict[str, int] = {}
    for value in values:
        name, raw_count = parse_assignment(value, option="--dataset")
        count = int(raw_count)
        if count <= 0:
            raise argparse.ArgumentTypeError(
                f"--dataset count must be positive: {value}"
            )
        if name in datasets:
            raise argparse.ArgumentTypeError(f"Duplicate dataset: {name}")
        datasets[name] = count
    return datasets


def locate_result_file(root: Path, dataset: str) -> Path:
    search_roots = [root / "trajectories", root]
    matches: list[Path] = []
    for search_root in search_roots:
        if search_root.is_dir():
            matches.extend(search_root.glob(f"full_logs_*_{dataset}.json"))
    unique = sorted(set(path.resolve() for path in matches))
    if len(unique) != 1:
        raise RuntimeError(
            f"Expected exactly one result file for {dataset} under {root}, "
            f"found {len(unique)}: {[str(path) for path in unique]}"
        )
    return unique[0]


def _is_error(value: Any) -> bool:
    return value not in (None, "", False)


def load_result_rows(path: Path, expected_count: int) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {path}")
    if len(payload) != expected_count:
        raise RuntimeError(
            f"Unexpected sample count in {path}: {len(payload)}/{expected_count}"
        )

    indexed: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Row {position} is not an object in {path}")
        if "sample_index" not in row:
            raise RuntimeError(f"Row {position} has no sample_index in {path}")
        sample_index = int(row["sample_index"])
        if sample_index in indexed:
            raise RuntimeError(f"Duplicate sample_index={sample_index} in {path}")
        if not str(row.get("question", "")).strip():
            raise RuntimeError(f"Empty question at sample_index={sample_index} in {path}")
        for metric in METRICS:
            value = float(row.get(metric, math.nan))
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Invalid {metric} at sample_index={sample_index} in {path}"
                )
        if _is_error(row.get("error")) or _is_error(row.get("pipeline_error")):
            raise RuntimeError(
                f"Evaluation error at sample_index={sample_index} in {path}: "
                f"error={row.get('error')!r}, pipeline_error={row.get('pipeline_error')!r}"
            )
        indexed[sample_index] = row
    return indexed


def load_run(
    label: str, root: Path, datasets: Mapping[str, int]
) -> FormalRun:
    rows = {
        dataset: load_result_rows(locate_result_file(root, dataset), expected)
        for dataset, expected in datasets.items()
    }
    return FormalRun(label=label, root=root.resolve(), rows=rows)


def row_alignment_key(row: Mapping[str, Any], alignment_key: str) -> Any:
    if alignment_key == "sample_index":
        return int(row["sample_index"])
    if alignment_key == "question_gold":
        return (
            str(row["question"]).strip(),
            json.dumps(row.get("gold_answer"), ensure_ascii=False, sort_keys=True),
        )
    raise ValueError(f"Unsupported alignment key: {alignment_key}")


def index_rows_for_alignment(
    run: FormalRun, dataset: str, alignment_key: str
) -> dict[Any, dict[str, Any]]:
    indexed: dict[Any, dict[str, Any]] = {}
    for row in run.rows[dataset].values():
        key = row_alignment_key(row, alignment_key)
        if key in indexed:
            raise RuntimeError(
                f"Duplicate {alignment_key} for {run.label}/{dataset}: {key!r}"
            )
        indexed[key] = row
    return indexed


def validate_alignment(
    runs: Sequence[FormalRun],
    datasets: Mapping[str, int],
    alignment_key: str = "sample_index",
) -> None:
    if len(runs) < 2:
        raise RuntimeError("At least two runs are required for a paired comparison")
    labels = [run.label for run in runs]
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"Run labels must be unique: {labels}")

    reference = runs[0]
    for dataset, expected in datasets.items():
        reference_rows = index_rows_for_alignment(reference, dataset, alignment_key)
        if len(reference_rows) != expected:
            raise RuntimeError(
                f"Unexpected indexed count for {reference.label}/{dataset}: "
                f"{len(reference_rows)}/{expected}"
            )
        reference_keys = set(reference_rows)
        for run in runs[1:]:
            rows = index_rows_for_alignment(run, dataset, alignment_key)
            if set(rows) != reference_keys:
                missing = sorted(reference_keys - set(rows))[:10]
                extra = sorted(set(rows) - reference_keys)[:10]
                raise RuntimeError(
                    f"{alignment_key} mismatch for {run.label}/{dataset}: "
                    f"missing={missing}, extra={extra}"
                )
            for key in reference_keys:
                left = reference_rows[key]
                right = rows[key]
                if str(left["question"]).strip() != str(right["question"]).strip():
                    raise RuntimeError(
                        f"Question mismatch for {dataset} alignment_key={key!r}: "
                        f"{reference.label}={left['question']!r}, "
                        f"{run.label}={right['question']!r}"
                    )
                if left.get("gold_answer") != right.get("gold_answer"):
                    raise RuntimeError(
                        f"Gold-answer mismatch for {dataset} "
                        f"alignment_key={key!r}"
                    )


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


def stratified_macro_bootstrap_ci(
    values_by_dataset: Mapping[str, Sequence[float]], *, samples: int, seed: int
) -> list[float]:
    if not values_by_dataset:
        raise RuntimeError("Cannot bootstrap an empty dataset collection")
    rng = np.random.default_rng(seed)
    macro_means = np.zeros(samples, dtype=np.float64)
    for values in values_by_dataset.values():
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            raise RuntimeError("Cannot bootstrap an empty dataset")
        chunk_size = min(512, samples)
        for start in range(0, samples, chunk_size):
            stop = min(start + chunk_size, samples)
            indices = rng.integers(
                0, array.size, size=(stop - start, array.size)
            )
            macro_means[start:stop] += array[indices].mean(axis=1)
    macro_means /= len(values_by_dataset)
    low, high = np.quantile(macro_means, [0.025, 0.975])
    return [float(low), float(high)]


def win_loss_tie(differences: Sequence[float], tolerance: float = 1e-12) -> dict[str, int]:
    wins = sum(value > tolerance for value in differences)
    losses = sum(value < -tolerance for value in differences)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(differences) - wins - losses,
    }


def exact_mcnemar_pvalue(candidate_only: int, reference_only: int) -> float:
    discordant = candidate_only + reference_only
    if discordant == 0:
        return 1.0
    smaller = min(candidate_only, reference_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def summarize_run(run: FormalRun, datasets: Mapping[str, int]) -> dict[str, Any]:
    dataset_metrics: dict[str, dict[str, float | int]] = {}
    for dataset in datasets:
        rows = run.rows[dataset].values()
        dataset_metrics[dataset] = {
            "samples": len(run.rows[dataset]),
            "em": statistics.fmean(float(row["em"]) for row in rows),
            "f1": statistics.fmean(
                float(row["f1"]) for row in run.rows[dataset].values()
            ),
            "errors": 0,
        }
    return {
        "root": str(run.root),
        "datasets": dataset_metrics,
        "macro_em": statistics.fmean(
            float(metrics["em"]) for metrics in dataset_metrics.values()
        ),
        "macro_f1": statistics.fmean(
            float(metrics["f1"]) for metrics in dataset_metrics.values()
        ),
    }


def selected_pairs(
    runs: Sequence[FormalRun], focus_label: str | None
) -> Iterable[tuple[FormalRun, FormalRun]]:
    if focus_label is None:
        return itertools.combinations(runs, 2)
    matches = [run for run in runs if run.label == focus_label]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown --focus-label: {focus_label}")
    focus = matches[0]
    return ((run, focus) for run in runs if run is not focus)


def compare_pair(
    reference: FormalRun,
    candidate: FormalRun,
    datasets: Mapping[str, int],
    *,
    bootstrap_samples: int,
    seed: int,
    alignment_key: str = "sample_index",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_summaries: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    all_differences: dict[str, list[float]] = {metric: [] for metric in METRICS}
    differences_by_dataset: dict[str, dict[str, list[float]]] = {
        metric: {} for metric in METRICS
    }

    for dataset in datasets:
        ref_rows = index_rows_for_alignment(reference, dataset, alignment_key)
        cand_rows = index_rows_for_alignment(candidate, dataset, alignment_key)
        metric_differences: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for key in sorted(ref_rows, key=str):
            ref_row = ref_rows[key]
            cand_row = cand_rows[key]
            output_row = {
                "reference": reference.label,
                "candidate": candidate.label,
                "dataset": dataset,
                "sample_index": int(ref_row["sample_index"]),
                "candidate_sample_index": int(cand_row["sample_index"]),
                "alignment_key": alignment_key,
                "question": str(ref_row["question"]),
                "gold_answer": json.dumps(ref_row.get("gold_answer"), ensure_ascii=False),
                "reference_prediction": str(ref_row.get("pred_answer", "")),
                "candidate_prediction": str(cand_row.get("pred_answer", "")),
            }
            for metric in METRICS:
                ref_value = float(ref_row[metric])
                cand_value = float(cand_row[metric])
                difference = cand_value - ref_value
                metric_differences[metric].append(difference)
                all_differences[metric].append(difference)
                output_row[f"reference_{metric}"] = ref_value
                output_row[f"candidate_{metric}"] = cand_value
                output_row[f"delta_{metric}"] = difference
            paired_rows.append(output_row)

        dataset_summary: dict[str, Any] = {"samples": len(ref_rows)}
        for metric in METRICS:
            differences = metric_differences[metric]
            differences_by_dataset[metric][dataset] = differences
            ref_mean = statistics.fmean(float(row[metric]) for row in ref_rows.values())
            cand_mean = statistics.fmean(
                float(row[metric]) for row in cand_rows.values()
            )
            metric_summary = {
                "reference": ref_mean,
                "candidate": cand_mean,
                "delta": cand_mean - ref_mean,
                "bootstrap_95ci": bootstrap_mean_ci(
                    differences,
                    samples=bootstrap_samples,
                    seed=stable_seed(seed, reference.label, candidate.label, dataset, metric),
                ),
                **win_loss_tie(differences),
            }
            if metric == "em":
                candidate_only = sum(value > 0.5 for value in differences)
                reference_only = sum(value < -0.5 for value in differences)
                metric_summary["mcnemar"] = {
                    "candidate_only": candidate_only,
                    "reference_only": reference_only,
                    "pvalue": exact_mcnemar_pvalue(candidate_only, reference_only),
                }
            dataset_summary[metric] = metric_summary
        dataset_summaries[dataset] = dataset_summary

    macro_summary: dict[str, Any] = {"samples": sum(datasets.values())}
    for metric in METRICS:
        dataset_metric_summaries = [
            dataset_summaries[dataset][metric] for dataset in datasets
        ]
        metric_summary = {
            "reference": statistics.fmean(
                float(summary["reference"]) for summary in dataset_metric_summaries
            ),
            "candidate": statistics.fmean(
                float(summary["candidate"]) for summary in dataset_metric_summaries
            ),
            "delta": statistics.fmean(
                float(summary["delta"]) for summary in dataset_metric_summaries
            ),
            "bootstrap_95ci": stratified_macro_bootstrap_ci(
                differences_by_dataset[metric],
                samples=bootstrap_samples,
                seed=stable_seed(seed, reference.label, candidate.label, "macro", metric),
            ),
            **win_loss_tie(all_differences[metric]),
        }
        if metric == "em":
            candidate_only = sum(value > 0.5 for value in all_differences[metric])
            reference_only = sum(value < -0.5 for value in all_differences[metric])
            metric_summary["mcnemar"] = {
                "candidate_only": candidate_only,
                "reference_only": reference_only,
                "pvalue": exact_mcnemar_pvalue(candidate_only, reference_only),
            }
        macro_summary[metric] = metric_summary
    return (
        {
            "reference": reference.label,
            "candidate": candidate.label,
            "datasets": dataset_summaries,
            "macro": macro_summary,
        },
        paired_rows,
    )


def compare_runs(
    runs: Sequence[FormalRun],
    datasets: Mapping[str, int],
    *,
    focus_label: str | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    alignment_key: str = "sample_index",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if alignment_key not in ALIGNMENT_KEYS:
        raise ValueError(f"Unsupported alignment key: {alignment_key}")
    validate_alignment(runs, datasets, alignment_key)
    comparisons: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for reference, candidate in selected_pairs(runs, focus_label):
        summary, rows = compare_pair(
            reference,
            candidate,
            datasets,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            alignment_key=alignment_key,
        )
        comparisons.append(summary)
        paired_rows.extend(rows)
    return (
        {
            "protocol": "DAGENT formal-v2 strict paired comparison",
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "datasets": dict(datasets),
            "focus_label": focus_label,
            "alignment_key": alignment_key,
            "runs": {run.label: summarize_run(run, datasets) for run in runs},
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


def flatten_comparisons(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in report["comparisons"]:
        sections = dict(comparison["datasets"])
        sections["macro"] = comparison["macro"]
        for dataset, summary in sections.items():
            for metric in METRICS:
                metric_summary = summary[metric]
                rows.append(
                    {
                        "reference": comparison["reference"],
                        "candidate": comparison["candidate"],
                        "dataset": dataset,
                        "metric": metric,
                        "samples": summary["samples"],
                        "reference_score": metric_summary["reference"],
                        "candidate_score": metric_summary["candidate"],
                        "delta": metric_summary["delta"],
                        "ci_low": metric_summary["bootstrap_95ci"][0],
                        "ci_high": metric_summary["bootstrap_95ci"][1],
                        "wins": metric_summary["wins"],
                        "losses": metric_summary["losses"],
                        "ties": metric_summary["ties"],
                        "mcnemar_pvalue": (
                            metric_summary.get("mcnemar", {}).get("pvalue")
                        ),
                    }
                )
    return rows


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# DAGENT Formal-v2 Paired Comparison",
        "",
        f"Bootstrap samples: {report['bootstrap_samples']}; seed: {report['seed']}",
        "",
        "| Reference | Candidate | Scope | Metric | Reference | Candidate | Delta | 95% CI | W/L/T | McNemar p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flatten_comparisons(report):
        lines.append(
            "| {reference} | {candidate} | {dataset} | {metric} | "
            "{reference_score:.6f} | {candidate_score:.6f} | {delta:+.6f} | "
            "[{ci_low:+.6f}, {ci_high:+.6f}] | {wins}/{losses}/{ties} | {mcnemar} |".format(
                mcnemar=(
                    f"{row['mcnemar_pvalue']:.6g}"
                    if row["mcnemar_pvalue"] is not None
                    else "--"
                ),
                **row
            )
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Formal-v2 run root containing a trajectories directory",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="NAME=COUNT",
        help="Override the default formal-v2 datasets (repeatable)",
    )
    parser.add_argument(
        "--focus-label",
        help="Compare every other run against this candidate instead of all pairs",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alignment-key",
        choices=ALIGNMENT_KEYS,
        default="sample_index",
        help="Strict row identity used for pairing across runs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = parse_datasets(args.dataset)
    run_specs = [parse_assignment(value, option="--run") for value in args.run]
    runs = [load_run(label, Path(path), datasets) for label, path in run_specs]
    report, paired_rows = compare_runs(
        runs,
        datasets,
        focus_label=args.focus_label,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        alignment_key=args.alignment_key,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "paired_comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "paired_comparison.csv", flatten_comparisons(report))
    write_csv(args.output_dir / "paired_questions.csv", paired_rows)
    (args.output_dir / "paired_comparison.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] Paired comparison: {report_path}")


if __name__ == "__main__":
    main()
