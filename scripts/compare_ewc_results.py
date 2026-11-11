#!/usr/bin/env python3
"""Build a reproducible, paired comparison report for EWC result JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_ewc_pilot import paired_summary


VARIANTS = ("base", "dependency_changed", "answer_preserving")
METRICS = (*VARIANTS, "joint")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty result file: {path}")
    required = {"pair_id", "variant", "em", "pipeline_error", "final_answer", "gold_answer"}
    for index, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{path}:{index}: missing {missing}")
        if row["variant"] not in VARIANTS:
            raise ValueError(f"{path}:{index}: unexpected variant {row['variant']!r}")
    return rows


def index_pairs(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        variant = str(row["variant"])
        if variant in pairs.setdefault(pair_id, {}):
            raise ValueError(f"{label}: duplicate {pair_id}/{variant}")
        pairs[pair_id][variant] = row
    incomplete = sorted(pair_id for pair_id, worlds in pairs.items() if set(worlds) != set(VARIANTS))
    if incomplete:
        raise ValueError(f"{label}: incomplete pairs: {incomplete}")
    return pairs


def metric_values(pairs: dict[str, dict[str, dict[str, Any]]], metric: str) -> dict[str, int]:
    if metric in VARIANTS:
        return {pair_id: int(bool(worlds[metric]["em"])) for pair_id, worlds in pairs.items()}
    return {
        pair_id: int(all(bool(worlds[variant]["em"]) for variant in VARIANTS))
        for pair_id, worlds in pairs.items()
    }


def compare(candidate: dict[str, dict[str, dict[str, Any]]], reference: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    if set(candidate) != set(reference):
        raise ValueError("comparison inputs have different pair IDs")
    count = len(candidate)
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        candidate_values = metric_values(candidate, metric)
        reference_values = metric_values(reference, metric)
        wins = sum(candidate_values[key] == 1 and reference_values[key] == 0 for key in candidate)
        losses = sum(candidate_values[key] == 0 and reference_values[key] == 1 for key in candidate)
        ties = count - wins - losses
        metrics[metric] = {
            "candidate_rate": sum(candidate_values.values()) / count,
            "reference_rate": sum(reference_values.values()) / count,
            "delta": (sum(candidate_values.values()) - sum(reference_values.values())) / count,
            "candidate_only_correct": wins,
            "reference_only_correct": losses,
            "ties": ties,
        }
    return {"pair_count": count, "metrics": metrics}


def report_entry(label: str, rows: list[dict[str, Any]], pairs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "label": label,
        "instances": len(rows),
        "pipeline_errors": sum(bool(row["pipeline_error"]) for row in rows),
        "em": sum(int(bool(row["em"])) for row in rows),
        "em_rate": sum(int(bool(row["em"])) for row in rows) / len(rows),
        "paired": paired_summary(rows),
        "pair_ids": sorted(pairs),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EWC-QA validation comparison",
        "",
        "This is a 20-pair validation pilot (60 worlds per model), not a formal benchmark result.",
        "",
        "## Per-model summary",
        "",
        "| Model | EM | Base | Changed | Preserve | Joint | Change sensitivity | Preserve invariance | Pipeline errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, entry in report["models"].items():
        paired = entry["paired"]
        lines.append(
            "| {label} | {em}/{instances} ({em_rate:.1%}) | {base:.1%} | {changed:.1%} | "
            "{preserve:.1%} | {joint:.1%} | {sensitivity:.1%} | {invariance:.1%} | {errors} |".format(
                label=label,
                em=entry["em"],
                instances=entry["instances"],
                em_rate=entry["em_rate"],
                base=paired["base_accuracy"],
                changed=paired["dependency_changed_accuracy"],
                preserve=paired["answer_preserving_accuracy"],
                joint=paired["joint_accuracy"],
                sensitivity=paired["changed_answer_rate"],
                invariance=paired["answer_preserving_invariance_rate"],
                errors=entry["pipeline_errors"],
            )
        )
    if report["comparisons"]:
        lines.extend(["", "## Paired endpoint comparisons", ""])
        for label, comparison in report["comparisons"].items():
            lines.extend([f"### {label}", "", "| Metric | Delta | Candidate-only | Reference-only | Ties |", "| --- | ---: | ---: | ---: | ---: |"])
            for metric, values in comparison["metrics"].items():
                lines.append(
                    f"| {metric} | {values['delta']:+.1%} | {values['candidate_only_correct']} | "
                    f"{values['reference_only_correct']} | {values['ties']} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=RESULTS_JSONL")
    parser.add_argument("--reference", help="Label used as the reference for paired comparisons")
    parser.add_argument(
        "--compare-label",
        action="append",
        help="Candidate label to compare with --reference; defaults to every non-reference input",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs: dict[str, Path] = {}
    for item in args.input:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            raise SystemExit(f"invalid --input {item!r}; expected LABEL=RESULTS_JSONL")
        if label in inputs:
            raise SystemExit(f"duplicate label: {label}")
        inputs[label] = Path(raw_path)
    if args.reference and args.reference not in inputs:
        raise SystemExit(f"reference {args.reference!r} is not an input label")
    if args.compare_label and not args.reference:
        raise SystemExit("--compare-label requires --reference")
    if args.compare_label:
        unknown = sorted(set(args.compare_label) - set(inputs))
        if unknown:
            raise SystemExit(f"comparison labels are not inputs: {unknown}")
        if args.reference in args.compare_label:
            raise SystemExit("--compare-label must not include --reference")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")

    rows_by_label = {label: read_jsonl(path) for label, path in inputs.items()}
    pairs_by_label = {label: index_pairs(rows, label) for label, rows in rows_by_label.items()}
    report = {
        "benchmark": "EWC-QA validation v6",
        "scope": "20-pair validation pilot; descriptive paired endpoint comparison only",
        "models": {
            label: report_entry(label, rows_by_label[label], pairs_by_label[label])
            for label in inputs
        },
        "comparisons": {},
    }
    if args.reference:
        labels = args.compare_label or [label for label in inputs if label != args.reference]
        for label in labels:
            report["comparisons"][f"{label} minus {args.reference}"] = compare(
                pairs_by_label[label], pairs_by_label[args.reference]
            )

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
