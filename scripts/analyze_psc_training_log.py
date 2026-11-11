#!/usr/bin/env python3
"""Summarize PSC signal coverage, gradient alignment, and training health."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"(?:^| - )([^:\n]+):([^\s]+)")
PREFIX = "aux/planner_outcome_contrast/"
MAX_INTERPRETABLE_GRADIENT_RATIO = 10.0
METRICS = {
    "valid_rate": "train/workflow/valid_rate",
    "rollout_em": "train/rollout/em_aggregate",
    "best_of_n_em": "train/rollout/best_of_n_em",
    "planned_reason_nodes": "train/workflow/planned_reason_nodes_mean",
    "reasoner_calls": "train/agent_calls/ReasonerAgent",
    "informative_questions_per_minibatch": PREFIX + "global_informative_question_count",
    "pairs_per_minibatch": PREFIX + "global_pair_count",
    "empty_refreshed_minibatch": PREFIX + "empty_refreshed_minibatch",
    "gradient_ratio": PREFIX + "refreshed_weighted_poc_to_planner_em_direct_gradient_ratio",
    "gradient_cosine": PREFIX + "refreshed_weighted_poc_planner_em_direct_gradient_cosine",
    "planner_em_gradient_norm": PREFIX + "refreshed_planner_em_direct_gradient_norm",
    "weighted_psc_gradient_norm": PREFIX + "refreshed_weighted_poc_direct_gradient_norm",
    "optimizer_steps_expected": "actor/optimizer_steps_expected",
    "optimizer_steps_executed": "actor/optimizer_steps_executed",
    "minibatches": "actor/question_preserving_minibatch_count",
    "global_signal_required": PREFIX + "global_signal_required",
    "non_planner_tokens": PREFIX + "non_planner_nonzero_tokens",
    "grad_norm": "actor/grad_norm",
}


def parse_metric_line(line: str) -> dict[str, float]:
    clean = ANSI_RE.sub("", line).replace("\r", "")
    metrics: dict[str, float] = {}
    for raw_key, raw_value in METRIC_RE.findall(clean):
        key = raw_key.strip().split()[-1]
        try:
            metrics[key] = float(raw_value)
        except ValueError:
            continue
    return metrics


def parse_steps(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    seen: set[int] = set()
    for line in text.splitlines():
        if "training/global_step:" not in line:
            continue
        metrics = parse_metric_line(line)
        if "training/global_step" not in metrics:
            continue
        step = int(round(metrics["training/global_step"]))
        if step in seen:
            raise RuntimeError(f"Duplicate completed metric line for step {step}")
        seen.add(step)
        rows.append(metrics)
    return sorted(rows, key=lambda row: row["training/global_step"])


def select_step_range(
    rows: Sequence[Mapping[str, float]], *, start: int, end: int, source: Path
) -> list[dict[str, float]]:
    if start <= 0 or end < start:
        raise RuntimeError(f"Invalid step range {start}:{end} for {source}")
    selected = [dict(row) for row in rows if start <= int(row["training/global_step"]) <= end]
    actual = [int(row["training/global_step"]) for row in selected]
    expected = list(range(start, end + 1))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise RuntimeError(
            f"Segment {source}@{start}:{end} is incomplete; missing steps: {missing[:20]}"
        )
    return selected


def merge_segments(
    segments: Sequence[tuple[Path, int, int]],
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    merged: dict[int, dict[str, float]] = {}
    provenance: list[dict[str, Any]] = []
    for path, start, end in segments:
        rows = select_step_range(
            parse_steps(path.read_text(encoding="utf-8", errors="replace")),
            start=start,
            end=end,
            source=path,
        )
        provenance.append(
            {
                "path": str(path.resolve()),
                "step_start": start,
                "step_end": end,
                "completed_steps": len(rows),
            }
        )
        for row in rows:
            step = int(row["training/global_step"])
            if step in merged:
                raise RuntimeError(f"Overlapping lineage segments contain step {step}")
            merged[step] = row
    ordered = [merged[step] for step in sorted(merged)]
    if ordered:
        first = int(ordered[0]["training/global_step"])
        last = int(ordered[-1]["training/global_step"])
        missing = sorted(set(range(first, last + 1)) - set(merged))
        if missing:
            raise RuntimeError(f"Lineage is not contiguous; missing steps: {missing[:20]}")
    return ordered, provenance


def describe(values: Sequence[float]) -> dict[str, float] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    ordered = sorted(finite)
    return {
        "count": len(finite),
        "sum": math.fsum(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": ordered[0],
        "p10": ordered[max(0, math.ceil(0.1 * len(ordered)) - 1)],
        "p90": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def window_summary(rows: Sequence[Mapping[str, float]], key: str) -> dict[str, float] | None:
    values = [row[key] for row in rows if key in row]
    if not values:
        return None
    width = max(1, len(values) // 4)
    first, last = values[:width], values[-width:]
    return {
        "first_quarter_mean": statistics.fmean(first),
        "last_quarter_mean": statistics.fmean(last),
        "mean_change": statistics.fmean(last) - statistics.fmean(first),
        "first_quarter_median": statistics.median(first),
        "last_quarter_median": statistics.median(last),
        "median_change": statistics.median(last) - statistics.median(first),
    }


def analyze(
    rows: Sequence[Mapping[str, float]],
    *,
    expected_workflows: int,
    expected_global_signal: bool | None = None,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No completed training steps found")
    normalized = []
    missing: dict[str, int] = {name: 0 for name in METRICS}
    violations = {
        "workflow_count": [],
        "incomplete_question_groups": [],
        "optimizer_step_mismatch": [],
        "global_signal_contract_mismatch": [],
        "non_planner_psc_tokens": [],
        "non_finite_metrics": [],
        "denominator_degenerate_gradient_ratio": [],
    }
    for row in rows:
        step = int(round(row["training/global_step"]))
        compact: dict[str, float] = {"step": float(step)}
        for name, key in METRICS.items():
            if key in row:
                compact[name] = float(row[key])
            else:
                missing[name] += 1
        minibatches = compact.get("minibatches")
        informative = compact.get("informative_questions_per_minibatch")
        pairs = compact.get("pairs_per_minibatch")
        if minibatches is not None and informative is not None:
            compact["estimated_informative_questions"] = minibatches * informative
        if minibatches is not None and pairs is not None:
            compact["estimated_pairs"] = minibatches * pairs
        gradient_ratio = compact.get("gradient_ratio")
        if (
            gradient_ratio is not None
            and gradient_ratio <= MAX_INTERPRETABLE_GRADIENT_RATIO
            and compact.get("planner_em_gradient_norm", 0.0) > 0
            and compact.get("weighted_psc_gradient_norm", 0.0) > 0
        ):
            compact["interpretable_gradient_ratio"] = gradient_ratio
        normalized.append(compact)

        if int(round(row.get("train/workflow/completed", -1))) != expected_workflows:
            violations["workflow_count"].append(step)
        if row.get("train/workflow/incomplete_question_groups", 0.0) != 0.0:
            violations["incomplete_question_groups"].append(step)
        if compact.get("optimizer_steps_expected") != compact.get("optimizer_steps_executed"):
            violations["optimizer_step_mismatch"].append(step)
        if (
            expected_global_signal is not None
            and compact.get("global_signal_required") != float(expected_global_signal)
        ):
            violations["global_signal_contract_mismatch"].append(step)
        if compact.get("non_planner_tokens", 0.0) != 0.0:
            violations["non_planner_psc_tokens"].append(step)
        if any(not math.isfinite(float(value)) for value in row.values()):
            violations["non_finite_metrics"].append(step)
        if compact.get("gradient_ratio", 0.0) > MAX_INTERPRETABLE_GRADIENT_RATIO:
            violations["denominator_degenerate_gradient_ratio"].append(step)

    summary: dict[str, Any] = {}
    trend: dict[str, Any] = {}
    derived_names = tuple(METRICS) + (
        "estimated_informative_questions",
        "estimated_pairs",
        "interpretable_gradient_ratio",
    )
    for name in derived_names:
        summary[name] = describe([row[name] for row in normalized if name in row])
        trend[name] = window_summary(normalized, name)
    cosine_values = [row["gradient_cosine"] for row in normalized if "gradient_cosine" in row]
    ratio_values = [row["gradient_ratio"] for row in normalized if "gradient_ratio" in row]
    interpretable_ratio_values = [
        row["interpretable_gradient_ratio"]
        for row in normalized
        if "interpretable_gradient_ratio" in row
    ]
    signal_values = [row["estimated_informative_questions"] for row in normalized if "estimated_informative_questions" in row]
    signal_present_every_step = len(signal_values) == len(normalized) and all(
        value > 0 for value in signal_values
    )
    signal_present_any_step = bool(signal_values) and any(value > 0 for value in signal_values)
    if expected_global_signal is True:
        signal_contract_pass = signal_present_every_step
    elif expected_global_signal is False:
        signal_contract_pass = signal_present_any_step
    else:
        signal_contract_pass = None
    audit = {
        "all_engineering_invariants_pass": not any(
            violations[name]
            for name in (
                "workflow_count", "incomplete_question_groups", "optimizer_step_mismatch",
                "global_signal_contract_mismatch", "non_planner_psc_tokens", "non_finite_metrics",
            )
        ),
        "signal_present_every_step": signal_present_every_step,
        "signal_present_any_step": signal_present_any_step,
        "signal_nonempty_step_rate": (
            sum(value > 0 for value in signal_values) / len(normalized)
            if len(signal_values) == len(normalized)
            else None
        ),
        "signal_contract_pass": signal_contract_pass,
        "positive_alignment_rate": (
            sum(value > 0 for value in cosine_values) / len(cosine_values) if cosine_values else None
        ),
        "gradient_budget_0p1_to_0p3_rate": (
            sum(0.1 <= value <= 0.3 for value in interpretable_ratio_values)
            / len(interpretable_ratio_values)
            if interpretable_ratio_values
            else None
        ),
        "raw_gradient_budget_0p1_to_0p3_rate": (
            sum(0.1 <= value <= 0.3 for value in ratio_values) / len(ratio_values)
            if ratio_values
            else None
        ),
        "interpretable_gradient_ratio_steps": len(interpretable_ratio_values),
        "denominator_degenerate_gradient_ratio_rate": (
            len(violations["denominator_degenerate_gradient_ratio"]) / len(normalized)
        ),
    }
    return {
        "steps": len(normalized),
        "step_min": int(normalized[0]["step"]),
        "step_max": int(normalized[-1]["step"]),
        "expected_workflows_per_step": expected_workflows,
        "expected_global_signal_required": expected_global_signal,
        "missing_metric_steps": {name: count for name, count in missing.items() if count},
        "violations": violations,
        "summary": summary,
        "first_vs_last_quarter": trend,
        "audit": audit,
        "per_step": normalized,
    }


def render_markdown(reports: Mapping[str, Mapping[str, Any]]) -> str:
    def mean_or_na(summary: Mapping[str, Any], key: str, digits: int) -> str:
        stats = summary.get(key)
        if not stats:
            return "n/a"
        return f"{stats['mean']:.{digits}f}"

    lines = [
        "# PSC Training-Signal Audit",
        "",
        "| Run | Steps | Range | Informative questions/step | Interpretable gradient ratio | Gradient cosine | Reason nodes/workflow | Engineering pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, report in reports.items():
        summary = report["summary"]
        lines.append(
            f"| {label} | {report['steps']} | {report['step_min']}-{report['step_max']} | "
            f"{mean_or_na(summary, 'estimated_informative_questions', 3)} | "
            f"{mean_or_na(summary, 'interpretable_gradient_ratio', 4)} | "
            f"{mean_or_na(summary, 'gradient_cosine', 4)} | "
            f"{mean_or_na(summary, 'planned_reason_nodes', 4)} | "
            f"{report['audit']['all_engineering_invariants_pass']} |"
        )
    return "\n".join(lines) + "\n"


def parse_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--log must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--log must use LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def parse_segment_assignment(value: str) -> tuple[str, Path, int, int]:
    if "=" not in value or "@" not in value:
        raise argparse.ArgumentTypeError("--segment must use LABEL=PATH@START:END")
    label, raw_spec = value.split("=", 1)
    raw_path, raw_range = raw_spec.rsplit("@", 1)
    if ":" not in raw_range:
        raise argparse.ArgumentTypeError("--segment must use LABEL=PATH@START:END")
    raw_start, raw_end = raw_range.split(":", 1)
    try:
        start, end = int(raw_start), int(raw_end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("START and END must be integers") from exc
    if not label.strip() or not raw_path.strip() or start <= 0 or end < start:
        raise argparse.ArgumentTypeError("--segment must use LABEL=PATH@START:END")
    return label.strip(), Path(raw_path.strip()), start, end


def parse_bool_assignment(value: str) -> tuple[str, bool]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--require-global-signal must use LABEL=BOOL")
    label, raw_value = value.split("=", 1)
    normalized = raw_value.strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise argparse.ArgumentTypeError("BOOL must be true/false/1/0")
    return label.strip(), normalized in {"true", "1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=parse_assignment, default=[])
    parser.add_argument(
        "--segment",
        action="append",
        type=parse_segment_assignment,
        default=[],
        help="A lineage segment as LABEL=PATH@START:END; repeat for resumed runs.",
    )
    parser.add_argument(
        "--require-global-signal", action="append", type=parse_bool_assignment, default=[]
    )
    parser.add_argument("--expected-workflows", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.log and not args.segment:
        parser.error("at least one --log or --segment is required")
    if args.expected_workflows <= 0:
        raise SystemExit("--expected-workflows must be positive")
    signal_contract = dict(args.require_global_signal)
    labels = {label for label, _ in args.log} | {label for label, *_ in args.segment}
    unknown_labels = set(signal_contract) - labels
    if unknown_labels:
        raise SystemExit(f"Signal contract supplied for unknown labels: {sorted(unknown_labels)}")
    reports: dict[str, dict[str, Any]] = {}
    for label, path in args.log:
        if label in reports or any(segment_label == label for segment_label, *_ in args.segment):
            raise SystemExit(f"Label {label!r} mixes --log with another input")
        reports[label] = {
            "path": str(path.resolve()),
            **analyze(
                parse_steps(path.read_text(encoding="utf-8", errors="replace")),
                expected_workflows=args.expected_workflows,
                expected_global_signal=signal_contract.get(label),
            ),
        }
    grouped_segments: dict[str, list[tuple[Path, int, int]]] = {}
    for label, path, start, end in args.segment:
        grouped_segments.setdefault(label, []).append((path, start, end))
    for label, segments in grouped_segments.items():
        rows, provenance = merge_segments(segments)
        reports[label] = {
            "lineage_segments": provenance,
            **analyze(
                rows,
                expected_workflows=args.expected_workflows,
                expected_global_signal=signal_contract.get(label),
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "psc_training_signal.json").write_text(
        json.dumps({"schema_version": 2, "runs": reports}, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "psc_training_signal.md").write_text(render_markdown(reports), encoding="utf-8")
    print(json.dumps({"schema_version": 2, "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
