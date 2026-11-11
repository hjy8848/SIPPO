#!/usr/bin/env python3
"""Summarize pure Self-SOD training metrics from the console log."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_PATTERN = re.compile(r"(?:^|\s)step:(\d+)(?:\s|$)")
NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)

ROLES = (
    "PlanningAgent",
    "QueryRewriteAgent",
    "ReasonerAgent",
    "AnswerSummarizationAgent",
    "AnswerGenerationAgent",
)

ROLE_TREND_METRICS = (
    "scored_row_count",
    "generated_row_count",
    "coverage",
    "teacher_gap_mean",
    "teacher_endorsement_rate",
    "divergence_mean",
    "weight_mean",
)

MECHANISM_COUNT_METRICS = (
    "aux/workflow_sod/included_invalid_target_trajectory_count",
    "aux/workflow_sod/scored_invalid_target_row_count",
    "aux/workflow_sod/missing_action_reference_count",
    "aux/workflow_sod/topology_mismatch_trajectory_count",
    "aux/workflow_sod/extra_reason_trajectory_count",
    "aux/workflow_sod/missing_reason_trajectory_count",
)

DERIVED_TREND_METRICS = (
    "derived/workflow_sod/active_trajectory_fraction",
    "derived/workflow_sod/scored_action_fraction",
    "derived/workflow_sod/missing_reference_fraction",
    "derived/workflow_sod/topology_mismatch_fraction",
    "derived/workflow_sod/invalid_target_row_fraction",
)

TREND_METRICS = (
    "train/workflow/valid_rate",
    "train/workflow/retrieval_calls_mean",
    "train/workflow/planned_search_nodes_mean",
    "train/workflow/planned_reason_nodes_mean",
    "train/workflow/summary_first_valid_rate",
    "train/rollout/em_aggregate",
    "train/rollout/best_of_n_em",
    "train/rollout/within_question_em_std",
    "actor/entropy",
    "aux/workflow_sod/questions_with_successful_sibling",
    "aux/workflow_sod/scored_row_count",
    "aux/workflow_sod/response_token_count",
    "aux/workflow_sod/empty_signal_step",
    "aux/workflow_sod/actor_update_skipped_no_signal",
    "aux/workflow_sod/trajectory_count",
    "aux/workflow_sod/action_count",
    "aux/workflow_sod/aligned_reference_rate",
    "aux/workflow_sod/missing_action_reference_count",
    "aux/workflow_sod/topology_mismatch_trajectory_count",
    "aux/workflow_sod/extra_reason_trajectory_count",
    "aux/workflow_sod/missing_reason_trajectory_count",
    "aux/workflow_sod/teacher_gap_mean",
    "aux/workflow_sod/teacher_endorsement_rate",
    "aux/workflow_sod/divergence_mean",
    "aux/workflow_sod/divergence_first_mean",
    "aux/workflow_sod/divergence_last_mean",
    "aux/workflow_sod/weight_mean",
    "aux/workflow_sod/attenuated_action_fraction",
    "aux/workflow_sod/amplified_action_fraction",
    "aux/workflow_sod/divergence_increase_trajectory_fraction",
    "aux/workflow_sod/recovery_after_increase_trajectory_fraction",
    "aux/workflow_sod/local_advantage_mean",
    "aux/workflow_sod/local_advantage_positive_fraction",
    "aux/workflow_sod/token_weight_mean",
    "aux/workflow_sod/added_advantage_rms",
    "aux/workflow_sod/forward_kl_duplicate_eligible_suppressed_count",
    "aux/workflow_sod/opening_tag_mask_fallback_count",
    "aux/workflow_sod/opening_tag_excluded_token_count",
    "aux/workflow_sod_forward_kl/forward_kl",
    "aux/workflow_sod_forward_kl/full_response_forward_kl",
    "aux/workflow_sod_forward_kl/gate_forward_kl",
    "aux/workflow_sod_forward_kl/format_prefix_forward_kl",
    "aux/workflow_sod_forward_kl/response_body_forward_kl",
    "aux/workflow_sod_forward_kl/teacher_confidence",
    "aux/workflow_sod_forward_kl/teacher_top1_match_rate",
    "aux/workflow_sod_forward_kl/selected_token_count",
    "aux/workflow_sod_forward_kl/full_selected_token_count",
    "aux/workflow_sod_forward_kl/loss_token_start",
    "aux/workflow_sod_forward_kl/exclude_opening_tag_from_loss",
    "aux/workflow_sod_forward_kl/excluded_opening_tag_token_count",
    "aux/workflow_sod_forward_kl/gate_on_full_response_kl",
    "aux/workflow_sod_forward_kl/gate_weight",
    "aux/workflow_sod_forward_kl/loss",
    "aux/workflow_sod_forward_kl/accepted_pair_count",
    "aux/workflow_sod_forward_kl/optimizer_accepted_pair_count",
    "aux/workflow_sod_forward_kl/accepted_slot_count",
    "aux/workflow_sod_forward_kl/gradient_scale",
    "aux/workflow_sod_forward_kl/grad_norm",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/pg_clipfrac_lower",
    "actor/ppo_kl",
    "actor/grad_norm",
    "response_length/mean",
    "prompt_length/mean",
    *DERIVED_TREND_METRICS,
)


def parse_numeric(value: str) -> float | None:
    value = value.strip()
    if not NUMBER_PATTERN.fullmatch(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def parse_step_line(line: str) -> tuple[int, dict[str, float]] | None:
    clean = ANSI_ESCAPE.sub("", line).replace("\r", "")
    match = STEP_PATTERN.search(clean)
    if match is None:
        return None
    step = int(match.group(1))
    metrics: dict[str, float] = {}
    payload = clean[match.end() :].lstrip()
    if payload.startswith("- "):
        payload = payload[2:]
    for field in payload.split(" - "):
        if ":" not in field:
            continue
        key, raw_value = field.rsplit(":", 1)
        number = parse_numeric(raw_value)
        if number is not None:
            metrics[key.strip()] = number
    return step, metrics


def read_steps(path: Path) -> tuple[dict[int, dict[str, float]], list[int]]:
    steps: dict[int, dict[str, float]] = {}
    seen: Counter[int] = Counter()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = parse_step_line(line)
            if parsed is None:
                continue
            step, metrics = parsed
            steps[step] = metrics
            seen[step] += 1
    duplicates = sorted(step for step, count in seen.items() if count > 1)
    return steps, duplicates


def linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = (len(values) - 1) / 2
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    if denominator == 0:
        return 0.0
    mean = statistics.fmean(values)
    return sum(
        (index - center) * (value - mean) for index, value in enumerate(values)
    ) / denominator


def longest_consecutive_run(steps: list[int]) -> int:
    longest = 0
    current = 0
    previous = None
    for step in sorted(steps):
        current = current + 1 if previous is not None and step == previous + 1 else 1
        longest = max(longest, current)
        previous = step
    return longest


def summarize_metric(
    records: list[tuple[int, dict[str, float]]], key: str, window: int
) -> dict[str, Any] | None:
    points = [(step, metrics[key]) for step, metrics in records if key in metrics]
    if not points:
        return None
    values = [value for _, value in points]
    size = min(window, len(values))
    return {
        "count": len(values),
        "first_step": points[0][0],
        "last_step": points[-1][0],
        "first_window_mean": statistics.fmean(values[:size]),
        "last_window_mean": statistics.fmean(values[-size:]),
        "delta_last_minus_first_window": (
            statistics.fmean(values[-size:]) - statistics.fmean(values[:size])
        ),
        "minimum": min(values),
        "maximum": max(values),
        "slope_per_step": linear_slope(values),
    }


def _add_ratio(
    metrics: dict[str, float], output_key: str, numerator_key: str, denominator_key: str
) -> None:
    numerator = metrics.get(numerator_key)
    denominator = metrics.get(denominator_key)
    if numerator is not None and denominator is not None and denominator > 0:
        metrics[output_key] = numerator / denominator


def add_derived_metrics(
    records: list[tuple[int, dict[str, float]]],
) -> list[tuple[int, dict[str, float]]]:
    enriched = []
    ratio_specs = (
        (
            "derived/workflow_sod/active_trajectory_fraction",
            "aux/workflow_sod/trajectory_count",
            "actor/workflow_loss_workflow_count",
        ),
        (
            "derived/workflow_sod/scored_action_fraction",
            "aux/workflow_sod/scored_row_count",
            "actor/workflow_loss_logical_action_count",
        ),
        (
            "derived/workflow_sod/missing_reference_fraction",
            "aux/workflow_sod/missing_action_reference_count",
            "aux/workflow_sod/scored_row_count",
        ),
        (
            "derived/workflow_sod/topology_mismatch_fraction",
            "aux/workflow_sod/topology_mismatch_trajectory_count",
            "aux/workflow_sod/fixed_reference_trajectory_count",
        ),
        (
            "derived/workflow_sod/invalid_target_row_fraction",
            "aux/workflow_sod/scored_invalid_target_row_count",
            "aux/workflow_sod/scored_row_count",
        ),
    )
    for step, raw_metrics in records:
        metrics = dict(raw_metrics)
        for output_key, numerator_key, denominator_key in ratio_specs:
            _add_ratio(metrics, output_key, numerator_key, denominator_key)
        enriched.append((step, metrics))
    return enriched


def summarize_mechanism(
    records: list[tuple[int, dict[str, float]]], window: int
) -> dict[str, Any]:
    count_totals = {
        key: sum(metrics.get(key, 0.0) for _, metrics in records)
        for key in MECHANISM_COUNT_METRICS
    }
    derived_trends = {
        key: summary
        for key in DERIVED_TREND_METRICS
        if (summary := summarize_metric(records, key, window)) is not None
    }
    roles: dict[str, Any] = {}
    for role in ROLES:
        role_trends = {}
        for metric in ROLE_TREND_METRICS:
            key = f"aux/workflow_sod/{metric}/{role}"
            summary = summarize_metric(records, key, window)
            if summary is not None:
                role_trends[metric] = summary
        scored_key = f"aux/workflow_sod/scored_row_count/{role}"
        generated_key = f"aux/workflow_sod/generated_row_count/{role}"
        scored_total = sum(metrics.get(scored_key, 0.0) for _, metrics in records)
        generated_total = sum(
            metrics.get(generated_key, 0.0) for _, metrics in records
        )
        roles[role] = {
            "scored_row_total": scored_total,
            "generated_row_total": generated_total,
            "aggregate_row_coverage": (
                scored_total / generated_total if generated_total > 0 else None
            ),
            "trends": role_trends,
        }
    return {
        "count_totals": count_totals,
        "derived_trends": derived_trends,
        "roles": roles,
    }


def audit_hard_gates(
    records: list[tuple[int, dict[str, float]]],
    expected_dp_size: float | None,
    expected_minibatch_workflows: float | None,
    expected_total_mass: float | None,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    failures: list[dict[str, Any]] = []
    missing: dict[str, list[int]] = {}
    gates: list[tuple[str, float, float]] = [
        ("aux/workflow_sod/sod_only", 1.0, tolerance),
        ("aux/workflow_sod/task_advantage_rms", 0.0, tolerance),
        ("aux/workflow_sod/reference_switch_count", 0.0, tolerance),
        ("aux/workflow_sod/future_reference_action_count", 0.0, tolerance),
    ]
    if expected_dp_size is not None:
        gates.append(("actor/workflow_dp_size", expected_dp_size, tolerance))
    if expected_total_mass is not None:
        gates.append(
            ("actor/workflow_loss_total_mass", expected_total_mass, 1e-4)
        )

    for key, expected, gate_tolerance in gates:
        missing_steps = []
        for step, metrics in records:
            if key not in metrics:
                missing_steps.append(step)
                continue
            actual = metrics[key]
            if abs(actual - expected) > gate_tolerance:
                failures.append(
                    {
                        "step": step,
                        "metric": key,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        if missing_steps:
            missing[key] = missing_steps

    for step, metrics in records:
        skipped = metrics.get(
            "aux/workflow_sod/actor_update_skipped_no_signal", 0.0
        )
        empty_signal = metrics.get("aux/workflow_sod/empty_signal_step", 0.0)
        if abs(skipped - empty_signal) > tolerance:
            failures.append(
                {
                    "step": step,
                    "metric": "aux/workflow_sod/actor_update_skipped_no_signal",
                    "expected": empty_signal,
                    "actual": skipped,
                }
            )
        gate_on_full = metrics.get(
            "aux/workflow_sod_forward_kl/gate_on_full_response_kl", 0.0
        )
        if gate_on_full > 0.5 and skipped <= 0.5:
            gate_key = "aux/workflow_sod_forward_kl/gate_forward_kl"
            full_key = "aux/workflow_sod_forward_kl/full_response_forward_kl"
            if gate_key not in metrics:
                missing.setdefault(gate_key, []).append(step)
            if full_key not in metrics:
                missing.setdefault(full_key, []).append(step)
            if gate_key in metrics and full_key in metrics:
                gate_kl = metrics[gate_key]
                full_kl = metrics[full_key]
                if abs(gate_kl - full_kl) > tolerance:
                    failures.append(
                        {
                            "step": step,
                            "metric": gate_key,
                            "expected": full_kl,
                            "actual": gate_kl,
                        }
                    )
        exclude_opening = metrics.get(
            "aux/workflow_sod_forward_kl/exclude_opening_tag_from_loss", 0.0
        )
        if exclude_opening > 0.5 and skipped <= 0.5:
            loss_start_key = "aux/workflow_sod_forward_kl/loss_token_start"
            actual_loss_start = metrics.get(loss_start_key)
            if actual_loss_start is None:
                missing.setdefault(loss_start_key, []).append(step)
            elif abs(actual_loss_start) > tolerance:
                failures.append(
                    {
                        "step": step,
                        "metric": loss_start_key,
                        "expected": 0.0,
                        "actual": actual_loss_start,
                    }
                )
        forward_kl_only = metrics.get("aux/workflow_sod/forward_kl_only", 0.0)
        expected_optimizer_steps = (
            0.0 if skipped > 0.5 else (1.0 if forward_kl_only > 0.5 else 4.0)
        )
        actual_optimizer_steps = metrics.get("actor/optimizer_steps_expected")
        if actual_optimizer_steps is None:
            missing.setdefault("actor/optimizer_steps_expected", []).append(step)
        elif abs(actual_optimizer_steps - expected_optimizer_steps) > tolerance:
            failures.append(
                {
                    "step": step,
                    "metric": "actor/optimizer_steps_expected",
                    "expected": expected_optimizer_steps,
                    "actual": actual_optimizer_steps,
                }
            )
        if expected_minibatch_workflows is not None and skipped <= 0.5:
            key = "actor/workflow_complete_workflows_per_minibatch"
            actual_minibatch_workflows = metrics.get(key)
            if actual_minibatch_workflows is None:
                missing.setdefault(key, []).append(step)
            elif (
                abs(actual_minibatch_workflows - expected_minibatch_workflows)
                > tolerance
            ):
                failures.append(
                    {
                        "step": step,
                        "metric": key,
                        "expected": expected_minibatch_workflows,
                        "actual": actual_minibatch_workflows,
                    }
                )
        for key in (
            "aux/workflow_sod/questions_with_successful_sibling",
            "aux/workflow_sod/scored_row_count",
            "aux/workflow_sod/response_token_count",
        ):
            if key in metrics and metrics[key] <= 0 and skipped <= 0.5:
                failures.append(
                    {
                        "step": step,
                        "metric": key,
                        "expected": "> 0",
                        "actual": metrics[key],
                    }
                )
    return failures, missing


def analyze(
    path: Path,
    *,
    window: int = 10,
    expected_dp_size: float | None = None,
    expected_minibatch_workflows: float | None = None,
    expected_total_mass: float | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    steps, duplicates = read_steps(path)
    records = add_derived_metrics(sorted(steps.items()))
    if not records:
        raise RuntimeError(f"No completed training steps found in {path}")
    first_step = records[0][0]
    last_step = records[-1][0]
    missing_steps = sorted(set(range(first_step, last_step + 1)) - set(steps))
    failures, missing_gate_metrics = audit_hard_gates(
        records,
        expected_dp_size,
        expected_minibatch_workflows,
        expected_total_mass,
        tolerance,
    )
    empty_signal_steps = [
        step
        for step, metrics in records
        if metrics.get("aux/workflow_sod/empty_signal_step", 0.0) > 0.5
    ]
    optimizer_update_steps = [
        step
        for step, metrics in records
        if metrics.get("actor/optimizer_steps_expected", 0.0) > 0
    ]
    optimizer_step_count = sum(
        metrics.get("actor/optimizer_steps_expected", 0.0)
        for _, metrics in records
    )

    trend_keys = list(TREND_METRICS)
    trend_keys.extend(f"aux/workflow_sod/coverage/{role}" for role in ROLES)
    trends = {
        key: summary
        for key in trend_keys
        if (summary := summarize_metric(records, key, window)) is not None
    }
    return {
        "log": str(path),
        "completed_step_count": len(records),
        "first_step": first_step,
        "last_step": last_step,
        "missing_steps": missing_steps,
        "duplicate_step_lines": duplicates,
        "hard_gate_failures": failures,
        "missing_hard_gate_metrics": missing_gate_metrics,
        "empty_signal_steps": empty_signal_steps,
        "empty_signal_step_fraction": len(empty_signal_steps) / len(records),
        "longest_consecutive_empty_signal_steps": longest_consecutive_run(
            empty_signal_steps
        ),
        "optimizer_update_steps": optimizer_update_steps,
        "effective_actor_update_step_count": len(optimizer_update_steps),
        "optimizer_step_count": optimizer_step_count,
        "mechanism": summarize_mechanism(records, window),
        "trends": trends,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--expected-dp-size", type=float)
    parser.add_argument("--expected-minibatch-workflows", type=float)
    parser.add_argument("--expected-total-mass", type=float)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.window < 1:
        raise ValueError("--window must be positive")
    report = analyze(
        args.log,
        window=args.window,
        expected_dp_size=args.expected_dp_size,
        expected_minibatch_workflows=args.expected_minibatch_workflows,
        expected_total_mass=args.expected_total_mass,
        tolerance=args.tolerance,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
