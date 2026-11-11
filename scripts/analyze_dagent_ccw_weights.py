#!/usr/bin/env python3
"""Analyze bounded DAG-CCW weights from causal replay artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence


DEFAULT_ALPHA = math.log(1.5)


def clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "p75": quantile(values, 0.75),
        "max": max(values) if values else 0.0,
    }


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def analyze(
    trajectories: Sequence[dict[str, Any]],
    alpha: float = DEFAULT_ALPHA,
    minimum_weight: float = 0.5,
    maximum_weight: float = 1.5,
) -> dict[str, Any]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not (0 < minimum_weight <= 1 <= maximum_weight):
        raise ValueError("weight bounds must be positive and contain 1")

    cases = []
    by_role: dict[str, list[float]] = defaultdict(list)
    by_dataset: dict[str, list[float]] = defaultdict(list)
    categories = Counter()
    for trajectory in trajectories:
        dataset = str(trajectory.get("dataset", "unknown"))
        for case in trajectory.get("hcapo_cases", []):
            factual = float(case["factual_em"])
            counterfactual = float(case["counterfactual_em"])
            if factual not in {0.0, 1.0} or counterfactual not in {0.0, 1.0}:
                raise ValueError("deterministic audit expects binary EM outcomes")
            advantage_sign = 1.0 if factual > 0.0 else -1.0
            causal_delta = factual - counterfactual
            signed_credit = advantage_sign * causal_delta
            weight = clip(
                math.exp(alpha * signed_credit),
                minimum_weight,
                maximum_weight,
            )
            weighted_advantage = weight * advantage_sign
            role = str(case.get("node_type", "unknown"))
            category = (
                "necessary_success_action"
                if factual == 1.0 and counterfactual == 0.0
                else "harmful_failed_action"
                if factual == 0.0 and counterfactual == 1.0
                else "outcome_unchanged"
            )
            categories[category] += 1
            by_role[role].append(weight)
            by_dataset[dataset].append(weight)
            cases.append(
                {
                    "case_id": case.get("case_id"),
                    "dataset": dataset,
                    "role": role,
                    "factual_em": factual,
                    "counterfactual_em": counterfactual,
                    "causal_delta_em": causal_delta,
                    "advantage_sign_proxy": advantage_sign,
                    "signed_credit": signed_credit,
                    "weight": weight,
                    "weighted_advantage_proxy": weighted_advantage,
                    "category": category,
                }
            )

    weights = [case["weight"] for case in cases]
    credits = [case["signed_credit"] for case in cases]
    direction_preserved = all(
        case["advantage_sign_proxy"] * case["weighted_advantage_proxy"] > 0
        for case in cases
    )
    within_bounds = all(minimum_weight <= weight <= maximum_weight for weight in weights)
    changed = [case for case in cases if case["causal_delta_em"] != 0]
    neutral = [case for case in cases if case["causal_delta_em"] == 0]

    report = {
        "schema_version": 1,
        "method": "deterministic_deletion_only_dag_ccw_weight_diagnostic",
        "formula": {
            "credit": "sign(A_EM) * (EM_factual - EM_counterfactual)",
            "weight": "clip(exp(alpha * credit), min_weight, max_weight)",
            "alpha": alpha,
            "minimum_weight": minimum_weight,
            "maximum_weight": maximum_weight,
            "alpha_selection": "pre_registered_log_1.5_no_sweep",
        },
        "trajectories": len(trajectories),
        "cases": len(cases),
        "changed_cases": len(changed),
        "categories": dict(categories),
        "weights": summary(weights),
        "credits": summary(credits),
        "changed_weight_mean": statistics.mean(case["weight"] for case in changed)
        if changed
        else 0.0,
        "unchanged_weight_mean": statistics.mean(case["weight"] for case in neutral)
        if neutral
        else 0.0,
        "weights_by_role": {role: summary(values) for role, values in sorted(by_role.items())},
        "weights_by_dataset": {
            dataset: summary(values) for dataset, values in sorted(by_dataset.items())
        },
        "invariants": {
            "all_weights_finite": all(math.isfinite(weight) for weight in weights),
            "all_weights_within_bounds": within_bounds,
            "em_advantage_direction_preserved": direction_preserved,
            "new_signal_has_no_standalone_gradient": True,
        },
        "diagnostic_limitations": {
            "uses_rloo_advantage_sign": False,
            "advantage_sign_is_binary_outcome_proxy": True,
            "uses_multiple_suffix_samples": False,
            "deletion_only_binary_em_can_produce_negative_credit": any(
                credit < 0 for credit in credits
            ),
            "requires_stochastic_paired_suffix_audit_before_training": True,
        },
        "case_records": cases,
    }
    report["passed_safety_invariants"] = bool(
        report["invariants"]["all_weights_finite"]
        and within_bounds
        and direction_preserved
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-replays", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--minimum-weight", type=float, default=0.5)
    parser.add_argument("--maximum-weight", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        load_rows(args.causal_replays),
        alpha=args.alpha,
        minimum_weight=args.minimum_weight,
        maximum_weight=args.maximum_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    printable = {key: value for key, value in report.items() if key != "case_records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
