#!/usr/bin/env python3
"""Audit Planner-only conditional-value credit on zero-variance EM groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Sequence

from scripts.audit_dagent_planner_conditional_value import (
    rloo_advantages,
    sign,
    weighted_mean,
    weighted_tie_mean,
)


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.mean(rows) if rows else 0.0


def source_name(group_id: str) -> str:
    sample_id = group_id.rsplit(":", 1)[-1].casefold()
    if sample_id.startswith("2wiki_"):
        return "2wiki"
    if sample_id.startswith("hotpot_"):
        return "hotpotqa"
    if sample_id.startswith("musique_"):
        return "musique"
    if sample_id.startswith("reason_search_"):
        return "reason_search"
    return "other"


def cluster_bootstrap_ci(
    values: Sequence[float], seed: int = 20260804, samples: int = 20_000
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975) - 1]]


def logged_rewards(arms: Sequence[dict[str, Any]]) -> list[int]:
    return [int(value) for arm in arms for value in arm["logged_ems"]]


def selector_gain(arms: Sequence[dict[str, Any]]) -> tuple[float, float, float]:
    uniform = weighted_mean(arms, "validation_q")
    selected = weighted_tie_mean(arms, "estimate_q", "validation_q")
    return selected - uniform, selected, uniform


def advantage_accuracy(arms: Sequence[dict[str, Any]]) -> tuple[int, int]:
    estimates = [
        float(arm["estimate_q"])
        for arm in arms
        for _ in range(int(arm["multiplicity"]))
    ]
    validation = [
        float(arm["validation_q"])
        for arm in arms
        for _ in range(int(arm["multiplicity"]))
    ]
    estimate_advantages = rloo_advantages(estimates)
    validation_advantages = rloo_advantages(validation)
    eligible = [
        (estimate, target)
        for estimate, target in zip(estimate_advantages, validation_advantages)
        if sign(target)
    ]
    return sum(sign(estimate) == sign(target) for estimate, target in eligible), len(
        eligible
    )


def summarize(
    results: Sequence[dict[str, Any]],
    *,
    expected_groups: int,
    minimum_eligible_groups: int,
    minimum_informative_groups: int,
    minimum_gain: float,
    minimum_sign_accuracy: float,
) -> dict[str, Any]:
    identity_mismatches = sum(
        not sample.get("identity_match", False)
        for result in results
        for arm in result["arms"]
        for sample in arm["samples"]
    )
    rows = []
    sign_correct = 0
    sign_total = 0
    for result in results:
        arms = result["arms"]
        rewards = logged_rewards(arms)
        if len(set(rewards)) != 1 or len(arms) < 2:
            continue
        gain, selected, uniform = selector_gain(arms)
        correct, total = advantage_accuracy(arms)
        sign_correct += correct
        sign_total += total
        rows.append(
            {
                "group_id": result["group_id"],
                "source": source_name(result["group_id"]),
                "logged_outcome": rewards[0],
                "unique_plan_count": len(arms),
                "estimate_varies": len({float(arm["estimate_q"]) for arm in arms}) > 1,
                "validation_varies": len({float(arm["validation_q"]) for arm in arms}) > 1,
                "uniform_validation": uniform,
                "rescue_selected_validation": selected,
                "rescue_gain": gain,
            }
        )

    gains = [row["rescue_gain"] for row in rows]
    informative = [row for row in rows if row["validation_varies"]]
    signal_groups = [row for row in rows if row["estimate_varies"]]
    by_outcome = {}
    for outcome in (0, 1):
        subset = [row for row in rows if row["logged_outcome"] == outcome]
        by_outcome[str(outcome)] = {
            "groups": len(subset),
            "mean_gain": mean(row["rescue_gain"] for row in subset),
            "bootstrap_95ci": cluster_bootstrap_ci(
                [row["rescue_gain"] for row in subset], seed=20260804 + outcome
            ),
        }
    by_source = {}
    source_regressions = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    for source, subset in sorted(grouped.items()):
        source_gain = mean(row["rescue_gain"] for row in subset)
        by_source[source] = {
            "groups": len(subset),
            "mean_gain": source_gain,
            "bootstrap_95ci": cluster_bootstrap_ci(
                [row["rescue_gain"] for row in subset],
                seed=20260804 + sum(map(ord, source)),
            ),
        }
        if len(subset) >= 5 and source_gain < 0:
            source_regressions.append(source)

    overall_gain = mean(gains)
    overall_ci = cluster_bootstrap_ci(gains)
    sign_accuracy = sign_correct / sign_total if sign_total else 0.0
    outcome_regressions = [
        outcome
        for outcome, metrics in by_outcome.items()
        if metrics["groups"] >= 5 and metrics["mean_gain"] < 0
    ]
    hard_gates = {
        "all_groups_completed": len(results) == expected_groups,
        "identity_controls_exact": identity_mismatches == 0,
        "enough_zero_variance_diverse_groups": len(rows) >= minimum_eligible_groups,
        "validation_signal_present": len(informative) >= minimum_informative_groups,
        "conditional_value_signal_present": len(signal_groups) >= minimum_informative_groups,
        "rescue_gain_large_enough": overall_gain >= minimum_gain,
        "rescue_gain_ci_excludes_zero": overall_ci[0] > 0,
        "advantage_sign_accuracy": sign_accuracy >= minimum_sign_accuracy,
        "no_outcome_slice_regression": not outcome_regressions,
        "no_source_regression": not source_regressions,
        "mixed_groups_exactly_pure_em_by_contract": True,
    }
    return {
        "method": "planner_zero_variance_conditional_value_rescue_v1",
        "contract": (
            "Use Q(plan)-RLOO only when all four logged workflow EM values are "
            "identical; use the original workflow-EM RLOO on every mixed group. "
            "Only Planner tokens receive the rescued advantage."
        ),
        "groups": len(results),
        "eligible_zero_variance_diverse_groups": len(rows),
        "informative_validation_groups": len(informative),
        "conditional_value_signal_groups": len(signal_groups),
        "identity_mismatches": identity_mismatches,
        "selector": {
            "uniform_validation": mean(row["uniform_validation"] for row in rows),
            "rescue_selected_validation": mean(
                row["rescue_selected_validation"] for row in rows
            ),
            "rescue_gain": overall_gain,
            "rescue_gain_cluster_bootstrap_95ci": overall_ci,
        },
        "advantage": {
            "nonzero_validation_advantages": sign_total,
            "conditional_value_sign_accuracy": sign_accuracy,
            "pure_em_sign_accuracy_on_eligible_groups": 0.0,
        },
        "by_logged_outcome": by_outcome,
        "by_source": by_source,
        "source_regressions": source_regressions,
        "outcome_regressions": outcome_regressions,
        "group_results": rows,
        "thresholds": {
            "expected_groups": expected_groups,
            "minimum_eligible_groups": minimum_eligible_groups,
            "minimum_informative_groups": minimum_informative_groups,
            "minimum_gain": minimum_gain,
            "minimum_sign_accuracy": minimum_sign_accuracy,
        },
        "hard_gates": hard_gates,
        "passed": all(hard_gates.values()),
    }


def load_results(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcov-replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, default=64)
    parser.add_argument("--minimum-eligible-groups", type=int, default=20)
    parser.add_argument("--minimum-informative-groups", type=int, default=8)
    parser.add_argument("--minimum-gain", type=float, default=0.05)
    parser.add_argument("--minimum-sign-accuracy", type=float, default=0.60)
    args = parser.parse_args()
    report = summarize(
        load_results(args.pcov_replays),
        expected_groups=args.expected_groups,
        minimum_eligible_groups=args.minimum_eligible_groups,
        minimum_informative_groups=args.minimum_informative_groups,
        minimum_gain=args.minimum_gain,
        minimum_sign_accuracy=args.minimum_sign_accuracy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
