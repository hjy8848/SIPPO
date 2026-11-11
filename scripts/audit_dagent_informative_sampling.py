#!/usr/bin/env python3
"""Audit a pre-rollout sampler for denser workflow-RLOO signal.

The selector may only use dataset counts known before generation. Post-rollout
reward variance is used as the held-out label, never as an input feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.analyze_dagent_workflow_reward_variance import parse_step_metrics


DATASET_KEYS = (
    "train/data/questions/2wiki",
    "train/data/questions/hotpotqa",
    "train/data/questions/musique",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(log_path: Path, report_path: Path) -> list[dict[str, Any]]:
    metrics = parse_step_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
    variance = json.loads(report_path.read_text(encoding="utf-8"))
    by_step = {int(row["step"]): row for row in variance["per_step"]}
    rows = []
    for metric in metrics:
        step = int(round(metric["training/global_step"]))
        report = by_step.get(step)
        if report is None:
            raise ValueError(f"missing variance report for step {step}")
        questions = int(report["questions"])
        dataset_counts = [int(round(metric.get(key, 0.0))) for key in DATASET_KEYS]
        if sum(dataset_counts) != questions:
            raise ValueError(
                f"step {step} dataset counts {dataset_counts} do not sum to {questions}"
            )
        mixed_count = int(round(report["mixed_signal_rate"] * questions))
        estimated_tokens = (
            metric["prompt_length/mean"] + metric["response_length/mean"]
        ) * metric["train/workflow/completed"]
        rows.append(
            {
                "step": step,
                "dataset_counts": dataset_counts,
                "questions": questions,
                "workflows": int(round(metric["train/workflow/completed"])),
                "mixed_count": mixed_count,
                "estimated_tokens": float(estimated_tokens),
            }
        )
    if sorted(by_step) != [row["step"] for row in rows]:
        raise ValueError("training log and variance report steps do not match exactly")
    return rows


def fit_dataset_signal_rates(rows: Sequence[dict[str, Any]], ridge: float = 1e-3) -> np.ndarray:
    features = np.asarray([row["dataset_counts"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["mixed_count"] for row in rows], dtype=np.float64)
    regularizer = ridge * np.eye(features.shape[1], dtype=np.float64)
    return np.linalg.solve(features.T @ features + regularizer, features.T @ targets)


def _percentile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def evaluate_selector(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    *,
    selection_fraction: float = 0.25,
    bootstrap_samples: int = 10000,
    seed: int = 20260827,
) -> dict[str, Any]:
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be strictly between zero and one")
    if len(validation_rows) < 8:
        raise ValueError("at least eight validation steps are required")

    coefficients = fit_dataset_signal_rates(train_rows)
    features = np.asarray(
        [row["dataset_counts"] for row in validation_rows], dtype=np.float64
    )
    predictions = features @ coefficients
    outcomes = np.asarray([row["mixed_count"] for row in validation_rows], dtype=np.float64)
    questions = np.asarray([row["questions"] for row in validation_rows], dtype=np.float64)
    tokens = np.asarray([row["estimated_tokens"] for row in validation_rows], dtype=np.float64)
    workflows = np.asarray([row["workflows"] for row in validation_rows], dtype=np.int64)
    selected_count = max(1, int(round(len(validation_rows) * selection_fraction)))

    order = np.lexsort(
        (
            np.asarray([row["step"] for row in validation_rows], dtype=np.int64),
            predictions,
        )
    )
    selected = order[-selected_count:]
    selected_rate = float(outcomes[selected].sum() / questions[selected].sum())
    overall_rate = float(outcomes.sum() / questions.sum())
    absolute_uplift = selected_rate - overall_rate
    relative_uplift = selected_rate / overall_rate - 1.0

    rng = np.random.default_rng(seed)
    bootstrap_differences = []
    indices = np.arange(len(validation_rows))
    for _ in range(bootstrap_samples):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        sampled_order = sampled[np.argsort(predictions[sampled], kind="stable")]
        sampled_selected = sampled_order[-selected_count:]
        sampled_rate = outcomes[sampled_selected].sum() / questions[sampled_selected].sum()
        sampled_overall = outcomes[sampled].sum() / questions[sampled].sum()
        bootstrap_differences.append(float(sampled_rate - sampled_overall))

    oracle = np.argsort(outcomes)[-selected_count:]
    token_ratio = float(tokens[selected].mean() / tokens.mean())
    ci = [
        _percentile(bootstrap_differences, 0.025),
        _percentile(bootstrap_differences, 0.975),
    ]
    return {
        "predictor_inputs": list(DATASET_KEYS),
        "post_rollout_features_used": False,
        "dataset_mixed_signal_rate_estimates": {
            key.rsplit("/", 1)[-1]: float(value)
            for key, value in zip(DATASET_KEYS, coefficients, strict=True)
        },
        "validation_steps": len(validation_rows),
        "selected_steps": selected_count,
        "selected_step_ids": sorted(int(validation_rows[index]["step"]) for index in selected),
        "workflows_per_step_values": sorted({int(value) for value in workflows}),
        "overall_mixed_signal_rate": overall_rate,
        "selected_mixed_signal_rate": selected_rate,
        "absolute_uplift": absolute_uplift,
        "absolute_uplift_bootstrap_95ci": ci,
        "relative_uplift": relative_uplift,
        "selected_to_overall_mean_token_ratio": token_ratio,
        "oracle_post_rollout_selected_rate": float(
            outcomes[oracle].sum() / questions[oracle].sum()
        ),
        "oracle_is_non_deployable": True,
        "gate": {
            "minimum_relative_uplift": 0.05,
            "absolute_uplift_ci_low_must_exceed": 0.0,
            "mean_token_ratio_range": [0.95, 1.05],
            "fixed_workflows_per_step_required": True,
        },
        "passed": bool(
            relative_uplift >= 0.05
            and ci[0] > 0.0
            and 0.95 <= token_ratio <= 1.05
            and len(set(workflows.tolist())) == 1
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("pure", "psc"):
        parser.add_argument(f"--{arm}-log", type=Path, required=True)
        parser.add_argument(f"--{arm}-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "pure_log": args.pure_log,
        "pure_report": args.pure_report,
        "psc_log": args.psc_log,
        "psc_report": args.psc_report,
    }
    pure = load_arm(args.pure_log, args.pure_report)
    psc = load_arm(args.psc_log, args.psc_report)
    if [row["step"] for row in pure] != [row["step"] for row in psc]:
        raise ValueError("PureEM and PSC step ids are not matched")
    if [row["dataset_counts"] for row in pure] != [row["dataset_counts"] for row in psc]:
        raise ValueError("PureEM and PSC batch compositions are not matched")

    primary = evaluate_selector(
        pure,
        psc,
        selection_fraction=args.selection_fraction,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    reciprocal = evaluate_selector(
        psc,
        pure,
        selection_fraction=args.selection_fraction,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    report = {
        "schema_version": 1,
        "method": "pre_rollout_dataset_mix_informative_sampling_audit_v1",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "scientific_scope": {
            "reward_candidate": False,
            "sampling_candidate": True,
            "changes_training_data_distribution": True,
            "offline_signal_audit": True,
            "one_batch_gradient_check": False,
            "short_training": False,
        },
        "primary_pure_to_psc": primary,
        "reciprocal_psc_to_pure": reciprocal,
        "advance_to_one_batch_sampling_check": bool(
            primary["passed"] and reciprocal["relative_uplift"] > 0.0
        ),
        "interpretation": (
            "Passing this gate only authorizes a one-batch, one-variable, equal-workflow "
            "and equal-token sampling check. It is not evidence for a new scalar reward "
            "or authorization for a short training run."
        ),
    }
    if not math.isfinite(primary["absolute_uplift"]):
        raise ValueError("non-finite selector result")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
