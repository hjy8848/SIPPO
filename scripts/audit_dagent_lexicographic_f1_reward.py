#!/usr/bin/env python3
"""Audit a strict groupwise EM-then-F1 advantage on aligned formal outputs.

The three evaluated policies are treated only as pseudo-siblings. Passing this
offline audit can authorize a current-policy rollout pilot, never training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_DATASETS = {
    "2wikimultihopqa": 500,
    "hotpotqa": 500,
    "musique": 500,
    "bamboogle": 125,
}
DEFAULT_MODELS = ("SFT", "PureEM", "PSC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_assignment(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=COUNT, got {value!r}")
    name, raw_count = value.split("=", 1)
    count = int(raw_count)
    if not name.strip() or count <= 0:
        raise argparse.ArgumentTypeError(f"invalid dataset assignment: {value!r}")
    return name.strip(), count


def _metric(value: Any, *, name: str, row_number: int) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError(f"invalid {name} at CSV row {row_number}: {value!r}")
    return result


def _record_model(
    group: dict[str, Any],
    *,
    label: str,
    prediction: str,
    em: float,
    f1: float,
    row_number: int,
) -> None:
    value = {"prediction": prediction, "em": em, "f1": f1}
    previous = group["models"].get(label)
    if previous is not None and previous != value:
        raise RuntimeError(
            f"inconsistent metrics for {label} at CSV row {row_number}: "
            f"{previous!r} != {value!r}"
        )
    group["models"][label] = value


def load_pseudo_siblings(
    path: Path,
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    datasets: Mapping[str, int] = DEFAULT_DATASETS,
) -> list[dict[str, Any]]:
    expected_models = set(models)
    expected_pairs = {
        frozenset(pair) for pair in itertools.combinations(models, 2)
    }
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "reference", "candidate", "dataset", "question", "gold_answer",
            "reference_prediction", "candidate_prediction", "reference_em",
            "candidate_em", "reference_f1", "candidate_f1",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"paired CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            reference = str(row["reference"])
            candidate = str(row["candidate"])
            if reference not in expected_models or candidate not in expected_models:
                raise RuntimeError(f"unexpected model label at CSV row {row_number}")
            if reference == candidate:
                raise RuntimeError(f"self-pair at CSV row {row_number}")
            dataset = str(row["dataset"])
            if dataset not in datasets:
                raise RuntimeError(f"unexpected dataset at CSV row {row_number}: {dataset}")
            question = str(row["question"]).strip()
            gold = str(row["gold_answer"]).strip()
            if not question or not gold:
                raise RuntimeError(f"empty alignment field at CSV row {row_number}")
            key = (dataset, question, gold)
            group = groups.setdefault(
                key,
                {
                    "dataset": dataset,
                    "question": question,
                    "gold_answer": gold,
                    "models": {},
                    "pairs": set(),
                },
            )
            pair = frozenset((reference, candidate))
            if pair in group["pairs"]:
                raise RuntimeError(f"duplicate model pair at CSV row {row_number}")
            group["pairs"].add(pair)
            _record_model(
                group,
                label=reference,
                prediction=str(row["reference_prediction"]),
                em=_metric(row["reference_em"], name="reference_em", row_number=row_number),
                f1=_metric(row["reference_f1"], name="reference_f1", row_number=row_number),
                row_number=row_number,
            )
            _record_model(
                group,
                label=candidate,
                prediction=str(row["candidate_prediction"]),
                em=_metric(row["candidate_em"], name="candidate_em", row_number=row_number),
                f1=_metric(row["candidate_f1"], name="candidate_f1", row_number=row_number),
                row_number=row_number,
            )

    counts: dict[str, int] = defaultdict(int)
    for key, group in groups.items():
        counts[group["dataset"]] += 1
        if set(group["models"]) != expected_models:
            raise RuntimeError(f"incomplete model set for {key!r}")
        if group["pairs"] != expected_pairs:
            raise RuntimeError(f"incomplete pair set for {key!r}")
        del group["pairs"]
    if dict(counts) != dict(datasets):
        raise RuntimeError(f"dataset counts do not match: {dict(counts)} != {dict(datasets)}")
    return sorted(groups.values(), key=lambda row: (row["dataset"], row["question"], row["gold_answer"]))


def rloo_advantages(rewards: Sequence[float]) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    if values.size < 2:
        raise ValueError("RLOO requires at least two siblings")
    return values - (values.sum() - values) / (values.size - 1)


def _sign(value: float, tolerance: float = 1e-12) -> int:
    return int(value > tolerance) - int(value < -tolerance)


def bootstrap_rate_ci(
    values: Sequence[bool], *, samples: int = 10_000, seed: int = 20260827
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or samples <= 0:
        raise ValueError("bootstrap requires non-empty values and positive samples")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk_size = min(512, samples)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def audit_groups(
    groups: Sequence[dict[str, Any]],
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    em_weight: float = 0.9,
    bootstrap_samples: int = 10_000,
    seed: int = 20260827,
    minimum_new_signal_groups: int = 25,
    minimum_datasets_with_new_signal: int = 3,
    new_signal_rate_ci_low_must_exceed: float = 0.01,
) -> dict[str, Any]:
    if not 0.5 < em_weight < 1.0:
        raise ValueError("EM-preserving audit requires 0.5 < em_weight < 1.0")
    if len(models) < 2:
        raise ValueError("at least two pseudo-sibling policies are required")

    per_dataset: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "groups": 0,
            "mixed_em_groups": 0,
            "all_em_zero_groups": 0,
            "new_f1_signal_groups": 0,
            "mixed_em_sign_mismatches": 0,
        }
    )
    new_signal_flags: list[bool] = []
    mixed_sign_mismatches = 0
    mixed_members_checked = 0
    maximum_mixed_advantage_delta = 0.0
    f1_weight = 1.0 - em_weight
    for group in groups:
        summary = per_dataset[group["dataset"]]
        summary["groups"] += 1
        em = np.asarray([group["models"][label]["em"] for label in models], dtype=np.float64)
        f1 = np.asarray([group["models"][label]["f1"] for label in models], dtype=np.float64)
        if not np.all(np.isin(em, [0.0, 1.0])):
            raise RuntimeError("EM values must be binary")
        mixed = bool(em.min() < em.max())
        all_zero = bool(em.max() == 0.0)
        new_signal = bool(all_zero and np.ptp(f1) > 1e-12)
        new_signal_flags.append(new_signal)
        summary["mixed_em_groups"] += int(mixed)
        summary["all_em_zero_groups"] += int(all_zero)
        summary["new_f1_signal_groups"] += int(new_signal)
        if mixed:
            pure_advantages = rloo_advantages(em)
            # Strict lexicographic semantics: F1 is absent whenever EM differs.
            candidate_advantages = pure_advantages.copy()
            maximum_mixed_advantage_delta = max(
                maximum_mixed_advantage_delta,
                float(np.max(np.abs(candidate_advantages - pure_advantages))),
            )
            for pure_advantage, candidate_advantage in zip(
                pure_advantages, candidate_advantages, strict=True
            ):
                mixed_members_checked += 1
                if _sign(pure_advantage) != _sign(candidate_advantage):
                    mixed_sign_mismatches += 1
                    summary["mixed_em_sign_mismatches"] += 1
        else:
            # EM has no preference in a tied group, so bounded terminal F1 may
            # supply an advantage without changing any EM decision.
            candidate_advantages = f1_weight * rloo_advantages(f1)
            if not np.all(np.isfinite(candidate_advantages)):
                raise RuntimeError("non-finite F1 tie-break advantage")

    total = len(groups)
    new_signal_groups = sum(new_signal_flags)
    datasets_with_new_signal = sum(
        values["new_f1_signal_groups"] > 0 for values in per_dataset.values()
    )
    ci = bootstrap_rate_ci(new_signal_flags, samples=bootstrap_samples, seed=seed)
    gate = {
        "mixed_em_sign_mismatches_must_equal": 0,
        "minimum_new_signal_groups": minimum_new_signal_groups,
        "minimum_datasets_with_new_signal": minimum_datasets_with_new_signal,
        "new_signal_rate_ci_low_must_exceed": new_signal_rate_ci_low_must_exceed,
        "maximum_mixed_em_advantage_delta": 0.0,
        "f1_tiebreak_weight_range": [0.0, 0.5],
    }
    passed = bool(
        mixed_sign_mismatches == 0
        and maximum_mixed_advantage_delta == 0.0
        and new_signal_groups >= gate["minimum_new_signal_groups"]
        and datasets_with_new_signal >= gate["minimum_datasets_with_new_signal"]
        and ci[0] > gate["new_signal_rate_ci_low_must_exceed"]
        and 0.0 < f1_weight < 0.5
    )
    return {
        "models": list(models),
        "pseudo_sibling_group_size": len(models),
        "groups": total,
        "em_weight": em_weight,
        "f1_tiebreak_weight": f1_weight,
        "advantage_rule": (
            "mixed-EM group: exact EM RLOO; EM-tied group: "
            "f1_tiebreak_weight * F1 RLOO"
        ),
        "mixed_em_groups": sum(row["mixed_em_groups"] for row in per_dataset.values()),
        "mixed_em_members_checked": mixed_members_checked,
        "mixed_em_sign_mismatches": mixed_sign_mismatches,
        "maximum_mixed_em_advantage_delta": maximum_mixed_advantage_delta,
        "all_em_zero_groups": sum(row["all_em_zero_groups"] for row in per_dataset.values()),
        "new_f1_signal_groups": new_signal_groups,
        "new_f1_signal_rate": new_signal_groups / total,
        "new_f1_signal_rate_bootstrap_95ci": ci,
        "datasets_with_new_signal": datasets_with_new_signal,
        "per_dataset": dict(per_dataset),
        "gate": gate,
        "passed": passed,
        "advance_to_current_policy_rollout_pilot": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--em-weight", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = tuple(args.models or DEFAULT_MODELS)
    datasets = dict(parse_assignment(value) for value in args.datasets) if args.datasets else dict(DEFAULT_DATASETS)
    groups = load_pseudo_siblings(
        args.paired_csv,
        models=models,
        datasets=datasets,
    )
    audit = audit_groups(
        groups,
        models=models,
        em_weight=args.em_weight,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report = {
        "schema_version": 1,
        "method": "strict_groupwise_em_then_f1_cross_policy_audit_v2",
        "input": {"path": str(args.paired_csv), "sha256": sha256(args.paired_csv)},
        "datasets": datasets,
        "scientific_scope": {
            "terminal_outcome_reward": True,
            "process_or_topology_reward": False,
            "cross_policy_pseudo_siblings": True,
            "current_policy_sibling_evidence": False,
            "one_batch_gradient_check": False,
            "short_training": False,
        },
        "audit": audit,
        "interpretation": (
            "Passing this cross-policy audit only authorizes a frozen-contract, "
            "current-policy four-sibling rollout pilot. It does not authorize an "
            "optimizer update or a short training run."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
