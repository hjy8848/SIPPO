#!/usr/bin/env python3
"""Decide whether a short OPSD pilot is allowed to continue to formal training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--minimum-em-gain", type=float, default=0.01)
    parser.add_argument("--required-candidate-steps", default="10,20")
    args = parser.parse_args()
    args.required_candidate_steps = {
        int(value)
        for value in args.required_candidate_steps.split(",")
        if value.strip()
    }
    if not args.required_candidate_steps:
        parser.error("--required-candidate-steps must not be empty")
    if args.minimum_em_gain < 0:
        parser.error("--minimum-em-gain must be non-negative")
    return args


def decide(
    rows: list[dict[str, str]],
    minimum_em_gain: float,
    required_candidate_steps: set[int],
) -> dict[str, object]:
    by_step = {int(row["step"]): row for row in rows}
    missing = sorted(({0} | required_candidate_steps) - set(by_step))
    if missing:
        return {
            "formal_training_allowed": False,
            "reason": "missing_required_evaluations",
            "missing_steps": missing,
        }
    base = by_step[0]
    candidates = [by_step[step] for step in sorted(required_candidate_steps)]
    best = max(candidates, key=lambda row: (float(row["em"]), -int(row["step"])))
    base_em = float(base["em"])
    best_em = float(best["em"])
    em_gain = best_em - base_em
    errors = int(best.get("api_errors") or 0) + int(
        best.get("pipeline_errors") or 0
    )
    allowed = em_gain >= minimum_em_gain and errors == 0
    reasons = []
    if em_gain < minimum_em_gain:
        reasons.append("heldout_em_gain_below_gate")
    if errors:
        reasons.append("candidate_has_pipeline_errors")
    return {
        "formal_training_allowed": allowed,
        "reason": "passed" if allowed else ",".join(reasons),
        "base_step": 0,
        "base_em": base_em,
        "best_candidate_step": int(best["step"]),
        "best_candidate_em": best_em,
        "best_candidate_f1": float(best["f1"]),
        "heldout_em_gain": em_gain,
        "minimum_required_em_gain": minimum_em_gain,
        "candidate_errors": errors,
        "required_candidate_steps": sorted(required_candidate_steps),
        "best_adapter": best["adapter"],
        "best_merged_model": best["merged_model"],
    }


def main() -> int:
    args = parse_args()
    with args.metrics_file.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    decision = decide(
        rows,
        args.minimum_em_gain,
        args.required_candidate_steps,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
