#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "OutOfMemoryError",
    "CUDA out of memory",
    "NCCL error",
    "CUDA error",
    "Error executing job with overrides",
)


def metric(text: str, name: str) -> float:
    matches = re.findall(rf"(?:^|\s){re.escape(name)}:([^\s]+)", text)
    if not matches:
        raise ValueError(f"missing smoke metric: {name}")
    value = float(matches[-1])
    if not math.isfinite(value):
        raise ValueError(f"non-finite smoke metric: {name}={value}")
    return value


def validate(text: str, expected_step: int, expected_workflows: int, poc_mode: str) -> None:
    clean = ANSI_ESCAPE.sub("", text)
    for pattern in FATAL_PATTERNS:
        if pattern in clean:
            raise ValueError(f"fatal pattern in smoke log: {pattern}")

    if int(metric(clean, "training/global_step")) != expected_step:
        raise ValueError("smoke did not complete the expected global step")
    if int(metric(clean, "train/workflow/completed")) != expected_workflows:
        raise ValueError("smoke workflow count does not match the contract")
    if int(metric(clean, "train/workflow/incomplete_question_groups")) != 0:
        raise ValueError("smoke contains incomplete question groups")

    expected = metric(clean, "actor/optimizer_steps_expected")
    executed = metric(clean, "actor/optimizer_steps_executed")
    if expected <= 0 or executed != expected:
        raise ValueError(f"optimizer mismatch: expected={expected}, executed={executed}")
    metric(clean, "actor/pg_loss")
    metric(clean, "actor/grad_norm")

    poc_enabled = metric(clean, "aux/planner_outcome_contrast/enabled")
    wanted = 1.0 if poc_mode == "real" else 0.0
    if poc_enabled != wanted:
        raise ValueError(f"PSC mode mismatch: expected enabled={wanted}, found {poc_enabled}")
    if poc_mode == "real":
        pair_count = metric(clean, "aux/planner_outcome_contrast/global_pair_count")
        informative = metric(
            clean, "aux/planner_outcome_contrast/global_informative_question_count"
        )
        signal_required = metric(
            clean, "aux/planner_outcome_contrast/global_signal_required"
        )
        empty_minibatch = metric(
            clean, "aux/planner_outcome_contrast/empty_refreshed_minibatch"
        )
        psc_grad_norm = metric(
            clean,
            "aux/planner_outcome_contrast/refreshed_weighted_poc_direct_gradient_norm",
        )
        if pair_count <= 0 or informative <= 0:
            raise ValueError("real PSC smoke has no eligible sibling signal")
        if signal_required != 1 or empty_minibatch != 0:
            raise ValueError("real PSC smoke did not satisfy the required-signal gate")
        if psc_grad_norm <= 0:
            raise ValueError("real PSC smoke has no direct PSC gradient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-gate a one-step DAGENT training smoke log.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=1)
    parser.add_argument("--expected-workflows", type=int, default=128)
    parser.add_argument("--expected-poc-mode", choices=("off", "real"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(
        args.log.read_text(encoding="utf-8", errors="replace"),
        expected_step=args.expected_step,
        expected_workflows=args.expected_workflows,
        poc_mode=args.expected_poc_mode,
    )
    print(
        f"[PASS] smoke hard gate: step={args.expected_step}, "
        f"workflows={args.expected_workflows}, poc_mode={args.expected_poc_mode}"
    )


if __name__ == "__main__":
    main()
