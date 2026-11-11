#!/usr/bin/env python3
"""Validate the two-update gate before allowing the full continuation."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


FATAL_PATTERNS = (
    "Traceback",
    "AssertionError",
    "CUDA out of memory",
    "OutOfMemoryError",
    "NCCL error",
    "RayTaskError",
)


def metric(line: str, name: str) -> float:
    match = re.search(rf"(?:^| - ){re.escape(name)}:([^ ]+)", line)
    if match is None:
        raise ValueError(f"missing metric {name}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {name}: {value}")
    return value


def audit(log: Path) -> dict[str, object]:
    text = log.read_text(encoding="utf-8", errors="replace")
    fatal = [pattern for pattern in FATAL_PATTERNS if pattern in text]
    if fatal:
        raise ValueError(f"fatal training markers: {fatal}")
    lines = {
        step: next(
            (
                line
                for line in text.splitlines()
                if f"step:{step} -" in line and f"training/global_step:{step}" in line
            ),
            None,
        )
        for step in (235, 236)
    }
    if any(line is None for line in lines.values()):
        raise ValueError("gate is missing a complete step235 or step236 metric line")
    reports = {}
    for step, line in lines.items():
        assert line is not None
        completed = metric(line, "train/workflow/completed")
        valid_rate = metric(line, "train/workflow/valid_rate")
        pair_count = metric(line, "aux/planner_outcome_contrast/global_pair_count")
        expected = metric(line, "actor/optimizer_steps_expected")
        executed = metric(line, "actor/optimizer_steps_executed")
        optimizer_step = metric(line, "actor/ppo_optimizer_step")
        grad_norm = metric(line, "actor/grad_norm")
        if completed != 128 or valid_rate != 1 or pair_count <= 0:
            raise ValueError(f"step{step} failed workflow/PSC gates")
        if expected <= 0 or executed != expected or optimizer_step != 1 or grad_norm <= 0:
            raise ValueError(f"step{step} failed optimizer/gradient gates")
        reports[str(step)] = {
            "completed": completed,
            "valid_rate": valid_rate,
            "pair_count": pair_count,
            "optimizer_steps": executed,
            "grad_norm": grad_norm,
        }
    return {"passed": True, "steps": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.log))


if __name__ == "__main__":
    main()
