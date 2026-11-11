#!/usr/bin/env python3
"""Recover per-question binary-reward histograms from DAGENT training logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"(?:^| - )([^:\n]+):([^\s]+)")


def parse_step_metrics(text: str) -> list[dict[str, float]]:
    rows = []
    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line).replace("\r", "")
        if "training/global_step:" not in line or "train/rollout/em_aggregate:" not in line:
            continue
        metrics: dict[str, float] = {}
        for key, value in METRIC_RE.findall(line):
            key = key.strip().split()[-1]
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        if "training/global_step" in metrics:
            rows.append(metrics)
    return rows


def _compositions(total: int, parts: int, prefix: tuple[int, ...] = ()) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from _compositions(total - value, parts - 1, prefix + (value,))


def histogram_metrics(histogram: tuple[int, ...]) -> tuple[float, float, float]:
    rollout_n = len(histogram) - 1
    question_count = sum(histogram)
    success_count = sum(successes * count for successes, count in enumerate(histogram))
    em = success_count / (question_count * rollout_n)
    best_of_n = (question_count - histogram[0]) / question_count
    mean_std = sum(
        count * math.sqrt((successes / rollout_n) * (1.0 - successes / rollout_n))
        for successes, count in enumerate(histogram)
    ) / question_count
    return em, best_of_n, mean_std


def reconstruct_histogram(
    question_count: int,
    rollout_n: int,
    em: float,
    best_of_n: float,
    mean_std: float,
    tolerance: float = 1e-6,
) -> tuple[int, ...]:
    best: tuple[float, tuple[int, ...]] | None = None
    for histogram in _compositions(question_count, rollout_n + 1):
        candidate = histogram_metrics(histogram)
        error = max(abs(candidate[0] - em), abs(candidate[1] - best_of_n), abs(candidate[2] - mean_std))
        if best is None or error < best[0]:
            best = (error, histogram)
    assert best is not None
    if best[0] > tolerance:
        raise ValueError(
            "could not reconstruct sibling reward histogram: "
            f"best_error={best[0]:.8g}, questions={question_count}, n={rollout_n}"
        )
    return best[1]


def analyze_rows(rows: list[dict[str, float]], rollout_n: int) -> dict:
    step_reports = []
    aggregate = Counter()
    for metrics in rows:
        completed = int(round(metrics["train/workflow/completed"]))
        if completed % rollout_n:
            raise ValueError(f"step {metrics['training/global_step']} has incomplete rollout groups")
        question_count = completed // rollout_n
        histogram = reconstruct_histogram(
            question_count=question_count,
            rollout_n=rollout_n,
            em=metrics["train/rollout/em_aggregate"],
            best_of_n=metrics["train/rollout/best_of_n_em"],
            mean_std=metrics["train/rollout/within_question_em_std"],
        )
        for successes, count in enumerate(histogram):
            aggregate[successes] += count
        mixed = question_count - histogram[0] - histogram[-1]
        step_reports.append(
            {
                "step": int(round(metrics["training/global_step"])),
                "questions": question_count,
                "reward_histogram": {str(index): count for index, count in enumerate(histogram)},
                "all_failed_rate": histogram[0] / question_count,
                "all_success_rate": histogram[-1] / question_count,
                "mixed_signal_rate": mixed / question_count,
                "em": metrics["train/rollout/em_aggregate"],
                "best_of_n_em": metrics["train/rollout/best_of_n_em"],
                "within_question_em_std": metrics["train/rollout/within_question_em_std"],
            }
        )

    total_questions = sum(aggregate.values())
    mixed_questions = total_questions - aggregate[0] - aggregate[rollout_n]
    return {
        "steps": len(step_reports),
        "step_min": min((row["step"] for row in step_reports), default=None),
        "step_max": max((row["step"] for row in step_reports), default=None),
        "rollout_n": rollout_n,
        "questions": total_questions,
        "workflows": total_questions * rollout_n,
        "reward_histogram": {str(index): aggregate[index] for index in range(rollout_n + 1)},
        "all_failed_questions": aggregate[0],
        "all_failed_rate": aggregate[0] / total_questions if total_questions else 0.0,
        "all_success_questions": aggregate[rollout_n],
        "all_success_rate": aggregate[rollout_n] / total_questions if total_questions else 0.0,
        "mixed_signal_questions": mixed_questions,
        "mixed_signal_rate": mixed_questions / total_questions if total_questions else 0.0,
        "zero_variance_rate": (
            (aggregate[0] + aggregate[rollout_n]) / total_questions if total_questions else 0.0
        ),
        "effective_advantage_workflows": mixed_questions * rollout_n,
        "effective_advantage_workflow_rate": (
            mixed_questions / total_questions if total_questions else 0.0
        ),
        "per_step": step_reports,
    }


def render_markdown(report: dict) -> str:
    histogram = report["reward_histogram"]
    lines = [
        "# DAGENT Workflow Reward-Variance Audit",
        "",
        f"- Steps: {report['step_min']} to {report['step_max']} ({report['steps']} total)",
        f"- Questions/workflows: {report['questions']} / {report['workflows']}",
        f"- All-failed questions: {report['all_failed_questions']} ({report['all_failed_rate']:.2%})",
        f"- All-success questions: {report['all_success_questions']} ({report['all_success_rate']:.2%})",
        f"- Mixed-signal questions: {report['mixed_signal_questions']} ({report['mixed_signal_rate']:.2%})",
        f"- Zero-variance questions: {report['zero_variance_rate']:.2%}",
        f"- Workflows with non-zero RLOO advantage: {report['effective_advantage_workflows']} "
        f"({report['effective_advantage_workflow_rate']:.2%})",
        "",
        "## Sibling Success Histogram",
        "",
        "| Successful siblings | Questions | Rate |",
        "|---:|---:|---:|",
    ]
    for successes in range(report["rollout_n"] + 1):
        count = histogram[str(successes)]
        lines.append(f"| {successes}/{report['rollout_n']} | {count} | {count / report['questions']:.2%} |")
    lines.extend(
        [
            "",
            "Only mixed-signal groups produce non-zero workflow-RLOO advantages. "
            "This report measures signal density; it does not by itself prove that filtering improves training.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-n", type=int, default=4)
    args = parser.parse_args()

    rows = parse_step_metrics(args.log.read_text(encoding="utf-8", errors="replace"))
    if not rows:
        raise ValueError(f"no DAGENT training-step metrics found in {args.log}")
    report = analyze_rows(rows, args.rollout_n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_step"}, indent=2))


if __name__ == "__main__":
    main()
