#!/usr/bin/env python3
"""Analyze role-swapped DAGENT runs on the frozen 2Wiki formal subset."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.analyze_dagent_final_supplement import (
        load_2wiki_types,
        paired_type_analysis,
    )
    from scripts.compare_dagent_formal_runs import (
        FormalRun,
        load_run,
        parse_assignment,
        validate_alignment,
    )
except ModuleNotFoundError:
    from analyze_dagent_final_supplement import load_2wiki_types, paired_type_analysis
    from compare_dagent_formal_runs import (
        FormalRun,
        load_run,
        parse_assignment,
        validate_alignment,
    )


DATASETS = {"2wikimultihopqa": 500}


def summarize_run(run: FormalRun) -> dict[str, Any]:
    rows = list(run.rows["2wikimultihopqa"].values())
    specialist_generated = 0
    workflow_failures = 0
    for row in rows:
        trace = row.get("raw_trace") or {}
        specialist_generated += bool(
            (trace.get("planner_routing") or {}).get("specialist_generated")
        )
        workflow_failures += bool(str(trace.get("execution_error") or "").strip())
    return {
        "samples": len(rows),
        "em": statistics.fmean(float(row["em"]) for row in rows),
        "f1": statistics.fmean(float(row["f1"]) for row in rows),
        "avg_search_steps": statistics.fmean(
            float(row.get("search_steps") or 0.0) for row in rows
        ),
        "avg_retrieval_calls": statistics.fmean(
            float(row.get("retrieval_calls") or 0.0) for row in rows
        ),
        "workflow_failures": workflow_failures,
        "specialist_generated_rate": specialist_generated / len(rows),
    }


def analyze_runs(
    runs: Sequence[FormalRun],
    question_types: Mapping[str, str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    validate_alignment(runs, DATASETS)
    comparisons = []
    for pair_index, (reference, candidate) in enumerate(itertools.combinations(runs, 2)):
        comparisons.append(
            {
                "reference": reference.label,
                "candidate": candidate.label,
                "types": paired_type_analysis(
                    reference.rows["2wikimultihopqa"],
                    candidate.rows["2wikimultihopqa"],
                    question_types,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + pair_index * 1009,
                ),
            }
        )
    return {
        "protocol": "DAGENT step468 Planner Swap on frozen 2Wiki-500",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "runs": {run.label: summarize_run(run) for run in runs},
        "comparisons": comparisons,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# DAGENT Planner Swap 2Wiki Analysis",
        "",
        "## Runs",
        "",
        "| Run | n | EM | F1 | Search | Retrieval | Failures | Specialist rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in report["runs"].items():
        lines.append(
            f"| {label} | {row['samples']} | {row['em']:.6f} | {row['f1']:.6f} | "
            f"{row['avg_search_steps']:.6f} | {row['avg_retrieval_calls']:.6f} | "
            f"{row['workflow_failures']} | {row['specialist_generated_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Official Question Types",
            "",
            "| Reference | Candidate | Type | n | Delta EM | 95% CI | McNemar p | Bonferroni p | Delta F1 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in report["comparisons"]:
        for question_type, row in comparison["types"].items():
            em = row["em"]
            f1 = row["f1"]
            mc = row["mcnemar"]
            lines.append(
                f"| {comparison['reference']} | {comparison['candidate']} | {question_type} | "
                f"{row['samples']} | {em['delta']:+.6f} | "
                f"[{em['bootstrap_95ci'][0]:+.6f}, {em['bootstrap_95ci'][1]:+.6f}] | "
                f"{mc['pvalue']:.6g} | {mc['bonferroni_pvalue']:.6g} | {f1['delta']:+.6f} |"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="2Wiki formal run root (repeat for each role configuration)",
    )
    parser.add_argument("--two-wiki-raw", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = [parse_assignment(value, option="--run") for value in args.run]
    runs = [load_run(label, Path(path), DATASETS) for label, path in specs]
    report = analyze_runs(
        runs,
        load_2wiki_types(args.two_wiki_raw),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "planner_swap_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "planner_swap_analysis.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] Planner Swap analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
