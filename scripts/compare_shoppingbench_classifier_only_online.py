#!/usr/bin/env python3
"""Audit an online ShoppingBench classifier-only replay against frozen r5 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = ("product", "shop", "voucher")


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row["sample_id"]): row for row in rows}


def failed(row: dict[str, Any]) -> bool:
    return any(row.get(key) for key in (
        "format_error", "contract_error", "pipeline_error", "request_error"
    ))


def query_rewrites(row: dict[str, Any]) -> int:
    return sum(
        event.get("agent_name") == "QueryRewriteAgent"
        for event in (row.get("raw_trace") or {}).get("sft_events") or []
        if isinstance(event, dict)
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "failures": sum(failed(row) for row in rows),
        "mean_exact_id_fraction": sum(
            float((row.get("score") or {}).get("exact_id_fraction") or 0.0)
            for row in rows
        ) / len(rows),
        "exact_sequences": sum(
            bool((row.get("score") or {}).get("exact_sequence_success")) for row in rows
        ),
        "first_id_exact": sum(
            bool((row.get("score") or {}).get("first_id_exact")) for row in rows
        ),
        "retrieval_calls": sum(int(row.get("retrieval_calls") or 0) for row in rows),
    }


def compare(baseline_root: Path, candidate_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    paired: list[dict[str, Any]] = []
    for task in TASKS:
        baseline = read_rows(baseline_root / task / "results.jsonl")
        candidate = read_rows(candidate_root / task / "results.jsonl")
        if set(baseline) != set(candidate) or len(baseline) != 16:
            raise ValueError(f"{task}: paired sample IDs are not exactly 16 and identical")
        for sample_id in sorted(baseline):
            old, new = baseline[sample_id], candidate[sample_id]
            old_trace, new_trace = old.get("raw_trace") or {}, new.get("raw_trace") or {}
            compiler = (new_trace.get("contract_protocol") or {}).get("plan_compiler") or {}
            paired.append({
                "sample_id": sample_id,
                "task": task,
                "raw_planner_identity": (
                    old_trace.get("raw_plan_output") == new_trace.get("raw_plan_output")
                ),
                "planner_output_override_used": (
                    new_trace.get("planner_output_override_used") is True
                ),
                "plan_override_unused": new_trace.get("plan_override_used") is False,
                "request_kind_correct": compiler.get("request_kind") == task,
                "query_rewrite_events": query_rewrites(new),
                "baseline_failure": failed(old),
                "candidate_failure": failed(new),
                "baseline_fraction": float(
                    (old.get("score") or {}).get("exact_id_fraction") or 0.0
                ),
                "candidate_fraction": float(
                    (new.get("score") or {}).get("exact_id_fraction") or 0.0
                ),
            })

    baseline_rows = [row for task in TASKS for row in read_rows(
        baseline_root / task / "results.jsonl"
    ).values()]
    candidate_rows = [row for task in TASKS for row in read_rows(
        candidate_root / task / "results.jsonl"
    ).values()]
    baseline_summary, candidate_summary = summarize(baseline_rows), summarize(candidate_rows)
    identity_pass = all(
        row["raw_planner_identity"]
        and row["planner_output_override_used"]
        and row["plan_override_unused"]
        for row in paired
    )
    routing_pass = all(row["request_kind_correct"] for row in paired)
    no_rewrite = sum(row["query_rewrite_events"] for row in paired) == 0
    outcome_pass = (
        candidate_summary["failures"] <= baseline_summary["failures"]
        and candidate_summary["mean_exact_id_fraction"]
        >= baseline_summary["mean_exact_id_fraction"]
    )
    report = {
        "schema": "shoppingbench-classifier-only-online-paired-v1",
        "evidence_level": "online classifier-only causal check",
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "identity_pass": identity_pass,
        "routing_pass": routing_pass,
        "query_rewrite_zero": no_rewrite,
        "outcome_nonregression_pass": outcome_pass,
        "passed": identity_pass and routing_pass and no_rewrite and outcome_pass,
        "training_authorized": False,
        "paired": paired,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline_root, args.candidate_root, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "baseline", "candidate", "identity_pass", "routing_pass",
        "query_rewrite_zero", "outcome_nonregression_pass", "passed"
    )}, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
