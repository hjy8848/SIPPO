#!/usr/bin/env python3
"""Slice paired voucher compiler runs by retrieval and selection failure."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.audit_shoppingbench_compiled_replay import aggregate, audit_row
from scripts.compare_shoppingbench_voucher_compiler_runs import load_rows


def retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions = sum(len(row["expected_product_ids"]) for row in rows)
    corresponding = sum(sum(row["gold_in_corresponding_node"]) for row in rows)
    any_node = sum(sum(row["gold_in_any_node"]) for row in rows)
    return {
        "rows": len(rows),
        "expected_positions": positions,
        "corresponding_hits": corresponding,
        "corresponding_recall": corresponding / positions if positions else 0.0,
        "any_node_hits": any_node,
        "any_node_recall": any_node / positions if positions else 0.0,
        "failure_categories": dict(sorted(Counter(row["failure_category"] for row in rows).items())),
    }


def analyze(
    baseline_path: Path,
    candidate_path: Path,
    output_dir: Path,
    expected_rows: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    baseline = load_rows(baseline_path)
    candidate = load_rows(candidate_path)
    if len(candidate) != expected_rows:
        raise ValueError(f"expected {expected_rows} candidate rows, found {len(candidate)}")
    selected = sorted(candidate)
    if any(sample_id not in baseline for sample_id in selected):
        raise ValueError("baseline does not cover every candidate sample")

    old_audits: list[dict[str, Any]] = []
    new_audits: list[dict[str, Any]] = []
    affected_old: list[dict[str, Any]] = []
    affected_new: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    pair_rows = []
    for sample_id in selected:
        old = baseline[sample_id]
        new = candidate[sample_id]
        old_trace = old.get("raw_trace") or {}
        new_trace = new.get("raw_trace") or {}
        if old.get("query") != new.get("query"):
            raise ValueError(f"{sample_id}: query mismatch")
        if old.get("expected_product_ids") != new.get("expected_product_ids"):
            raise ValueError(f"{sample_id}: expected ID mismatch")
        if old_trace.get("raw_plan_output") != new_trace.get("raw_plan_output"):
            raise ValueError(f"{sample_id}: raw Planner mismatch")
        old_audit = audit_row(old)
        new_audit = audit_row(new)
        old_audits.append(old_audit)
        new_audits.append(new_audit)
        affected = old_trace.get("plan_output") != new_trace.get("plan_output")
        if affected:
            affected_old.append(old_audit)
            affected_new.append(new_audit)
        transition = f"{old_audit['failure_category']}->{new_audit['failure_category']}"
        transitions[transition] += 1
        pair_rows.append(
            {
                "sample_id": sample_id,
                "compiler_affected": affected,
                "transition": transition,
                "baseline_corresponding_hits": sum(old_audit["gold_in_corresponding_node"]),
                "candidate_corresponding_hits": sum(new_audit["gold_in_corresponding_node"]),
                "baseline_any_node_hits": sum(old_audit["gold_in_any_node"]),
                "candidate_any_node_hits": sum(new_audit["gold_in_any_node"]),
                "expected_positions": len(new_audit["expected_product_ids"]),
                "baseline_queries": old_audit["queries"],
                "candidate_queries": new_audit["queries"],
            }
        )

    old_summary = retrieval_summary(old_audits)
    new_summary = retrieval_summary(new_audits)
    affected_old_summary = retrieval_summary(affected_old)
    affected_new_summary = retrieval_summary(affected_new)
    report = {
        "protocol": "post-hoc paired voucher compiler retrieval/selection attribution",
        "rows": expected_rows,
        "baseline": old_summary,
        "candidate": new_summary,
        "delta": {
            "corresponding_recall": (
                new_summary["corresponding_recall"] - old_summary["corresponding_recall"]
            ),
            "any_node_recall": new_summary["any_node_recall"] - old_summary["any_node_recall"],
        },
        "compiler_affected": {
            "baseline": affected_old_summary,
            "candidate": affected_new_summary,
            "delta_corresponding_recall": (
                affected_new_summary["corresponding_recall"]
                - affected_old_summary["corresponding_recall"]
            ),
            "delta_any_node_recall": (
                affected_new_summary["any_node_recall"]
                - affected_old_summary["any_node_recall"]
            ),
        },
        "category_transitions": dict(sorted(transitions.items())),
        "candidate_remaining_bottleneck": max(
            new_summary["failure_categories"],
            key=new_summary["failure_categories"].get,
        ),
        "gold_used_by_runtime": False,
        "gold_used_for_posthoc_audit": True,
        "training_authorized": False,
        "pairs": pair_rows,
        "audit_schema": aggregate(new_audits)["category_definitions"],
    }
    output_dir.mkdir(parents=True)
    (output_dir / "failure_slice.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(args.baseline, args.candidate, args.output_dir, args.expected_rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
