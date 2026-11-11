#!/usr/bin/env python3
"""Replay a gold-blind conservative Product selection rule on frozen traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qa_manager.ROSA.shoppingbench_protocol import parse_product_set
from qa_manager.ROSA.shoppingbench_product_selection_fallback import (
    apply_conservative_product_selection,
)


def audit(attribution_path: Path, results_path: Path) -> dict[str, Any]:
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    visible_queries = {row["sample_id"]: row["visible_query"] for row in attribution["rows"]}
    results = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for result in results:
        sample_id = result["sample_id"]
        trace = result.get("raw_trace") or {}
        context = "\n".join(
            str(node.get("search_context") or "")
            for node in trace.get("nodes_execution") or []
            if str(node.get("type") or node.get("node_type") or "").lower() == "search"
        )
        products = parse_product_set(context)
        baseline_ids = [str(value) for value in result.get("predicted_product_ids") or []]
        annotated = [
            {**product, "model_selected": str(product["product_id"]) in baseline_ids}
            for product in products
        ]
        _, decision = apply_conservative_product_selection(
            visible_queries[sample_id], annotated
        )
        changed = bool(decision["applied"])
        candidate_ids = [decision["selected_id"]] if decision["selected_id"] else []

        # Gold is intentionally consulted only after the candidate decision is frozen.
        gold_ids = [str(value) for value in result.get("expected_product_ids") or []]
        baseline_exact = baseline_ids == gold_ids
        candidate_exact = candidate_ids == gold_ids
        rows.append(
            {
                "sample_id": sample_id,
                "visible_query": visible_queries[sample_id],
                "constraint_field": decision["constraint_field"],
                "constraint_value": decision["constraint_value"],
                "candidate_statuses": decision["candidate_statuses"],
                "satisfying_ids_without_gold": decision["satisfying_ids"],
                "baseline_ids": baseline_ids,
                "baseline_status": decision["baseline_status"],
                "candidate_ids_without_gold": candidate_ids,
                "changed": changed,
                "decision_reason": "unique_satisfier_replaces_non_satisfier" if changed else "preserve",
                "gold_ids_posthoc": gold_ids,
                "baseline_exact_posthoc": baseline_exact,
                "candidate_exact_posthoc": candidate_exact,
                "posthoc_outcome": (
                    "improved"
                    if candidate_exact and not baseline_exact
                    else "regressed"
                    if baseline_exact and not candidate_exact
                    else "unchanged"
                ),
            }
        )

    improved = sum(row["posthoc_outcome"] == "improved" for row in rows)
    regressed = sum(row["posthoc_outcome"] == "regressed" for row in rows)
    return {
        "schema": "shoppingbench-product-conservative-selection-audit-v1",
        "evidence_level": "zero-model frozen-trace counterfactual",
        "examples": len(rows),
        "changed_rows": sum(row["changed"] for row in rows),
        "baseline_exact": sum(row["baseline_exact_posthoc"] for row in rows),
        "candidate_exact": sum(row["candidate_exact_posthoc"] for row in rows),
        "improved_rows": improved,
        "regressed_rows": regressed,
        "unchanged_rows": len(rows) - improved - regressed,
        "gold_used_by_decision": False,
        "gold_used_posthoc": True,
        "model_calls": 0,
        "external_api_calls": 0,
        "runtime_changes": 0,
        "scientific_improvement_claim": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    report = audit(args.attribution, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    keys = (
        "examples",
        "changed_rows",
        "baseline_exact",
        "candidate_exact",
        "improved_rows",
        "regressed_rows",
    )
    print(json.dumps({key: report[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
