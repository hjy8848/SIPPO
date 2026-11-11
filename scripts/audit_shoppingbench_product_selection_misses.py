#!/usr/bin/env python3
"""Audit whether Product selection misses are identifiable from visible attributes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_manager.ROSA.shoppingbench_protocol import parse_product_set
from qa_manager.ROSA.shoppingbench_product_selection_fallback import (
    product_constraint_status,
    visible_product_constraint,
)


def audit(attribution_path: Path, results_path: Path) -> dict[str, Any]:
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    results = {
        row["sample_id"]: row
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    rows = []
    for source in attribution["rows"]:
        if source.get("category") != "selection_miss":
            continue
        result = results[source["sample_id"]]
        trace = result.get("raw_trace") or {}
        search_nodes = [
            node
            for node in trace.get("nodes_execution") or []
            if str(node.get("type") or node.get("node_type") or "").lower() == "search"
        ]
        context = "\n".join(str(node.get("search_context") or "") for node in search_nodes)
        products = parse_product_set(context)
        field, value = visible_product_constraint(source.get("visible_query", ""), products)
        statuses = {
            str(product["product_id"]): product_constraint_status(product, field, value)
            for product in products
        }
        satisfying = [product_id for product_id, status in statuses.items() if status == "satisfies"]
        selected_id = str((source.get("predicted_ids") or [""])[0])
        gold_id = str((source.get("gold_ids") or [""])[0])
        selected_status = statuses.get(selected_id, "missing")
        if selected_status != "satisfies" and len(satisfying) == 1:
            diagnosis = "unique_structured_match_missed"
        elif selected_status == "contradicts" and len(satisfying) > 1:
            diagnosis = "selected_conflict_but_multiple_satisfiers"
        elif selected_status == "satisfies" and len(satisfying) > 1:
            diagnosis = "multiple_visible_satisfiers"
        else:
            diagnosis = "insufficient_structured_evidence"
        model_response = ""
        for event in trace.get("sft_events") or []:
            if event.get("agent_name") == "AnswerGenerationAgent":
                model_response = str(event.get("response") or "")
        rows.append(
            {
                "sample_id": source["sample_id"],
                "visible_query": source.get("visible_query"),
                "constraint_field": field,
                "constraint_value": value,
                "retrieved_ids": [str(product["product_id"]) for product in products],
                "candidate_statuses": statuses,
                "satisfying_ids_without_gold": satisfying,
                "selected_id": selected_id,
                "selected_status": selected_status,
                "gold_id_posthoc": gold_id,
                "gold_status_posthoc": statuses.get(gold_id, "missing"),
                "diagnosis": diagnosis,
                "answer_generation_response": model_response,
                "final_answer": trace.get("final_answer"),
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["diagnosis"]] = counts.get(row["diagnosis"], 0) + 1
    return {
        "schema": "shoppingbench-product-selection-miss-audit-v1",
        "evidence_level": "zero-model post-hoc structured-attribute audit",
        "examples": len(rows),
        "diagnosis_counts": counts,
        "gold_used_to_classify_candidates": False,
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
    print(json.dumps({key: report[key] for key in ("examples", "diagnosis_counts")}, indent=2))


if __name__ == "__main__":
    main()
