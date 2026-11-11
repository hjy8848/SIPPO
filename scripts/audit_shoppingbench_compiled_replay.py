"""Audit gold retrieval coverage after ShoppingBench system-plan compilation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PRODUCT_ID_RE = re.compile(r"Product ID:\s*(\d+)", re.I)
CATEGORIES = (
    "execution_failure",
    "success",
    "retrieval_miss",
    "cross_node_mismatch",
    "selection_miss",
)


def execution_errors(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("raw_trace") or {}
    candidates = {
        "request_error": row.get("request_error"),
        "pipeline_error": row.get("pipeline_error"),
        "format_error": row.get("format_error"),
        "contract_error": row.get("contract_error"),
        "trace_error": trace.get("error"),
        "trace_execution_error": trace.get("execution_error"),
    }
    return {key: value for key, value in candidates.items() if value not in (None, "", False)}


def classify_row(
    *,
    expected: list[str],
    predicted: list[str],
    corresponding_hits: list[bool],
    any_node_hits: list[bool],
    errors: dict[str, Any],
) -> str:
    if errors:
        return "execution_failure"
    if predicted == expected:
        return "success"
    if not all(any_node_hits):
        return "retrieval_miss"
    if not all(corresponding_hits):
        return "cross_node_mismatch"
    return "selection_miss"


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = [str(value) for value in row.get("expected_product_ids") or []]
    predicted = [str(value) for value in row.get("predicted_product_ids") or []]
    search_nodes = [
        node
        for node in ((row.get("raw_trace") or {}).get("nodes_execution") or [])
        if node.get("type") == "search"
    ]
    retrieved_by_node = [
        list(dict.fromkeys(PRODUCT_ID_RE.findall(str(node.get("search_context") or ""))))
        for node in search_nodes
    ]
    corresponding_hits = [
        index < len(retrieved_by_node) and product_id in retrieved_by_node[index]
        for index, product_id in enumerate(expected)
    ]
    any_node_hits = [
        any(product_id in retrieved for retrieved in retrieved_by_node)
        for product_id in expected
    ]
    errors = execution_errors(row)
    category = classify_row(
        expected=expected,
        predicted=predicted,
        corresponding_hits=corresponding_hits,
        any_node_hits=any_node_hits,
        errors=errors,
    )
    return {
        "sample_id": row.get("sample_id"),
        "task": str(row.get("sample_id") or "").split("-", 1)[0],
        "expected_product_ids": expected,
        "predicted_product_ids": predicted,
        "queries": [node.get("executed_query") for node in search_nodes],
        "retrieved_product_ids_by_node": retrieved_by_node,
        "gold_in_corresponding_node": corresponding_hits,
        "gold_in_any_node": any_node_hits,
        "execution_errors": errors,
        "failure_category": category,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        positions = sum(len(row["expected_product_ids"]) for row in group)
        corresponding = sum(sum(row["gold_in_corresponding_node"]) for row in group)
        any_node = sum(sum(row["gold_in_any_node"]) for row in group)
        category_counts = {
            category: sum(row["failure_category"] == category for row in group)
            for category in CATEGORIES
        }
        return {
            "examples": len(group),
            "expected_positions": positions,
            "corresponding_node_hits": corresponding,
            "corresponding_node_recall": corresponding / positions if positions else 0.0,
            "any_node_hits": any_node,
            "any_node_recall": any_node / positions if positions else 0.0,
            "category_counts": category_counts,
            "category_rates": {
                category: count / len(group) if group else 0.0
                for category, count in category_counts.items()
            },
        }

    return {
        "overall": summarize(rows),
        "by_task": {task: summarize(group) for task, group in sorted(by_task.items())},
        "rows": rows,
        "gold_used_by_runtime": False,
        "gold_used_for_label_side_audit": True,
        "classification_precedence": list(CATEGORIES),
        "category_definitions": {
            "execution_failure": "Request, pipeline, format, contract, or trace execution failed.",
            "success": "Predicted product ID sequence exactly matches the expected sequence.",
            "retrieval_miss": "At least one expected product ID is absent from every Search result.",
            "cross_node_mismatch": "All expected IDs were retrieved, but at least one only appeared under a different Search node.",
            "selection_miss": "All expected IDs appeared under their corresponding Search nodes, but final ID selection was not exact.",
        },
        "evidence_level": "post-hoc mutually exclusive failure attribution audit",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite: {args.output}")
    paths = sorted(args.input_dir.glob("*/results.jsonl"))
    if not paths:
        raise SystemExit(f"No results found under {args.input_dir}")
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(audit_row(json.loads(line)) for line in handle if line.strip())
    report = aggregate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
