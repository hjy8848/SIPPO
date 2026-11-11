#!/usr/bin/env python3
"""Audit a Product-only invalid-query fallback on frozen ShoppingBench traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qa_manager.ROSA.shoppingbench_product_query_fallback import select_product_query
from qa_manager.ROSA.shoppingbench_protocol import extract_atomic_search_queries


def audit(input_path: Path) -> dict[str, Any]:
    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trace = row.get("raw_trace") or {}
        model_queries = extract_atomic_search_queries(trace.get("raw_plan_output", ""))
        selected, metadata = select_product_query(row.get("query", ""), model_queries)
        rows.append({
            "sample_id": row.get("sample_id"),
            "model_queries": model_queries,
            "selected_query": selected,
            "fallback": metadata,
        })

    return {
        "schema": "shoppingbench-product-query-fallback-audit-v1",
        "evidence_level": "offline frozen-trace compiler audit",
        "examples": len(rows),
        "fallback_activations": sum(row["fallback"]["applied"] for row in rows),
        "preserved_model_queries": sum(not row["fallback"]["applied"] for row in rows),
        "gold_used": False,
        "model_calls": 0,
        "external_api_calls": 0,
        "rows": rows,
        "scientific_improvement_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    report = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: report[key]
        for key in ("examples", "fallback_activations", "preserved_model_queries")
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
