#!/usr/bin/env python3
"""Replay frozen ShoppingBench plans after changing only request classification."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qa_manager.ROSA.contract_protocol import parse_contract_plan
from qa_manager.ROSA.shoppingbench_protocol import (
    classify_shopping_request,
    compile_shoppingbench_plan,
    extract_atomic_search_queries,
)


TASKS = ("product", "shop", "voucher")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replay(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")

    samples: list[dict[str, Any]] = []
    for task in TASKS:
        path = run_dir / "grounding_pilot" / "contract" / task / "results.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 16:
            raise ValueError(f"{task}: expected 16 frozen rows, found {len(rows)}")
        for row in rows:
            trace = row.get("raw_trace") or {}
            old_compiler = (trace.get("contract_protocol") or {}).get("plan_compiler") or {}
            raw_plan = str(trace.get("raw_plan_output") or "")
            new_kind = classify_shopping_request(str(row.get("query") or ""))
            extracted_queries = extract_atomic_search_queries(raw_plan)
            compile_error = ""
            compiled_searches = 0
            compiled_reason_op = ""
            try:
                compiled, metadata = compile_shoppingbench_plan(
                    str(row.get("query") or ""), raw_plan, max_nodes=5
                )
                parsed = parse_contract_plan(compiled)
                compiled_searches = sum(
                    node.get("kind") == "search" for node in parsed.get("nodes") or []
                )
                reason_nodes = [
                    node for node in parsed.get("nodes") or [] if node.get("kind") == "reason"
                ]
                compiled_reason_op = str(reason_nodes[-1].get("op") or "") if reason_nodes else ""
                if metadata.get("request_kind") != new_kind:
                    raise ValueError("compiler/classifier request-kind disagreement")
            except ValueError as exc:
                compile_error = str(exc)

            old_kind = str(old_compiler.get("request_kind") or "")
            samples.append(
                {
                    "sample_id": row["sample_id"],
                    "task": task,
                    "old_kind": old_kind,
                    "new_kind": new_kind,
                    "old_kind_correct": old_kind == task,
                    "new_kind_correct": new_kind == task,
                    "route_corrected": old_kind != task and new_kind == task,
                    "route_regressed": old_kind == task and new_kind != task,
                    "extracted_query_count": len(extracted_queries),
                    "compile_succeeded": not compile_error,
                    "compile_error": compile_error,
                    "compiled_search_count": compiled_searches,
                    "compiled_reason_op": compiled_reason_op,
                }
            )

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(group),
            "old_kind_correct": sum(row["old_kind_correct"] for row in group),
            "new_kind_correct": sum(row["new_kind_correct"] for row in group),
            "route_corrected": sum(row["route_corrected"] for row in group),
            "route_regressed": sum(row["route_regressed"] for row in group),
            "compile_succeeded": sum(row["compile_succeeded"] for row in group),
            "atomic_query_count_sufficient": sum(
                row["extracted_query_count"] >= (2 if row["task"] == "shop" else 1)
                for row in group
            ),
        }

    report = {
        "schema": "shoppingbench-classifier-only-frozen-replay-v1",
        "evidence_level": "zero-model-call classifier-only replay",
        "changed_variable": "deterministic request classifier",
        "frozen_variables": [
            "question",
            "raw_planner_output",
            "query_extractor",
            "plan_compiler",
            "node_limit",
        ],
        "overall": summarize(samples),
        "by_task": {
            task: summarize([row for row in samples if row["task"] == task])
            for task in TASKS
        },
        "compile_errors": dict(sorted(Counter(
            row["compile_error"] for row in samples if row["compile_error"]
        ).items())),
        "gold_used_by_runtime": False,
        "retrieval_executed": False,
        "model_calls": 0,
        "training_authorized": False,
        "samples": samples,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.run_dir, args.output_dir)
    print(json.dumps({key: report[key] for key in ("overall", "by_task", "compile_errors")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
