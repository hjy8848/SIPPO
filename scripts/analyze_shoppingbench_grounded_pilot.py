#!/usr/bin/env python3
"""Slice a grounded ShoppingBench pilot by routing, topology, and retrieval."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qa_manager.ROSA.shoppingbench_protocol import classify_shopping_request


TASKS = ("product", "shop", "voucher")
PRODUCT_ID_RE = re.compile(r"\bProduct ID:\s*(\d+)\b", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_cardinalities(runtime_root: Path) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for task in TASKS:
        path = runtime_root / "source" / "data" / f"synthesize_{task}_test.jsonl"
        for position, row in enumerate(read_jsonl(path)):
            reward = row.get("reward")
            products = [reward] if isinstance(reward, dict) else list(reward or [])
            result[(task, position)] = len(products)
    return result


def compiler_metadata(trace: dict[str, Any]) -> dict[str, Any]:
    protocol = trace.get("contract_protocol") or {}
    value = protocol.get("plan_compiler") or {}
    return value if isinstance(value, dict) else {}


def corresponding_hits(row: dict[str, Any]) -> int:
    expected = [str(value) for value in row.get("expected_product_ids") or []]
    searches = [
        node
        for node in (row.get("raw_trace") or {}).get("nodes_execution") or []
        if node.get("type") == "search"
    ]
    return sum(
        product_id in set(PRODUCT_ID_RE.findall(str(node.get("search_context") or "")))
        for product_id, node in zip(expected, searches)
    )


def analyze(run_dir: Path, runtime_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    cardinalities = source_cardinalities(runtime_root)
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        results = read_jsonl(run_dir / "grounding_pilot" / "contract" / task / "results.jsonl")
        if len(results) != 16:
            raise ValueError(f"{task}: expected 16 rows, found {len(results)}")
        for row in results:
            position = int(row["position"])
            expected = [str(value) for value in row.get("expected_product_ids") or []]
            trace = row.get("raw_trace") or {}
            compiler = compiler_metadata(trace)
            searches = [
                node for node in trace.get("nodes_execution") or [] if node.get("type") == "search"
            ]
            query_rewrites = sum(
                event.get("agent_name") == "QueryRewriteAgent"
                for event in trace.get("sft_events") or []
                if isinstance(event, dict)
            )
            classified = classify_shopping_request(str(row.get("query") or ""))
            compiler_kind = str(compiler.get("request_kind") or "")
            expected_count = cardinalities[(task, position)]
            if expected_count != len(expected):
                raise ValueError(f"{row.get('sample_id')}: source/result cardinality mismatch")
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "task": task,
                    "cardinality": expected_count,
                    "classifier_kind": classified,
                    "compiler_kind": compiler_kind,
                    "compiler_applied": compiler.get("applied") is True,
                    "compiler_query_count": len(compiler.get("queries") or []),
                    "executed_search_count": len(searches),
                    "search_count_sufficient": len(searches) == expected_count,
                    "corresponding_hits": corresponding_hits(row),
                    "corresponding_recall": corresponding_hits(row) / expected_count,
                    "exact_sequence": bool((row.get("score") or {}).get("exact_sequence_success")),
                    "request_kind_correct": compiler_kind == task,
                    "classifier_kind_correct": classified == task,
                    "query_rewrite_events": query_rewrites,
                    "execution_failure": any(
                        row.get(key)
                        for key in ("format_error", "contract_error", "pipeline_error", "request_error")
                    ),
                }
            )

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        expected_products = sum(row["cardinality"] for row in group)
        return {
            "rows": len(group),
            "expected_products": expected_products,
            "classifier_correct": sum(row["classifier_kind_correct"] for row in group),
            "request_kind_correct": sum(row["request_kind_correct"] for row in group),
            "compiler_applied": sum(row["compiler_applied"] for row in group),
            "search_count_sufficient": sum(row["search_count_sufficient"] for row in group),
            "corresponding_recall": (
                sum(row["corresponding_hits"] for row in group) / expected_products
                if expected_products else 0.0
            ),
            "exact_sequences": sum(row["exact_sequence"] for row in group),
            "execution_failures": sum(row["execution_failure"] for row in group),
            "query_rewrite_events": sum(row["query_rewrite_events"] for row in group),
        }

    by_task = {task: summarize([row for row in rows if row["task"] == task]) for task in TASKS}
    by_task_cardinality = {
        task: {
            str(cardinality): summarize(
                [row for row in rows if row["task"] == task and row["cardinality"] == cardinality]
            )
            for cardinality in sorted({row["cardinality"] for row in rows if row["task"] == task})
        }
        for task in TASKS
    }
    defect_overlap = Counter()
    for row in rows:
        route_failure = not row["request_kind_correct"]
        topology_failure = not row["search_count_sufficient"]
        defect_overlap[
            f"route_{'fail' if route_failure else 'pass'}__topology_{'fail' if topology_failure else 'pass'}"
        ] += 1
    classifier_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        classifier_confusion[row["task"]][row["classifier_kind"]] += 1
    report = {
        "schema": "shoppingbench-grounded-pilot-posthoc-v1",
        "evidence_level": "zero-model-call posthoc audit",
        "run_dir": str(run_dir),
        "runtime_root": str(runtime_root),
        "rows": len(rows),
        "overall": summarize(rows),
        "by_task": by_task,
        "by_task_cardinality": by_task_cardinality,
        "classifier_confusion": {
            task: dict(sorted(counts.items())) for task, counts in classifier_confusion.items()
        },
        "defect_overlap": dict(sorted(defect_overlap.items())),
        "gold_used_by_runtime": False,
        "gold_used_for_posthoc_scoring": True,
        "training_authorized": False,
        "samples": rows,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(args.run_dir, args.runtime_root, args.output_dir)
    print(json.dumps({key: report[key] for key in ("overall", "by_task", "classifier_confusion", "defect_overlap")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
