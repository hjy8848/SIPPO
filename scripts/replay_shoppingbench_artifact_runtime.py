"""Replay frozen ShoppingBench traces through the artifact-only CP runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from qa_manager.ROSA.shoppingbench_protocol import (
    annotate_product_selection,
    execute_artifact_reason,
    extract_product_ids,
    parse_product_set,
    select_supported_products,
    serialize_product_sink,
    validate_final_answer,
)
from scripts.run_shoppingbench_predictions import score_ids


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*/results.jsonl"))
    if not paths:
        raise ValueError(f"No task results found under {input_dir}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def replay_row(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("raw_trace") or {}
    if trace.get("protocol_profile") != "shoppingbench":
        raise ValueError(f"{row.get('sample_id')}: trace is not ShoppingBench")
    if not trace.get("plan_override_used"):
        raise ValueError(f"{row.get('sample_id')}: trace did not use a frozen plan")

    artifacts: dict[str, list[dict[str, Any]]] = {}
    methods: list[str] = []
    retrieved_count = 0
    selected_count = 0
    for node in trace.get("nodes_execution") or []:
        node_id = str(node.get("id", ""))
        if node.get("type") == "search":
            retrieved = parse_product_set(
                node.get("search_context", ""),
                query=node.get("executed_query", ""),
                source_node=node_id,
            )
            selected = select_supported_products(retrieved, node.get("conclusion", ""))
            artifacts[node_id] = annotate_product_selection(
                retrieved, node.get("conclusion", "")
            )
            retrieved_count += len(retrieved)
            selected_count += len(selected)
        elif node.get("type") == "reason":
            contract_node = node.get("contract_node") or {}
            source_sets = [
                artifacts.get(str(parent_id), [])
                for parent_id in contract_node.get("input_ids") or []
            ]
            artifacts[node_id], method = execute_artifact_reason(
                contract_node,
                source_sets,
                row.get("query", ""),
            )
            methods.append(method)

    sink_id = str(trace.get("answer_sink_id") or "")
    contract_nodes = ((trace.get("contract_protocol") or {}).get("plan_nodes") or [])
    sink_node = next(
        (node for node in contract_nodes if str(node.get("id", "")) == sink_id),
        {},
    )
    replayed_answer = serialize_product_sink(
        artifacts.get(sink_id, []),
        sink_node=sink_node,
    )
    predicted_ids = extract_product_ids(replayed_answer)
    expected_ids = [str(value) for value in row.get("expected_product_ids") or []]
    return {
        "sample_id": row.get("sample_id"),
        "old_answer": trace.get("final_answer"),
        "old_format_valid": row.get("format_error") is None,
        "old_score": row.get("score") or score_ids([], expected_ids),
        "replayed_answer": replayed_answer,
        "replayed_product_ids": predicted_ids,
        "replayed_format_valid": validate_final_answer(replayed_answer) is None,
        "replayed_score": score_ids(predicted_ids, expected_ids),
        "retrieved_product_count": retrieved_count,
        "model_selected_product_count": selected_count,
        "artifact_reason_methods": methods,
        "model_calls_replayed": 0,
    }


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    count = len(rows)
    return {
        "examples": count,
        "old_format_valid": sum(row["old_format_valid"] for row in rows),
        "replayed_format_valid": sum(row["replayed_format_valid"] for row in rows),
        "old_mean_exact_id_fraction": (
            sum(row["old_score"]["exact_id_fraction"] for row in rows) / count
        ),
        "replayed_mean_exact_id_fraction": (
            sum(row["replayed_score"]["exact_id_fraction"] for row in rows) / count
        ),
        "old_exact_sequence_successes": sum(
            row["old_score"]["exact_sequence_success"] for row in rows
        ),
        "replayed_exact_sequence_successes": sum(
            row["replayed_score"]["exact_sequence_success"] for row in rows
        ),
        "old_first_id_exact_successes": sum(
            row["old_score"]["first_id_exact"] for row in rows
        ),
        "replayed_first_id_exact_successes": sum(
            row["replayed_score"]["first_id_exact"] for row in rows
        ),
        "runtime_changed_only": True,
        "gold_used_by_runtime": False,
        "model_calls_replayed": 0,
        "evidence_level": "offline frozen-trace artifact-runtime replay",
        "scientific_improvement_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse output directory: {output_dir}")
    rows = [replay_row(row) for row in load_rows(input_dir)]
    report = aggregate(rows)
    output_dir.mkdir(parents=True)
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
