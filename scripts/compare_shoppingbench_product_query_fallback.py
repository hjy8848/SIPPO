#!/usr/bin/env python3
"""Compare Product-only invalid-query fallback against frozen online rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qa_manager.ROSA.shoppingbench_product_query_fallback import select_product_query
from qa_manager.ROSA.shoppingbench_protocol import (
    compile_shoppingbench_plan,
    extract_atomic_search_queries,
)


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def failed(row: dict[str, Any]) -> bool:
    return any(
        row.get(key)
        for key in ("format_error", "contract_error", "pipeline_error", "request_error")
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "failures": sum(failed(row) for row in rows),
        "exact_sequences": sum(
            bool((row.get("score") or {}).get("exact_sequence_success")) for row in rows
        ),
        "mean_exact_id_fraction": sum(
            float((row.get("score") or {}).get("exact_id_fraction") or 0.0)
            for row in rows
        ) / len(rows),
    }


def runtime_question(question: str) -> str:
    """Mirror the question normalization performed before ROSA planning."""
    return question if question.endswith("?") else f"{question}?"


def expected_product_compilation(
    question: str,
    raw_plan: str,
) -> tuple[str, dict[str, Any]]:
    """Replay the isolated wrapper without invoking a model or retriever."""
    normalized_question = runtime_question(question)
    _, initial = compile_shoppingbench_plan(normalized_question, raw_plan, max_nodes=5)
    selected, fallback = select_product_query(
        normalized_question,
        initial.get("queries") or [],
    )
    if not fallback["applied"]:
        return (initial.get("queries") or [""])[0], fallback

    _, compiled = compile_shoppingbench_plan(
        normalized_question,
        f"<plan><search>{selected}</search></plan>",
        max_nodes=5,
    )
    return (compiled.get("queries") or [""])[0], fallback


def executed_search_queries(row: dict[str, Any]) -> list[str]:
    queries = []
    for node in (row.get("raw_trace") or {}).get("nodes_execution") or []:
        if node.get("type") != "search":
            continue
        query = str(node.get("executed_query") or "").strip()
        if query:
            queries.append(query)
    return queries


def compare(baseline_path: Path, candidate_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    baseline, candidate = read_rows(baseline_path), read_rows(candidate_path)
    if set(baseline) != set(candidate) or len(candidate) != 16:
        raise ValueError("paired Product sample IDs are not exactly 16 and identical")

    paired = []
    for sample_id in sorted(candidate):
        old, new = baseline[sample_id], candidate[sample_id]
        old_trace, new_trace = old.get("raw_trace") or {}, new.get("raw_trace") or {}
        raw_plan = new_trace.get("raw_plan_output", "")
        model_queries = extract_atomic_search_queries(raw_plan)
        expected_query, expected_fallback = expected_product_compilation(
            new.get("query", ""), raw_plan
        )
        compiler = (new_trace.get("contract_protocol") or {}).get("plan_compiler") or {}
        observed_fallback = compiler.get("product_query_fallback") or {}
        compiled_query = (compiler.get("queries") or [""])[0]
        executed_queries = executed_search_queries(new)
        paired.append({
            "sample_id": sample_id,
            "input_query_identity": old.get("query") == new.get("query"),
            "raw_planner_identity": (
                old_trace.get("raw_plan_output") == new_trace.get("raw_plan_output")
            ),
            "planner_output_override_used": new_trace.get("planner_output_override_used") is True,
            "plan_override_unused": new_trace.get("plan_override_used") is False,
            "compiler_applied": compiler.get("applied") is True,
            "compiler_kind_correct": compiler.get("request_kind") == "product",
            "compiled_query": compiled_query,
            "expected_query": expected_query,
            "executed_queries": executed_queries,
            "compiled_query_executed": executed_queries == [compiled_query],
            "fallback_expected": expected_fallback["applied"],
            "fallback_observed": observed_fallback.get("applied") is expected_fallback["applied"],
            "gold_unused": compiler.get("gold_used") is False and observed_fallback.get("gold_used") is False,
            "baseline_failure": failed(old),
            "candidate_failure": failed(new),
            "baseline_fraction": float((old.get("score") or {}).get("exact_id_fraction") or 0.0),
            "candidate_fraction": float((new.get("score") or {}).get("exact_id_fraction") or 0.0),
        })

    baseline_summary = summarize(list(baseline.values()))
    candidate_summary = summarize(list(candidate.values()))
    identity_pass = all(
        row["input_query_identity"]
        and row["raw_planner_identity"]
        and row["planner_output_override_used"]
        and row["plan_override_unused"]
        for row in paired
    )
    compiler_pass = all(
        row["compiler_applied"]
        and row["compiler_kind_correct"]
        and row["compiled_query"] == row["expected_query"]
        and row["compiled_query_executed"]
        and row["fallback_observed"]
        and row["gold_unused"]
        for row in paired
    )
    activation_pass = sum(row["fallback_expected"] for row in paired) == 4
    outcome_pass = (
        candidate_summary["failures"] <= baseline_summary["failures"]
        and candidate_summary["mean_exact_id_fraction"]
        >= baseline_summary["mean_exact_id_fraction"]
    )
    report = {
        "schema": "shoppingbench-product-query-fallback-online-paired-v2",
        "evidence_level": "online Product-only invalid-query fallback causal check",
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "identity_pass": identity_pass,
        "compiler_pass": compiler_pass,
        "activation_pass": activation_pass,
        "outcome_nonregression_pass": outcome_pass,
        "passed": identity_pass and compiler_pass and activation_pass and outcome_pass,
        "training_authorized": False,
        "paired": paired,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.output_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "paired"}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
