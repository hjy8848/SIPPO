#!/usr/bin/env python3
"""Compare the visible-atom shop fallback against frozen online trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qa_manager.ROSA.shoppingbench_protocol import (
    classify_shopping_request,
    extract_atomic_search_queries,
    extract_visible_product_queries,
)


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row["sample_id"]): row for row in rows}


def failed(row: dict[str, Any]) -> bool:
    return any(
        row.get(key)
        for key in ("format_error", "contract_error", "pipeline_error", "request_error")
    )


def exact_fraction(row: dict[str, Any]) -> float:
    return float((row.get("score") or {}).get("exact_id_fraction") or 0.0)


def query_rewrites(row: dict[str, Any]) -> int:
    events = (row.get("raw_trace") or {}).get("sft_events") or []
    return sum(
        isinstance(event, dict) and event.get("agent_name") == "QueryRewriteAgent"
        for event in events
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "failures": sum(failed(row) for row in rows),
        "mean_exact_id_fraction": sum(exact_fraction(row) for row in rows) / len(rows),
        "exact_sequences": sum(
            bool((row.get("score") or {}).get("exact_sequence_success")) for row in rows
        ),
        "first_id_exact": sum(
            bool((row.get("score") or {}).get("first_id_exact")) for row in rows
        ),
        "retrieval_calls": sum(int(row.get("retrieval_calls") or 0) for row in rows),
    }


def compare(baseline_path: Path, candidate_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    baseline, candidate = read_rows(baseline_path), read_rows(candidate_path)
    if set(baseline) != set(candidate) or len(baseline) != 16:
        raise ValueError("paired shop sample IDs are not exactly 16 and identical")

    paired = []
    for sample_id in sorted(baseline):
        old, new = baseline[sample_id], candidate[sample_id]
        old_trace, new_trace = old.get("raw_trace") or {}, new.get("raw_trace") or {}
        question = str(new.get("query") or "")
        raw_plan = str(new_trace.get("raw_plan_output") or "")
        model_queries = extract_atomic_search_queries(raw_plan)
        visible_queries = extract_visible_product_queries(question)
        compiler = (new_trace.get("contract_protocol") or {}).get("plan_compiler") or {}
        compiled_queries = list(compiler.get("queries") or [])
        fallback_expected = len(model_queries) < 2 and len(visible_queries) >= 2
        fallback_observed = fallback_expected and compiled_queries == visible_queries
        paired.append(
            {
                "sample_id": sample_id,
                "raw_planner_identity": (
                    old_trace.get("raw_plan_output") == new_trace.get("raw_plan_output")
                ),
                "planner_output_override_used": (
                    new_trace.get("planner_output_override_used") is True
                ),
                "plan_override_unused": new_trace.get("plan_override_used") is False,
                "classifier_correct": classify_shopping_request(question) == "shop",
                "compiler_applied": compiler.get("applied") is True,
                "compiler_kind_correct": compiler.get("request_kind") == "shop",
                "fallback_expected": fallback_expected,
                "fallback_observed": fallback_observed,
                "model_queries_preserved": (
                    fallback_expected or compiled_queries == model_queries
                ),
                "query_rewrite_events": query_rewrites(new),
                "baseline_failure": failed(old),
                "candidate_failure": failed(new),
                "baseline_fraction": exact_fraction(old),
                "candidate_fraction": exact_fraction(new),
            }
        )

    baseline_summary = summarize(list(baseline.values()))
    candidate_summary = summarize(list(candidate.values()))
    identity_pass = all(
        row["raw_planner_identity"]
        and row["planner_output_override_used"]
        and row["plan_override_unused"]
        for row in paired
    )
    compiler_pass = all(
        row["classifier_correct"]
        and row["compiler_applied"]
        and row["compiler_kind_correct"]
        for row in paired
    )
    fallback_pass = (
        sum(row["fallback_expected"] for row in paired) == 7
        and all(
            row["fallback_observed"] if row["fallback_expected"] else row["model_queries_preserved"]
            for row in paired
        )
    )
    no_rewrite = sum(row["query_rewrite_events"] for row in paired) == 0
    outcome_pass = (
        candidate_summary["failures"] <= baseline_summary["failures"]
        and candidate_summary["mean_exact_id_fraction"]
        >= baseline_summary["mean_exact_id_fraction"]
    )
    report = {
        "schema": "shoppingbench-visible-atomic-shop-online-paired-v1",
        "evidence_level": "online shop-only fallback causal check",
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "identity_pass": identity_pass,
        "compiler_pass": compiler_pass,
        "fallback_pass": fallback_pass,
        "query_rewrite_zero": no_rewrite,
        "outcome_nonregression_pass": outcome_pass,
        "passed": identity_pass and compiler_pass and fallback_pass and no_rewrite and outcome_pass,
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
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"paired"}},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
