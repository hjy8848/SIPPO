#!/usr/bin/env python3
"""Compare the isolated Product selector with the frozen query-fallback run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TRIGGER = "product-0011"


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
        )
        / len(rows),
    }


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
        selection = new_trace.get("product_selection_fallback") or {}
        decisions = selection.get("decisions") or []
        applied = [decision for decision in decisions if decision.get("applied") is True]
        query_fallback_old = (
            ((old_trace.get("contract_protocol") or {}).get("plan_compiler") or {}).get(
                "product_query_fallback"
            )
            or {}
        )
        query_fallback_new = (
            ((new_trace.get("contract_protocol") or {}).get("plan_compiler") or {}).get(
                "product_query_fallback"
            )
            or {}
        )
        paired.append(
            {
                "sample_id": sample_id,
                "input_query_identity": old.get("query") == new.get("query"),
                "raw_planner_identity": (
                    old_trace.get("raw_plan_output") == new_trace.get("raw_plan_output")
                ),
                "query_fallback_identity": query_fallback_old == query_fallback_new,
                "planner_output_override_used": (
                    new_trace.get("planner_output_override_used") is True
                ),
                "selection_enabled": selection.get("enabled") is True,
                "selection_gold_unused": (
                    selection.get("gold_used") is False
                    and all(decision.get("gold_used") is False for decision in decisions)
                ),
                "decision_count": len(decisions),
                "applied_count": len(applied),
                "selected_ids": [decision.get("selected_id") for decision in applied],
                "baseline_ids": old.get("predicted_product_ids") or [],
                "candidate_ids": new.get("predicted_product_ids") or [],
                "prediction_identity": (
                    (old.get("predicted_product_ids") or [])
                    == (new.get("predicted_product_ids") or [])
                ),
                "baseline_exact": bool(
                    (old.get("score") or {}).get("exact_sequence_success")
                ),
                "candidate_exact": bool(
                    (new.get("score") or {}).get("exact_sequence_success")
                ),
            }
        )

    baseline_summary = summarize(list(baseline.values()))
    candidate_summary = summarize(list(candidate.values()))
    trigger_rows = [row["sample_id"] for row in paired if row["applied_count"]]
    identity_pass = all(
        row["input_query_identity"]
        and row["raw_planner_identity"]
        and row["query_fallback_identity"]
        and row["planner_output_override_used"]
        for row in paired
    )
    metadata_pass = all(
        row["selection_enabled"]
        and row["selection_gold_unused"]
        and row["decision_count"] == 1
        for row in paired
    )
    activation_pass = trigger_rows == [EXPECTED_TRIGGER]
    nontrigger_identity_pass = all(
        row["prediction_identity"]
        for row in paired
        if row["sample_id"] != EXPECTED_TRIGGER
    )
    trigger_outcome_pass = all(
        not row["baseline_exact"] and row["candidate_exact"]
        for row in paired
        if row["sample_id"] == EXPECTED_TRIGGER
    )
    outcome_pass = (
        candidate_summary["failures"] == 0
        and candidate_summary["exact_sequences"] >= 13
        and candidate_summary["exact_sequences"] > baseline_summary["exact_sequences"]
        and candidate_summary["mean_exact_id_fraction"]
        >= baseline_summary["mean_exact_id_fraction"]
    )
    report = {
        "schema": "shoppingbench-product-selection-fallback-online-paired-v1",
        "evidence_level": "online Product-only conservative-selection causal check",
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "identity_pass": identity_pass,
        "metadata_pass": metadata_pass,
        "activation_pass": activation_pass,
        "nontrigger_prediction_identity_pass": nontrigger_identity_pass,
        "trigger_outcome_pass": trigger_outcome_pass,
        "outcome_pass": outcome_pass,
        "passed": all(
            (
                identity_pass,
                metadata_pass,
                activation_pass,
                nontrigger_identity_pass,
                trigger_outcome_pass,
                outcome_pass,
            )
        ),
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
