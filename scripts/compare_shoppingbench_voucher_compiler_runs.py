#!/usr/bin/env python3
"""Compare frozen-Planner ShoppingBench voucher compiler interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FAILURE_FIELDS = ("format_error", "contract_error", "pipeline_error", "request_error")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in rows:
                raise ValueError(f"missing or duplicate sample_id at line {line_number}")
            rows[sample_id] = row
    return rows


def score(row: dict[str, Any], key: str) -> float:
    return float((row.get("score") or {}).get(key) or 0.0)


def failed(row: dict[str, Any]) -> bool:
    return any(row.get(field) for field in FAILURE_FIELDS)


def search_queries(row: dict[str, Any]) -> list[str]:
    trace = row.get("raw_trace") or {}
    return [
        str(node.get("executed_query") or "")
        for node in trace.get("nodes_execution") or []
        if node.get("type") == "search"
    ]


def query_rewrite_events(row: dict[str, Any]) -> int:
    trace = row.get("raw_trace") or {}
    return sum(
        bool(node.get("rewrite_queries"))
        for node in trace.get("nodes_execution") or []
        if node.get("type") == "search"
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "rows": count,
        "mean_exact_id_fraction": (
            sum(score(row, "exact_id_fraction") for row in rows) / count if count else 0.0
        ),
        "exact_sequence_successes": sum(
            bool((row.get("score") or {}).get("exact_sequence_success")) for row in rows
        ),
        "first_id_exact_successes": sum(
            bool((row.get("score") or {}).get("first_id_exact")) for row in rows
        ),
        "failures": sum(failed(row) for row in rows),
        "search_nodes": sum(int(row.get("search_nodes") or 0) for row in rows),
        "retrieval_calls": sum(int(row.get("retrieval_calls") or 0) for row in rows),
        "query_rewrite_events": sum(query_rewrite_events(row) for row in rows),
    }


def compare(
    baseline_path: Path,
    candidate_path: Path,
    overrides_path: Path,
    output_dir: Path,
    expected_rows: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    baseline = load_rows(baseline_path)
    candidate = load_rows(candidate_path)
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    if len(candidate) != expected_rows:
        raise ValueError(f"expected {expected_rows} candidate rows, found {len(candidate)}")
    selected = sorted(candidate)
    missing = [sample_id for sample_id in selected if sample_id not in baseline or sample_id not in overrides]
    if missing:
        raise ValueError(f"missing paired input for: {missing}")

    pairs: list[dict[str, Any]] = []
    identity_controls: list[str] = []
    affected: list[str] = []
    for sample_id in selected:
        old = baseline[sample_id]
        new = candidate[sample_id]
        old_trace = old.get("raw_trace") or {}
        new_trace = new.get("raw_trace") or {}
        frozen = overrides[sample_id]
        if old.get("query") != new.get("query"):
            raise ValueError(f"{sample_id}: query mismatch")
        if old.get("expected_product_ids") != new.get("expected_product_ids"):
            raise ValueError(f"{sample_id}: expected ID mismatch")
        if old_trace.get("raw_plan_output") != frozen or new_trace.get("raw_plan_output") != frozen:
            raise ValueError(f"{sample_id}: raw Planner identity mismatch")

        compiled_plan_identity = old_trace.get("plan_output") == new_trace.get("plan_output")
        group = "identity_control" if compiled_plan_identity else "compiler_affected"
        (identity_controls if compiled_plan_identity else affected).append(sample_id)
        old_queries = search_queries(old)
        new_queries = search_queries(new)
        old_ids = list(old.get("predicted_product_ids") or [])
        new_ids = list(new.get("predicted_product_ids") or [])
        terminal_identity = (
            old_trace.get("final_answer") == new_trace.get("final_answer")
            and old_ids == new_ids
            and (old.get("score") or {}) == (new.get("score") or {})
        )
        pairs.append(
            {
                "sample_id": sample_id,
                "group": group,
                "compiled_plan_identity": compiled_plan_identity,
                "executed_query_identity": old_queries == new_queries,
                "terminal_identity": terminal_identity,
                "baseline_search_queries": old_queries,
                "candidate_search_queries": new_queries,
                "baseline_predicted_ids": old_ids,
                "candidate_predicted_ids": new_ids,
                "baseline_exact_id_fraction": score(old, "exact_id_fraction"),
                "candidate_exact_id_fraction": score(new, "exact_id_fraction"),
                "delta_exact_id_fraction": (
                    score(new, "exact_id_fraction") - score(old, "exact_id_fraction")
                ),
                "baseline_failure": failed(old),
                "candidate_failure": failed(new),
            }
        )

    old_rows = [baseline[sample_id] for sample_id in selected]
    new_rows = [candidate[sample_id] for sample_id in selected]
    old_summary = summarize(old_rows)
    new_summary = summarize(new_rows)
    controls = [pair for pair in pairs if pair["group"] == "identity_control"]
    affected_pairs = [pair for pair in pairs if pair["group"] == "compiler_affected"]
    identity_gate = bool(controls) and all(
        pair["executed_query_identity"] and pair["terminal_identity"] for pair in controls
    )
    runtime_gate = (
        bool(affected_pairs)
        and new_summary["failures"] <= old_summary["failures"]
        and identity_gate
    )
    delta = (
        new_summary["mean_exact_id_fraction"] - old_summary["mean_exact_id_fraction"]
    )
    affected_delta = (
        sum(pair["delta_exact_id_fraction"] for pair in affected_pairs) / len(affected_pairs)
        if affected_pairs
        else 0.0
    )
    report = {
        "protocol": "frozen raw-Planner compiler-v1-to-v2 paired intervention",
        "expected_rows": expected_rows,
        "baseline": old_summary,
        "candidate": new_summary,
        "delta_mean_exact_id_fraction": delta,
        "identity_controls": {
            "rows": len(controls),
            "sample_ids": identity_controls,
            "passed": identity_gate,
            "criterion": "unchanged compiled plan implies identical executed queries and terminal outcome",
        },
        "compiler_affected": {
            "rows": len(affected_pairs),
            "sample_ids": affected,
            "mean_delta_exact_id_fraction": affected_delta,
            "wins": sum(pair["delta_exact_id_fraction"] > 0 for pair in affected_pairs),
            "losses": sum(pair["delta_exact_id_fraction"] < 0 for pair in affected_pairs),
            "ties": sum(pair["delta_exact_id_fraction"] == 0 for pair in affected_pairs),
        },
        "runtime_causal_gate_passed": runtime_gate,
        "outcome_direction_positive": delta > 0,
        "training_authorized": False,
        "scope": "mechanism pilot only; requires a larger matched replication before data or training promotion",
        "inputs": {
            "baseline": str(baseline_path),
            "baseline_sha256": sha256(baseline_path),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "planner_output_overrides": str(overrides_path),
            "planner_output_overrides_sha256": sha256(overrides_path),
        },
        "pairs": pairs,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--planner-output-overrides", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(
        args.baseline,
        args.candidate,
        args.planner_output_overrides,
        args.output_dir,
        args.expected_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
