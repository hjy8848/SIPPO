#!/usr/bin/env python3
"""Audit whether strict Reason-to-Search traces can recover a selected entity."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any, Mapping, Sequence


REF_RE = re.compile(r"#(\d+)#")


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def contains_name(text: Any, name: Any) -> bool:
    key = normalize(name)
    return bool(key and key in normalize(text))


def conclusion(node: Mapping[str, Any]) -> str:
    return str(node.get("conclusion") or node.get("raw_output") or "").strip()


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_key = normalize(row.get("question"))
            blueprint = row.get("blueprint") or {}
            if not question_key or question_key in output:
                raise RuntimeError(f"Invalid or duplicate metadata row {line_number}")
            output[question_key] = {
                "sample_id": str(row.get("trajectory_id") or ""),
                "question": str(row.get("question") or ""),
                "operation": str((row.get("reasoning") or {}).get("operation") or "unknown"),
                "winner": str(blueprint.get("selected_candidate") or ""),
                "candidate_a": str(blueprint.get("candidate_a") or ""),
                "candidate_b": str(blueprint.get("candidate_b") or ""),
            }
    return output


def candidate_names_in_query(query: str, metadata: Mapping[str, str]) -> list[str]:
    return [
        name
        for name in (metadata.get("candidate_a", ""), metadata.get("candidate_b", ""))
        if contains_name(query, name)
    ]


def audit_workflow(
    nodes: Sequence[Mapping[str, Any]], metadata: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Return one row per Search whose raw query consumes an earlier Reason."""

    previous: dict[str, Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        node_type = str(node.get("type") or "").casefold()
        if node_type != "search":
            if node_id:
                previous[node_id] = node
            continue

        raw_query = str(node.get("query") or "")
        reason_ids = [
            reference
            for reference in REF_RE.findall(raw_query)
            if str(previous.get(reference, {}).get("type") or "").casefold() == "reason"
        ]
        for reason_id in reason_ids:
            reason = previous[reason_id]
            reason_value = conclusion(reason)
            parent_ids = REF_RE.findall(str(reason.get("query") or ""))
            parent_searches = [
                previous[parent_id]
                for parent_id in parent_ids
                if str(previous.get(parent_id, {}).get("type") or "").casefold() == "search"
            ]
            value_matches = [
                parent
                for parent in parent_searches
                if normalize(conclusion(parent)) == normalize(reason_value)
                and normalize(reason_value)
            ]
            matched_parent = value_matches[0] if len(value_matches) == 1 else None
            matched_names = (
                candidate_names_in_query(str(matched_parent.get("query") or ""), metadata)
                if matched_parent
                else []
            )
            recovered_candidate = matched_names[0] if len(matched_names) == 1 else ""
            winner = metadata.get("winner", "")
            executed_query = str(node.get("executed_query") or "")
            executed_names = candidate_names_in_query(executed_query, metadata)
            binding_correct = bool(
                recovered_candidate and normalize(recovered_candidate) == normalize(winner)
            )
            current_query_mentions_winner = contains_name(executed_query, winner)
            current_query_unique_candidate = len(executed_names) == 1
            current_query_binding_correct = bool(
                current_query_unique_candidate
                and normalize(executed_names[0]) == normalize(winner)
            )
            rows.append(
                {
                    **metadata,
                    "reason_id": reason_id,
                    "search_id": node_id,
                    "reason_query": str(reason.get("query") or ""),
                    "reason_resolved_query": str(reason.get("resolved_query") or ""),
                    "reason_conclusion": reason_value,
                    "parent_search_count": len(parent_searches),
                    "value_match_count": len(value_matches),
                    "unique_value_match": len(value_matches) == 1,
                    "matched_parent_id": str(matched_parent.get("id") or "") if matched_parent else "",
                    "matched_parent_query": str(matched_parent.get("query") or "") if matched_parent else "",
                    "recovered_candidate": recovered_candidate,
                    "unique_candidate_binding": bool(recovered_candidate),
                    "binding_correct": binding_correct,
                    "binding_wrong": bool(recovered_candidate and not binding_correct),
                    "downstream_raw_query": raw_query,
                    "downstream_resolved_query": str(node.get("resolved_query") or ""),
                    "downstream_executed_query": executed_query,
                    "current_executed_candidates": " || ".join(executed_names),
                    "current_query_mentions_winner": current_query_mentions_winner,
                    "current_query_unique_candidate": current_query_unique_candidate,
                    "current_query_binding_correct": current_query_binding_correct,
                    "current_query_ambiguous": len(executed_names) > 1,
                    "correction_opportunity": bool(
                        binding_correct and not current_query_binding_correct
                    ),
                }
            )
        if node_id:
            previous[node_id] = node
    return rows


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    workflow_count: int,
    minimum_edges: int = 30,
    minimum_binding_coverage: float = 0.80,
    minimum_binding_precision: float = 0.95,
    minimum_correction_opportunities: int = 10,
    maximum_wrong_bindings: int = 0,
) -> dict[str, Any]:
    strict_workflows = len({str(row["sample_id"]) for row in rows})
    unique_values = sum(bool(row["unique_value_match"]) for row in rows)
    bound = sum(bool(row["unique_candidate_binding"]) for row in rows)
    correct = sum(bool(row["binding_correct"]) for row in rows)
    wrong = sum(bool(row["binding_wrong"]) for row in rows)
    current_mentions_winner = sum(bool(row["current_query_mentions_winner"]) for row in rows)
    current_unique = sum(bool(row["current_query_unique_candidate"]) for row in rows)
    current_correct = sum(bool(row["current_query_binding_correct"]) for row in rows)
    current_ambiguous = sum(bool(row["current_query_ambiguous"]) for row in rows)
    corrections = sum(bool(row["correction_opportunity"]) for row in rows)

    by_operation: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["operation"])].append(row)
    for operation, values in sorted(grouped.items()):
        operation_bound = sum(bool(row["unique_candidate_binding"]) for row in values)
        operation_correct = sum(bool(row["binding_correct"]) for row in values)
        by_operation[operation] = {
            "edges": len(values),
            "unique_candidate_bindings": operation_bound,
            "binding_coverage": safe_rate(operation_bound, len(values)),
            "binding_precision": safe_rate(operation_correct, operation_bound),
            "current_query_mentions_winner": sum(
                bool(row["current_query_mentions_winner"]) for row in values
            ),
            "current_query_unique_candidate": sum(
                bool(row["current_query_unique_candidate"]) for row in values
            ),
            "current_query_binding_correct": sum(
                bool(row["current_query_binding_correct"]) for row in values
            ),
            "current_query_ambiguous": sum(
                bool(row["current_query_ambiguous"]) for row in values
            ),
            "correction_opportunities": sum(bool(row["correction_opportunity"]) for row in values),
        }

    hard_gates = {
        "enough_strict_edges": len(rows) >= minimum_edges,
        "binding_coverage": safe_rate(bound, len(rows)) >= minimum_binding_coverage,
        "binding_precision": safe_rate(correct, bound) >= minimum_binding_precision,
        "correction_opportunities": corrections >= minimum_correction_opportunities,
        "no_wrong_bindings": wrong <= maximum_wrong_bindings,
    }
    return {
        "method": "reason_to_search_typed_handoff_recoverability_v1",
        "workflow_count": workflow_count,
        "strict_workflows": strict_workflows,
        "strict_edges": len(rows),
        "unique_value_matches": unique_values,
        "unique_candidate_bindings": bound,
        "binding_coverage": safe_rate(bound, len(rows)),
        "correct_bindings": correct,
        "wrong_bindings": wrong,
        "binding_precision": safe_rate(correct, bound),
        "current_query_mentions_winner": current_mentions_winner,
        "current_query_unique_candidate": current_unique,
        "current_query_binding_correct": current_correct,
        "current_query_ambiguous": current_ambiguous,
        "correction_opportunities": corrections,
        "by_operation": by_operation,
        "thresholds": {
            "minimum_edges": minimum_edges,
            "minimum_binding_coverage": minimum_binding_coverage,
            "minimum_binding_precision": minimum_binding_precision,
            "minimum_correction_opportunities": minimum_correction_opportunities,
            "maximum_wrong_bindings": maximum_wrong_bindings,
        },
        "hard_gates": hard_gates,
        "passed": all(hard_gates.values()),
        "interpretation": (
            "A pass establishes only that logged Reason conclusions can be safely bound "
            "back to candidate entities. It does not establish an end-to-end accuracy gain."
        ),
    }


def write_outputs(output_dir: Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (output_dir / "edges.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    payload = json.loads(args.trajectory.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for workflow in payload:
        key = normalize(workflow.get("question"))
        if key not in metadata:
            raise RuntimeError(f"Missing metadata for question: {workflow.get('question')}")
        trace = workflow.get("raw_trace") or {}
        edge_rows = audit_workflow(trace.get("nodes_execution") or [], metadata[key])
        for row in edge_rows:
            row["em"] = float(workflow.get("em") or 0.0)
        rows.extend(edge_rows)

    report = summarize(rows, workflow_count=len(payload))
    report["metadata"] = str(args.metadata.resolve())
    report["trajectory"] = str(args.trajectory.resolve())
    write_outputs(args.output_dir, report, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
