#!/usr/bin/env python3
"""Build grounded ShoppingBench SFT rows from successful contract predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


TASKS = ("product", "shop", "voucher")
ACTOR_AGENTS = {"PlanningAgent", "AnswerGenerationAgent", "ReasonerAgent"}
_PRODUCT_ID_RE = re.compile(r"\bProduct ID:\s*(\d+)\b", re.I)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def compiler_metadata(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = trace.get("contract_protocol")
    return (protocol.get("plan_compiler") or {}) if isinstance(protocol, Mapping) else {}


def _event_node_ids(events: list[Mapping[str, Any]], agent: str) -> set[str]:
    return {
        str((event.get("metadata") or {}).get("node_id") or "")
        for event in events
        if event.get("agent_name") == agent
    }


def validate_prediction(row: Mapping[str, Any], task: str) -> list[str]:
    reasons: list[str] = []
    trace = row.get("raw_trace") if isinstance(row.get("raw_trace"), Mapping) else {}
    expected = [str(value) for value in row.get("expected_product_ids") or []]
    predicted = [str(value) for value in row.get("predicted_product_ids") or []]
    score = row.get("score") if isinstance(row.get("score"), Mapping) else {}
    if row.get("arm") != "contract":
        reasons.append("not contract arm")
    if any(row.get(key) for key in ("format_error", "contract_error", "pipeline_error", "request_error")):
        reasons.append("prediction has execution or format error")
    if not score.get("exact_sequence_success") or predicted != expected or not expected:
        reasons.append("final product ID sequence is not exact")
    if trace.get("protocol_profile") != "shoppingbench":
        reasons.append("wrong protocol profile")
    ablation = trace.get("ablation_config") or {}
    if not ablation.get("contract_preserving_dag_v1") or not ablation.get("shopping_system_plan_compiler_v1"):
        reasons.append("contract/compiler runtime flags missing")
    if trace.get("plan_override_used"):
        reasons.append("plan override was used")
    compiler = compiler_metadata(trace)
    if not compiler.get("applied") or compiler.get("gold_used") is not False:
        reasons.append("system compiler did not prove no-gold application")
    if compiler.get("request_kind") != task:
        reasons.append("compiler request kind mismatch")
    protocol = trace.get("contract_protocol") or {}
    if protocol.get("validation_errors"):
        reasons.append("compiled plan has validation errors")
    plan_output = str(trace.get("plan_output") or "").strip()
    if not plan_output.startswith("<plan ") or not plan_output.endswith("</plan>"):
        reasons.append("compiled plan is missing")

    nodes = trace.get("nodes_execution") if isinstance(trace.get("nodes_execution"), list) else []
    searches = [node for node in nodes if node.get("type") == "search"]
    reason_nodes = [node for node in nodes if node.get("type") == "reason"]
    if len(searches) != len(expected):
        reasons.append("search count does not match expected product count")
    else:
        for product_id, node in zip(expected, searches, strict=True):
            context_ids = set(_PRODUCT_ID_RE.findall(str(node.get("search_context") or "")))
            if product_id not in context_ids:
                reasons.append(f"gold product {product_id} missing from corresponding Search")

    events = [event for event in trace.get("sft_events") or [] if isinstance(event, Mapping)]
    planner_events = [event for event in events if event.get("agent_name") == "PlanningAgent"]
    if len(planner_events) != 1:
        reasons.append("expected exactly one Planner SFT event")
    search_ids = {str(node.get("id")) for node in searches}
    model_reason_ids = {
        str(node.get("id"))
        for node in reason_nodes
        if not str(node.get("reason_execution_method") or "").startswith("deterministic_")
    }
    if not search_ids.issubset(_event_node_ids(events, "AnswerGenerationAgent")):
        reasons.append("at least one Search lacks an SFT event")
    if not model_reason_ids.issubset(_event_node_ids(events, "ReasonerAgent")):
        reasons.append("at least one Reason lacks an SFT event")

    events_by_node = {
        (str(event.get("agent_name")), str((event.get("metadata") or {}).get("node_id"))): event
        for event in events
    }
    for product_id, node in zip(expected, searches):
        event = events_by_node.get(("AnswerGenerationAgent", str(node.get("id"))))
        if event is not None and product_id not in str(event.get("response") or ""):
            reasons.append(f"Search SFT response omits selected product {product_id}")
    if reason_nodes:
        final_event = events_by_node.get(("ReasonerAgent", str(reason_nodes[-1].get("id"))))
        if final_event is not None and any(
            product_id not in str(final_event.get("response") or "") for product_id in expected
        ):
            reasons.append("sink Reason SFT response omits an exact product ID")
    return list(dict.fromkeys(reasons))


def sft_rows_for_prediction(row: Mapping[str, Any], task: str) -> list[dict[str, Any]]:
    trace = row["raw_trace"]
    result = []
    for event_index, event in enumerate(trace.get("sft_events") or []):
        if not isinstance(event, Mapping) or event.get("agent_name") not in ACTOR_AGENTS:
            continue
        messages = deepcopy(event.get("messages") or [])
        if not messages or any(not isinstance(message, dict) for message in messages):
            raise ValueError(f"{row.get('sample_id')}: malformed SFT messages")
        response = str(event.get("response") or "").strip()
        target_source = "model_grounded_action"
        if event.get("agent_name") == "PlanningAgent":
            response = str(trace.get("plan_output") or "").strip()
            target_source = "system_compiled_model_search_semantics"
        if not response:
            raise ValueError(f"{row.get('sample_id')}: empty SFT response")
        messages.append({"role": "assistant", "content": response, "loss_mask": 1})
        result.append(
            {
                "messages": messages,
                "enable_thinking": False,
                "agent_name": str(event["agent_name"]),
                "trajectory_id": str(row["sample_id"]),
                "task": task,
                "event_index": event_index,
                "target_source": target_source,
                "event_metadata": json.dumps(event.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
            }
        )
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(
    input_root: Path,
    output_dir: Path,
    *,
    expected_per_task: int,
    min_total: int,
    min_per_task: int,
) -> tuple[dict[str, Any], bool]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    input_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()

    for task in TASKS:
        path = input_root / task / "results.jsonl"
        rows = load_jsonl(path)
        input_counts[task] = len(rows)
        if len(rows) != expected_per_task:
            raise ValueError(f"{task}: expected {expected_per_task} rows, found {len(rows)}")
        for row in rows:
            reasons = validate_prediction(row, task)
            if reasons:
                rejected.append({"sample_id": row.get("sample_id"), "task": task, "reasons": reasons})
                rejection_reasons.update(reasons)
                continue
            accepted_counts[task] += 1
            accepted.append(
                {
                    "sample_id": row["sample_id"],
                    "task": task,
                    "query": row["query"],
                    "expected_product_ids": row["expected_product_ids"],
                    "predicted_product_ids": row["predicted_product_ids"],
                    "plan_output": row["raw_trace"]["plan_output"],
                }
            )
            sft_rows.extend(sft_rows_for_prediction(row, task))

    hard_gates = {
        "all_inputs_materialized": sum(input_counts.values()) == expected_per_task * len(TASKS),
        "minimum_total_accepted": len(accepted) >= min_total,
        "minimum_per_task_accepted": all(accepted_counts[task] >= min_per_task for task in TASKS),
        "planner_targets_system_compiled": all(
            row["target_source"] == "system_compiled_model_search_semantics"
            for row in sft_rows
            if row["agent_name"] == "PlanningAgent"
        ),
        "no_query_rewrite_events": all(row["agent_name"] != "QueryRewriteAgent" for row in sft_rows),
    }
    passed = all(hard_gates.values())
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "accepted_trajectories.jsonl", accepted)
    write_jsonl(output_dir / "rejected_trajectories.jsonl", rejected)
    write_jsonl(output_dir / "sft_events.jsonl", sft_rows)
    report = {
        "schema": "shoppingbench-grounded-sft-pilot-v1",
        "evidence_level": "online grounded-SFT engineering gate",
        "input_root": str(input_root),
        "expected_per_task": expected_per_task,
        "input_counts": dict(input_counts),
        "accepted_trajectories": len(accepted),
        "accepted_by_task": {task: accepted_counts[task] for task in TASKS},
        "sft_rows": len(sft_rows),
        "sft_role_counts": dict(Counter(row["agent_name"] for row in sft_rows)),
        "top_rejection_reasons": rejection_reasons.most_common(20),
        "hard_gates": hard_gates,
        "passed": passed,
        "training_authorized": False,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-per-task", type=int, required=True)
    parser.add_argument("--min-total", type=int, default=8)
    parser.add_argument("--min-per-task", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, passed = build(
        args.input_root,
        args.output_dir,
        expected_per_task=args.expected_per_task,
        min_total=args.min_total,
        min_per_task=args.min_per_task,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
