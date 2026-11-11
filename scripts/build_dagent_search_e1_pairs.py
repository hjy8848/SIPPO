#!/usr/bin/env python3
"""Build reproducible OPSD examples from repeated DAGENT rollouts.

The legacy failed_sibling mode scores a failed sibling action under standard
and privileged prompts. The aligned_success mode instead scores each successful
on-policy action from its own inference-time state and redacts that current
action from the privileged successful-workflow reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROLES = (
    "PlanningAgent",
    "QueryRewriteAgent",
    "AnswerGenerationAgent",
    "ReasonerAgent",
    "AnswerSummarizationAgent",
)


@dataclass(frozen=True)
class Workflow:
    source_file: str
    sample_index: str
    rollout_index: int
    question: str
    outcome: int
    retrieval_calls: int
    reason_to_search: bool
    events: tuple[dict[str, Any], ...]
    serialized: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-log", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("failed_sibling", "aligned_success"),
        default="failed_sibling",
        help=(
            "failed_sibling preserves the legacy objective. aligned_success "
            "distills successful on-policy actions from their own inference-time states."
        ),
    )
    parser.add_argument("--require-grounded-by-nodes", action="store_true")
    parser.add_argument("--require-no-plan-warnings", action="store_true")
    parser.add_argument("--require-complete-nodes", action="store_true")
    parser.add_argument(
        "--roles",
        default=",".join(DEFAULT_ROLES),
        help="Comma-separated policy roles to distill. The default keeps every DAGENT policy role.",
    )
    parser.add_argument(
        "--max-reference-chars",
        type=int,
        default=0,
        help="Optional serialized-reference character cap; 0 keeps the full successful workflow.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1)")
    if args.max_reference_chars < 0:
        parser.error("--max-reference-chars must be non-negative")
    args.roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    if not args.roles:
        parser.error("--roles must contain at least one role")
    return args


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if role:
            normalized.append({"role": role, "content": content})
    return normalized


def extract_events(row: dict[str, Any], allowed_roles: set[str]) -> tuple[dict[str, Any], ...]:
    trace = row.get("raw_trace") or {}
    raw_events = trace.get("sft_events") or row.get("sft_events") or []
    events = []
    for step, event in enumerate(raw_events):
        if not isinstance(event, dict):
            continue
        role = str(event.get("agent_name") or "").strip()
        response = str(event.get("response") or "").strip()
        messages = normalize_messages(event.get("messages"))
        if role not in allowed_roles or not messages or not response:
            continue
        events.append(
            {
                "workflow_step": step,
                "role": role,
                "messages": messages,
                "response": response,
            }
        )
    return tuple(events)


def retrieval_calls(row: dict[str, Any]) -> int:
    value = row.get("retrieval_calls")
    if value is not None:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            pass
    trace = row.get("raw_trace") or {}
    nodes = trace.get("nodes_execution") or []
    return sum(1 for node in nodes if isinstance(node, dict) and node.get("type") == "search")


def executed_reason_to_search(row: dict[str, Any]) -> bool:
    """Return whether a later executed search consumes a prior reason node."""
    trace = row.get("raw_trace") or {}
    reason_ids: set[str] = set()
    for node in trace.get("nodes_execution") or []:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "").strip().lower()
        node_id = str(node.get("id") or "").strip()
        if node_type == "reason" and node_id:
            reason_ids.add(node_id)
            continue
        if node_type != "search" or not reason_ids:
            continue
        raw_query = str(node.get("query") or "")
        if any(f"#{reason_id}#" in raw_query for reason_id in reason_ids):
            return True
    return False


def serialize_workflow(row: dict[str, Any], events: tuple[dict[str, Any], ...]) -> str:
    """Serialize policy actions and environment observations from one workflow."""
    trace = row.get("raw_trace") or {}
    lines = ["<successful_sibling_workflow>"]
    for event in events:
        role = event["role"]
        lines.extend(
            [
                f'<agent role="{role}" step="{event["workflow_step"]}">',
                event["response"],
                "</agent>",
            ]
        )
    for node in trace.get("nodes_execution") or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        node_type = str(node.get("type") or "")
        lines.append(f'<environment_node id="{node_id}" type="{node_type}">')
        query = str(node.get("executed_query") or node.get("resolved_query") or node.get("query") or "").strip()
        context = str(node.get("search_context") or "").strip()
        conclusion = str(node.get("conclusion") or "").strip()
        if query:
            lines.extend(["<query>", query, "</query>"])
        if context:
            lines.extend(["<information>", context, "</information>"])
        if conclusion:
            lines.extend(["<conclusion>", conclusion, "</conclusion>"])
        lines.append("</environment_node>")
    final_answer = str(trace.get("final_answer") or row.get("pred_answer") or "").strip()
    if final_answer:
        lines.extend(["<workflow_answer>", final_answer, "</workflow_answer>"])
    lines.append("</successful_sibling_workflow>")
    return "\n".join(lines)


def read_workflows(
    paths: Iterable[str],
    allowed_roles: set[str],
    max_reference_chars: int,
    rejected: Counter[str],
    require_grounded_by_nodes: bool = False,
    require_no_plan_warnings: bool = False,
    require_complete_nodes: bool = False,
) -> list[Workflow]:
    workflows = []
    for raw_path in paths:
        path = Path(raw_path)
        for position, row in enumerate(load_json(path)):
            if not isinstance(row, dict):
                rejected["non_object_row"] += 1
                continue
            question = str(row.get("question") or "").strip()
            if not question:
                rejected["missing_question"] += 1
                continue
            events = extract_events(row, allowed_roles)
            if not events:
                rejected["missing_policy_events"] += 1
                continue
            serialized = serialize_workflow(row, events)
            if max_reference_chars and len(serialized) > max_reference_chars:
                serialized = serialized[:max_reference_chars] + "\n[REFERENCE_TRUNCATED]"
            verification = (row.get("raw_trace") or {}).get("verification") or {}
            workflow_valid = not bool(
                row.get("error")
                or row.get("pipeline_error")
                or verification.get("direct_fallback_used")
            )
            if require_grounded_by_nodes and not bool(verification.get("grounded_by_nodes")):
                workflow_valid = False
                rejected["teacher_not_grounded_by_nodes"] += int(float(row.get("em") or 0.0) >= 1.0)
            if require_no_plan_warnings and bool((row.get("raw_trace") or {}).get("plan_warnings")):
                workflow_valid = False
                rejected["teacher_has_plan_warnings"] += int(float(row.get("em") or 0.0) >= 1.0)
            if require_complete_nodes:
                nodes = (row.get("raw_trace") or {}).get("nodes_execution") or []
                nodes_complete = bool(nodes)
                for node in nodes:
                    if not isinstance(node, dict):
                        nodes_complete = False
                        break
                    node_type = str(node.get("type") or "").strip().lower()
                    if node_type not in {"search", "reason"}:
                        nodes_complete = False
                        break
                    if not str(node.get("conclusion") or "").strip():
                        nodes_complete = False
                        break
                    if node_type == "search" and not (
                        str(
                            node.get("executed_query")
                            or node.get("resolved_query")
                            or node.get("query")
                            or ""
                        ).strip()
                        and str(node.get("search_context") or "").strip()
                    ):
                        nodes_complete = False
                        break
                if not nodes_complete:
                    workflow_valid = False
                    rejected["teacher_has_incomplete_nodes"] += int(
                        float(row.get("em") or 0.0) >= 1.0
                    )
            outcome = int(workflow_valid and float(row.get("em") or 0.0) >= 1.0)
            workflows.append(
                Workflow(
                    source_file=path.name,
                    sample_index=str(row.get("sample_index", position)),
                    rollout_index=int(row.get("rollout_index", position)),
                    question=question,
                    outcome=outcome,
                    retrieval_calls=retrieval_calls(row),
                    reason_to_search=executed_reason_to_search(row),
                    events=events,
                    serialized=serialized,
                )
            )
    return workflows


def workflow_key(workflow: Workflow) -> tuple[str, str, str]:
    return workflow.source_file, workflow.sample_index, workflow.question


def trajectory_distance(left: Workflow, right: Workflow) -> float:
    return 1.0 - SequenceMatcher(None, left.serialized, right.serialized).ratio()


def choose_pair(
    group: list[Workflow], reference_index: int = 0
) -> tuple[Workflow, Workflow] | None:
    successful = sorted(
        (workflow for workflow in group if workflow.outcome == 1),
        key=lambda workflow: workflow.rollout_index,
    )
    if not successful:
        return None
    reference = successful[reference_index % len(successful)]
    incorrect = [workflow for workflow in group if workflow.outcome == 0]
    candidates = [
        workflow
        for workflow in incorrect
        if workflow.serialized != reference.serialized
    ]
    if not candidates:
        return None
    student = max(
        candidates,
        key=lambda workflow: (
            trajectory_distance(reference, workflow),
            len(workflow.serialized),
            -workflow.rollout_index,
        ),
    )
    return reference, student


def choose_successful_reference(
    group: list[Workflow], reference_index: int = 0
) -> Workflow | None:
    successful = sorted(
        (workflow for workflow in group if workflow.outcome == 1),
        key=lambda workflow: workflow.rollout_index,
    )
    if not successful:
        return None
    return successful[reference_index % len(successful)]


def redact_current_action(serialized: str, role: str, workflow_step: int) -> str:
    marker = f'<agent role="{role}" step="{workflow_step}">\n'
    start = serialized.find(marker)
    if start < 0:
        raise ValueError(
            f"Cannot redact missing action block role={role} step={workflow_step}"
        )
    content_start = start + len(marker)
    content_end = serialized.find("\n</agent>", content_start)
    if content_end < 0:
        raise ValueError(
            f"Cannot redact unterminated action block role={role} step={workflow_step}"
        )
    return (
        serialized[:content_start]
        + "[CURRENT_ACTION_REDACTED]"
        + serialized[content_end:]
    )


def stable_group_id(key: tuple[str, str, str]) -> str:
    return hashlib.sha1("\x1f".join(key).encode("utf-8")).hexdigest()[:20]


def build_examples(
    workflows: list[Workflow],
    rejected: Counter[str],
    mode: str = "failed_sibling",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Workflow]] = defaultdict(list)
    for workflow in workflows:
        grouped[workflow_key(workflow)].append(workflow)
    examples = []
    for key, raw_group in sorted(grouped.items()):
        unique: dict[tuple[int, str], Workflow] = {}
        for workflow in raw_group:
            unique.setdefault((workflow.outcome, workflow.serialized), workflow)
        group = list(unique.values())
        minimum_group_size = 1 if mode == "aligned_success" else 2
        if len(group) < minimum_group_size:
            rejected["group_too_small_after_dedup"] += 1
            continue
        reference_index = int(stable_group_id(key), 16)
        if mode == "aligned_success":
            reference = choose_successful_reference(group, reference_index=reference_index)
            if reference is None:
                rejected["no_correct_reference"] += 1
                continue
            group_id = stable_group_id(key)
            for event in reference.events:
                privileged_reference = redact_current_action(
                    reference.serialized,
                    str(event["role"]),
                    int(event["workflow_step"]),
                )
                if str(event["response"]).strip() in {
                    "",
                    "[CURRENT_ACTION_REDACTED]",
                }:
                    rejected["empty_success_action"] += 1
                    continue
                examples.append(
                    {
                        "schema_version": 3,
                        "group_id": group_id,
                        "source_file": reference.source_file,
                        "sample_index": reference.sample_index,
                        "question": reference.question,
                        "role": event["role"],
                        "workflow_step": event["workflow_step"],
                        "student_outcome": 1,
                        "target_source": "successful_on_policy_action",
                        "student_rollout_index": reference.rollout_index,
                        "reference_rollout_index": reference.rollout_index,
                        "reference_retrieval_calls": reference.retrieval_calls,
                        "student_retrieval_calls": reference.retrieval_calls,
                        "reference_reason_to_search": reference.reason_to_search,
                        "student_reason_to_search": reference.reason_to_search,
                        "trajectory_distance": 0.0,
                        "student_messages": event["messages"],
                        "student_response": event["response"],
                        "privileged_reference": privileged_reference,
                        "current_action_redacted": True,
                    }
                )
            continue

        pair = choose_pair(group, reference_index=reference_index)
        if pair is None:
            if not any(workflow.outcome == 1 for workflow in group):
                rejected["no_correct_reference"] += 1
            else:
                rejected["no_incorrect_student"] += 1
            continue
        reference, student = pair
        group_id = stable_group_id(key)
        distance = trajectory_distance(reference, student)
        for event in student.events:
            examples.append(
                {
                    "schema_version": 2,
                    "group_id": group_id,
                    "source_file": student.source_file,
                    "sample_index": student.sample_index,
                    "question": student.question,
                    "role": event["role"],
                    "workflow_step": event["workflow_step"],
                    "student_outcome": student.outcome,
                    "student_rollout_index": student.rollout_index,
                    "reference_rollout_index": reference.rollout_index,
                    "reference_retrieval_calls": reference.retrieval_calls,
                    "student_retrieval_calls": student.retrieval_calls,
                    "reference_reason_to_search": reference.reason_to_search,
                    "student_reason_to_search": student.reason_to_search,
                    "trajectory_distance": distance,
                    "student_messages": event["messages"],
                    "student_response": event["response"],
                    "privileged_reference": reference.serialized,
                }
            )
    return examples


def split_by_group(
    examples: list[dict[str, Any]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_ids = sorted({row["group_id"] for row in examples})
    random.Random(seed).shuffle(group_ids)
    val_count = round(len(group_ids) * val_ratio)
    if val_ratio > 0 and len(group_ids) > 1:
        val_count = min(max(1, val_count), len(group_ids) - 1)
    val_ids = set(group_ids[:val_count])
    train = [row for row in examples if row["group_id"] not in val_ids]
    val = [row for row in examples if row["group_id"] in val_ids]
    return train, val


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rejected: Counter[str] = Counter()
    workflows = read_workflows(
        args.rollout_log,
        set(args.roles),
        args.max_reference_chars,
        rejected,
        require_grounded_by_nodes=args.require_grounded_by_nodes,
        require_no_plan_warnings=args.require_no_plan_warnings,
        require_complete_nodes=args.require_complete_nodes,
    )
    examples = build_examples(workflows, rejected, mode=args.mode)
    if not examples:
        raise RuntimeError(f"No Search-E1 pairs were built; rejected={dict(rejected)}")
    train, val = split_by_group(examples, args.val_ratio, args.seed)
    write_jsonl(args.output_dir / "opsd_train.jsonl", train)
    write_jsonl(args.output_dir / "opsd_val.jsonl", val)
    summary = {
        "schema_version": 3 if args.mode == "aligned_success" else 2,
        "method": (
            "action_aligned_hindsight_opsd"
            if args.mode == "aligned_success"
            else "search_e1_style_sibling_opsd"
        ),
        "mode": args.mode,
        "rollout_logs": args.rollout_log,
        "workflow_count": len(workflows),
        "paired_question_count": len({row["group_id"] for row in examples}),
        "example_count": len(examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "train_groups": len({row["group_id"] for row in train}),
        "val_groups": len({row["group_id"] for row in val}),
        "role_counts": dict(Counter(row["role"] for row in examples)),
        "student_outcome_counts": dict(Counter(str(row["student_outcome"]) for row in examples)),
        "rejected": dict(rejected),
        "leakage_contract": {
            "gold_answer_field_used_as_teacher_text": False,
            "gold_document_metadata_used": False,
            "successful_on_policy_workflow_is_privileged_context": True,
            "current_target_action_redacted_from_privileged_context": (
                args.mode == "aligned_success"
            ),
        },
        "teacher_gate": {
            "require_grounded_by_nodes": args.require_grounded_by_nodes,
            "require_no_plan_warnings": args.require_no_plan_warnings,
            "require_complete_nodes": args.require_complete_nodes,
        },
    }
    with (args.output_dir / "pair_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] train={args.output_dir / 'opsd_train.jsonl'}")
    print(f"[DONE] val={args.output_dir / 'opsd_val.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
