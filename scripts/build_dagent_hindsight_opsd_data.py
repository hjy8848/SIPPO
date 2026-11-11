#!/usr/bin/env python3
"""Build outcome-conditioned planner distillation data from repeated rollouts.

The teacher sees only sibling plans and binary outcome labels. Retrieved
documents, node conclusions, final answers, and gold answers are deliberately
excluded so hindsight cannot leak the target answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


PLAN_RE = re.compile(r"<plan>.*?</plan>", re.IGNORECASE | re.DOTALL)
NODE_RE = re.compile(
    r"<(search|reason)\s+id=[\"'](\d+)[\"']\s*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Rollout:
    source_kind: str
    policy_role: str
    source_file: str
    sample_index: str
    rollout_index: int
    question: str
    outcome: int
    messages: list[dict[str, str]]
    response: str
    plan: str
    target_topology: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-log", action="append", default=[])
    parser.add_argument("--reason-log", action="append", default=[])
    parser.add_argument("--reason-reference-log", action="append", default=[])
    parser.add_argument("--reason-trajectory-file", action="append", default=[])
    parser.add_argument("--reason-reference-parquet", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-incorrect-per-group", type=int, default=2)
    parser.add_argument("--max-hindsight-rollouts", type=int, default=6)
    parser.add_argument("--max-hindsight-chars", type=int, default=6000)
    args = parser.parse_args()
    if not any(
        (
            args.standard_log,
            args.reason_log,
            args.reason_reference_log,
            args.reason_trajectory_file,
            args.reason_reference_parquet,
        )
    ):
        parser.error("at least one rollout log is required")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    return args


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def clean_plan(text: Any) -> str:
    value = str(text or "").strip()
    match = PLAN_RE.search(value)
    if not match:
        return ""
    return re.sub(r"[ \t]+\n", "\n", match.group(0).strip())


def has_reason_search_topology(plan: str) -> bool:
    nodes = [(kind.lower(), int(node_id), body.strip()) for kind, node_id, body in NODE_RE.findall(plan)]
    for index, (kind, reason_id, _) in enumerate(nodes):
        if kind != "reason" or not any(previous[0] == "search" for previous in nodes[:index]):
            continue
        if any(later[0] == "search" and f"#{reason_id}#" in later[2] for later in nodes[index + 1 :]):
            return True
    return False


def extract_planner_event(row: dict[str, Any]) -> tuple[list[dict[str, str]], str] | None:
    trace = row.get("raw_trace") or {}
    events = trace.get("sft_events") or row.get("sft_events") or []
    for event in events:
        if event.get("agent_name") != "PlanningAgent":
            continue
        messages = event.get("messages") or []
        response = str(event.get("response") or "")
        if isinstance(messages, list) and messages and response:
            normalized = [
                {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
                for message in messages
                if isinstance(message, dict)
            ]
            if normalized:
                return normalized, response
    response = str(trace.get("plan_output") or trace.get("raw_plan_output") or "")
    if not response:
        response = str(row.get("plan_output") or row.get("raw_plan_output") or "")
    return None if not response else ([], response)


def infer_question(row: dict[str, Any], messages: list[dict[str, str]]) -> str:
    question = str(row.get("question") or "").strip()
    if question:
        return question
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "").strip()
    return ""


def read_rollouts(
    paths: Iterable[str], source_kind: str, policy_role: str, rejected: Counter[str]
) -> list[Rollout]:
    rollouts: list[Rollout] = []
    for raw_path in paths:
        path = Path(raw_path)
        for position, row in enumerate(load_json(path)):
            event = extract_planner_event(row)
            if event is None:
                rejected["missing_planner_event"] += 1
                continue
            messages, response = event
            if not messages:
                rejected["missing_exact_planner_messages"] += 1
                continue
            plan = clean_plan(response)
            if not plan:
                rejected["unparseable_plan"] += 1
                continue
            question = infer_question(row, messages)
            if not question:
                rejected["missing_question"] += 1
                continue
            if row.get("error") or row.get("pipeline_error"):
                outcome = 0
            else:
                outcome = int(float(row.get("em", 0.0)) >= 1.0)
            rollouts.append(
                Rollout(
                    source_kind=source_kind,
                    policy_role=policy_role,
                    source_file=path.name,
                    sample_index=str(row.get("sample_index", position)),
                    rollout_index=int(row.get("rollout_index", position)),
                    question=question,
                    outcome=outcome,
                    messages=messages,
                    response=response,
                    plan=plan,
                    target_topology=has_reason_search_topology(plan),
                )
            )
    return rollouts


def read_verified_reason_trajectories(
    paths: Iterable[str], rejected: Counter[str]
) -> list[Rollout]:
    """Load accepted Wiki18 trajectories as positive planner references."""
    rollouts: list[Rollout] = []
    for raw_path in paths:
        path = Path(raw_path)
        for position, row in enumerate(load_jsonl(path)):
            event = extract_planner_event(row)
            if event is None:
                rejected["trajectory_missing_planner_event"] += 1
                continue
            messages, response = event
            plan = clean_plan(response)
            question = infer_question(row, messages)
            if not messages or not plan or not question:
                rejected["trajectory_unusable_planner_event"] += 1
                continue
            if not has_reason_search_topology(plan):
                rejected["trajectory_missing_reason_search_topology"] += 1
                continue
            rollouts.append(
                Rollout(
                    source_kind="reason",
                    policy_role="reference",
                    source_file=path.name,
                    sample_index=str(row.get("trajectory_id", row.get("sample_index", position))),
                    rollout_index=position,
                    question=question,
                    outcome=1,
                    messages=messages,
                    response=response,
                    plan=plan,
                    target_topology=True,
                )
            )
    return rollouts


def verified_reference_from_parquet_row(
    row: dict[str, Any], position: int, source_file: str, rejected: Counter[str]
) -> Rollout | None:
    extra = row.get("extra_info") or {}
    question = str(extra.get("question") or "").strip()
    plan = clean_plan(extra.get("gold_plan"))
    if not question or not plan:
        rejected["parquet_reference_missing_question_or_plan"] += 1
        return None
    if not has_reason_search_topology(plan):
        rejected["parquet_reference_missing_reason_search_topology"] += 1
        return None
    return Rollout(
        source_kind="reason",
        policy_role="reference",
        source_file=source_file,
        sample_index=str(extra.get("id", position)),
        rollout_index=position,
        question=question,
        outcome=1,
        messages=[],
        response=plan,
        plan=plan,
        target_topology=True,
    )


def read_verified_reason_parquets(
    paths: Iterable[str], rejected: Counter[str]
) -> list[Rollout]:
    import pandas as pd

    rollouts: list[Rollout] = []
    for raw_path in paths:
        path = Path(raw_path)
        frame = pd.read_parquet(path, columns=["extra_info"])
        for position, row in enumerate(frame.to_dict(orient="records")):
            reference = verified_reference_from_parquet_row(row, position, path.name, rejected)
            if reference is not None:
                rollouts.append(reference)
    return rollouts


def group_key(rollout: Rollout) -> tuple[str, str, str]:
    return rollout.source_kind, rollout.sample_index, rollout.question


def deduplicate(group: list[Rollout]) -> list[Rollout]:
    seen: set[tuple[str, int, str]] = set()
    output = []
    for rollout in sorted(group, key=lambda item: (item.rollout_index, item.plan)):
        key = (rollout.plan, rollout.outcome, rollout.policy_role)
        if key not in seen:
            seen.add(key)
            output.append(rollout)
    return output


def plan_distance(left: str, right: str) -> float:
    return 1.0 - SequenceMatcher(None, left, right).ratio()


def choose_references(group: list[Rollout], source_kind: str) -> list[Rollout]:
    successful = [item for item in group if item.outcome == 1]
    if source_kind == "reason":
        successful = [item for item in successful if item.target_topology]
        external = [item for item in successful if item.policy_role == "reference"]
        if external:
            successful = external
    return sorted(successful, key=lambda item: (len(item.plan), item.rollout_index))


def choose_students(
    group: list[Rollout], references: list[Rollout], source_kind: str, limit: int
) -> list[Rollout]:
    failed = [item for item in group if item.outcome == 0 and item.policy_role == "candidate"]
    if source_kind == "reason":
        failed = [item for item in failed if not item.target_topology]
    if not references or not failed:
        return []
    reference = references[0]
    return sorted(
        failed,
        key=lambda item: (plan_distance(item.plan, reference.plan), -item.rollout_index),
        reverse=True,
    )[:limit]


def balanced_hindsight(
    group: list[Rollout], focal: Rollout, references: list[Rollout], limit: int
) -> list[Rollout]:
    selected: list[Rollout] = []
    for item in references:
        if item.plan != focal.plan and item not in selected:
            selected.append(item)
        if len(selected) >= max(1, limit // 2):
            break
    for item in group:
        if item.plan == focal.plan or item in selected:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def hindsight_block(siblings: list[Rollout], labels: list[int], max_chars: int) -> str:
    lines = [
        "<workflow_hindsight>",
        "Sibling plans below were sampled for the same question. The outcome label only says whether the complete workflow's final answer was exactly correct.",
        "Use the successful and failed alternatives to refine the distribution of the current plan. Do not mention this hindsight in the response.",
    ]
    for index, (rollout, label) in enumerate(zip(siblings, labels), start=1):
        lines.extend(
            [
                f"[Sibling {index}] [Outcome: {'Correct' if label else 'Incorrect'}]",
                rollout.plan,
            ]
        )
    lines.append("</workflow_hindsight>")
    block = "\n".join(lines)
    if len(block) <= max_chars:
        return block
    header = "\n".join(lines[:3]) + "\n"
    return header + block[len(header) : max_chars - len("\n</workflow_hindsight>")] + "\n</workflow_hindsight>"


def add_hindsight(messages: list[dict[str, str]], block: str) -> list[dict[str, str]]:
    output = [dict(message) for message in messages]
    for index, message in enumerate(output):
        if message.get("role") == "system":
            output[index] = {"role": "system", "content": message.get("content", "") + "\n\n" + block}
            return output
    return [{"role": "system", "content": block}, *output]


def reference_teacher_messages(
    messages: list[dict[str, str]], reference_plan: str, outcome_label: str
) -> list[dict[str, str]]:
    """Condition the frozen teacher with one same-question planner demonstration."""
    output = [dict(message) for message in messages]
    instruction = (
        "A prior planner response for this same question and its workflow outcome are provided as a demonstration. "
        "Treat a Correct response as a positive policy reference and an Incorrect response as a negative reference. "
        "The current planner attempt eventually produced an incorrect final answer. Improve the current plan policy, "
        "but output only the plan requested by the original prompt and never mention the demonstration or outcome."
    )
    for index, message in enumerate(output):
        if message.get("role") == "system":
            output[index] = {
                "role": "system",
                "content": message.get("content", "") + "\n\n" + instruction,
            }
            break
    else:
        output.insert(0, {"role": "system", "content": instruction})
    final_user_index = max(
        (index for index, message in enumerate(output) if message.get("role") == "user"),
        default=len(output),
    )
    output[final_user_index:final_user_index] = [
        {
            "role": "user",
            "content": f"Prior planner attempt for the same question. Workflow outcome: {outcome_label}.",
        },
        {"role": "assistant", "content": reference_plan},
    ]
    return output


def shuffled_labels(labels: list[int], rng: random.Random) -> list[int]:
    shuffled = list(labels)
    if len(set(shuffled)) > 1:
        for _ in range(10):
            rng.shuffle(shuffled)
            if shuffled != labels:
                break
    else:
        shuffled = [1 - value for value in shuffled]
    return shuffled


def stable_group_id(key: tuple[str, str, str]) -> str:
    return hashlib.sha1("\x1f".join(key).encode("utf-8")).hexdigest()[:20]


def build_examples(
    rollouts: list[Rollout], args: argparse.Namespace, rejected: Counter[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        grouped[group_key(rollout)].append(rollout)
    rng = random.Random(args.seed)
    examples: list[dict[str, Any]] = []
    for key, raw_group in sorted(grouped.items()):
        group = deduplicate(raw_group)
        source_kind = key[0]
        if len(group) < 2:
            rejected[f"{source_kind}_group_too_small"] += 1
            continue
        references = choose_references(group, source_kind)
        if not references:
            rejected[f"{source_kind}_no_successful_reference"] += 1
            continue
        students = choose_students(
            group,
            references,
            source_kind,
            args.max_incorrect_per_group,
        )
        if not students:
            rejected[f"{source_kind}_no_eligible_failed_student"] += 1
            continue
        group_id = stable_group_id(key)
        for focal in students:
            siblings = balanced_hindsight(
                group,
                focal,
                references,
                args.max_hindsight_rollouts,
            )
            labels = [item.outcome for item in siblings]
            positive_reference = references[0]
            negative_candidates = [
                item for item in group if item.outcome == 0 and item.plan != focal.plan
            ]
            negative_reference = negative_candidates[0] if negative_candidates else positive_reference
            examples.append(
                {
                    "schema_version": 1,
                    "group_id": group_id,
                    "source_kind": source_kind,
                    "source_file": focal.source_file,
                    "sample_index": focal.sample_index,
                    "rollout_index": focal.rollout_index,
                    "question": focal.question,
                    "outcome": focal.outcome,
                    "objective": "planner" if source_kind == "reason" else "search_query",
                    "student_messages": focal.messages,
                    "teacher_messages": reference_teacher_messages(
                        focal.messages, positive_reference.plan, "Correct"
                    ),
                    "control_teacher_messages": reference_teacher_messages(
                        focal.messages, negative_reference.plan, "Incorrect"
                    ),
                    "shuffled_teacher_messages": reference_teacher_messages(
                        focal.messages, negative_reference.plan, "Incorrect"
                    ),
                    "response": focal.response,
                    "focal_target_topology": focal.target_topology,
                    "successful_reference_target_topology": references[0].target_topology,
                    "positive_reference_response": positive_reference.plan,
                    "control_reference_response": negative_reference.plan,
                    "sibling_outcomes": labels,
                    "sibling_plans": [item.plan for item in siblings],
                }
            )
    return examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rejected: Counter[str] = Counter()
    rollouts = read_rollouts(args.standard_log, "standard", "candidate", rejected)
    rollouts.extend(read_rollouts(args.reason_log, "reason", "candidate", rejected))
    rollouts.extend(read_rollouts(args.reason_reference_log, "reason", "reference", rejected))
    rollouts.extend(read_verified_reason_trajectories(args.reason_trajectory_file, rejected))
    rollouts.extend(read_verified_reason_parquets(args.reason_reference_parquet, rejected))
    examples = build_examples(rollouts, args, rejected)
    if not examples:
        raise RuntimeError(f"No distillation examples were built. Rejections: {dict(rejected)}")

    group_ids = sorted({row["group_id"] for row in examples})
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)
    val_groups = set(group_ids[: round(len(group_ids) * args.val_ratio)])
    train = [row for row in examples if row["group_id"] not in val_groups]
    val = [row for row in examples if row["group_id"] in val_groups]
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)

    report = {
        "rollouts_loaded": len(rollouts),
        "groups_loaded": len({group_key(item) for item in rollouts}),
        "examples": len(examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "source_counts": Counter(row["source_kind"] for row in examples),
        "objective_counts": Counter(row["objective"] for row in examples),
        "rejections": rejected,
        "leakage_policy": {
            "gold_answer_in_teacher": False,
            "retrieved_documents_in_teacher": False,
            "node_conclusions_in_teacher": False,
            "final_answer_in_teacher": False,
            "teacher_inputs": "planner prompt plus sibling plan XML and Correct/Incorrect labels only",
        },
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
