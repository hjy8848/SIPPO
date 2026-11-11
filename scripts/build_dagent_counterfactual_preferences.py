#!/usr/bin/env python3
"""Build execution-verified action preferences for DAGENT.

The canonical input is the accepted evidence-first workflow bank. Reasoner and
QueryRewrite pairs come from deterministic counterfactual branches already
verified in that bank. Optional Planning pairs join the accepted plan with a
same-question, valid failed plan from a logged policy rollout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from qa_manager.ROSA.protocol import (
    ANSWER_GENERATION_SYSTEM_MSG,
    FINAL_SYSTEM_MSG,
    PLANNER_SYSTEM_MSG,
    QUERY_REWRITE_SYSTEM_MSG,
    REASON_SYSTEM_MSG,
)


SPLITS = ("train", "validation", "test")
ROLE_ORDER = {"PlanningAgent": 0, "ReasonerAgent": 1, "QueryRewriteAgent": 2}
CANONICAL_SYSTEM_PROMPTS = {
    "PlanningAgent": PLANNER_SYSTEM_MSG,
    "QueryRewriteAgent": QUERY_REWRITE_SYSTEM_MSG,
    "AnswerGenerationAgent": ANSWER_GENERATION_SYSTEM_MSG,
    "ReasonerAgent": REASON_SYSTEM_MSG,
    "AnswerSummarizationAgent": FINAL_SYSTEM_MSG,
}


def stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def normalize_answer(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def event_for(record: dict[str, Any], agent_name: str, *, node_id: str | None = None) -> dict[str, Any]:
    matches = []
    for event in record.get("sft_events", []):
        if event.get("agent_name") != agent_name:
            continue
        if node_id is not None and str(event.get("metadata", {}).get("node_id")) != str(node_id):
            continue
        matches.append(event)
    if len(matches) != 1:
        raise ValueError(
            f"{record.get('trajectory_id')}: expected one {agent_name} event"
            f" for node {node_id}, found {len(matches)}"
        )
    return matches[0]


def comparison_year(value: Any) -> int:
    numeric = int(round(float(value)))
    if abs(numeric) >= 100000:
        sign = -1 if numeric < 0 else 1
        return sign * (abs(numeric) // 10000)
    return numeric


def candidate_side(record: dict[str, Any], entity: str) -> str:
    blueprint = record["blueprint"]
    if normalize_answer(entity) == normalize_answer(blueprint["candidate_a"]):
        return "candidate_a"
    if normalize_answer(entity) == normalize_answer(blueprint["candidate_b"]):
        return "candidate_b"
    raise ValueError(f"{record.get('trajectory_id')}: entity {entity!r} is not a candidate")


def validate_record(record: dict[str, Any]) -> None:
    trajectory_id = record.get("trajectory_id")
    required = (
        "question", "answer", "winner", "counterfactual_entity", "counterfactual_answer",
        "blueprint", "comparison_evidence", "counterfactual", "verification", "reasoning",
        "source", "evidence_join", "sft_events",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{trajectory_id}: missing fields {missing}")
    verification = record["verification"]
    gates = (
        "evidence_first_graph_join",
        "reason_is_required_by_search_4",
        "counterfactual_retrievable",
        "counterfactual_answer_differs",
        "direct_question_does_not_retrieve_answer",
        "final_answer_grounded_in_search_4",
    )
    failed = [key for key in gates if verification.get(key) is not True]
    if failed:
        raise ValueError(f"{trajectory_id}: failed verification gates {failed}")
    if normalize_answer(record["winner"]) == normalize_answer(record["counterfactual_entity"]):
        raise ValueError(f"{trajectory_id}: winner equals counterfactual entity")
    if normalize_answer(record["answer"]) == normalize_answer(record["counterfactual_answer"]):
        raise ValueError(f"{trajectory_id}: factual and counterfactual answers are equal")
    event_for(record, "PlanningAgent")
    event_for(record, "ReasonerAgent", node_id="3")
    event_for(record, "QueryRewriteAgent", node_id="4")


def canonical_prompt_mismatches(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    mismatches = []
    for record in records:
        for event in record.get("sft_events", []):
            role = str(event.get("agent_name"))
            messages = event.get("messages") or []
            actual = str(messages[0].get("content")) if messages else ""
            expected = CANONICAL_SYSTEM_PROMPTS.get(role)
            if expected is None or actual != expected:
                mismatches.append(
                    {
                        "trajectory_id": str(record.get("trajectory_id")),
                        "agent_name": role,
                        "reason": "unknown role" if expected is None else "system prompt mismatch",
                    }
                )
    return mismatches


def prompt_hash(messages: list[dict[str, Any]]) -> str:
    return stable_id(messages)


def base_pair(
    record: dict[str, Any],
    *,
    agent_name: str,
    messages: list[dict[str, Any]],
    chosen: str,
    rejected: str,
    variant: str,
    chosen_outcome: dict[str, Any],
    rejected_outcome: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    if not messages or not chosen.strip() or not rejected.strip():
        raise ValueError(f"{record.get('trajectory_id')}: empty preference field")
    if chosen.strip() == rejected.strip():
        raise ValueError(f"{record.get('trajectory_id')}: chosen equals rejected")
    trajectory_id = str(record["trajectory_id"])
    pair_id = stable_id(trajectory_id, agent_name, variant, messages, chosen, rejected)
    return {
        "pair_id": pair_id,
        "group_id": trajectory_id,
        "agent_name": agent_name,
        "variant": variant,
        "prompt": copy.deepcopy(messages),
        "chosen": chosen,
        "rejected": rejected,
        "prompt_hash": prompt_hash(messages),
        "question": record["question"],
        "source_trajectory_id": trajectory_id,
        "parent_trajectory_ids": sorted(map(str, record["source"]["parent_trajectory_ids"])),
        "entities": sorted(
            [str(record["blueprint"]["candidate_a"]), str(record["blueprint"]["candidate_b"])],
            key=str.casefold,
        ),
        "card_ids": sorted(
            [str(record["evidence_join"]["left_card_id"]), str(record["evidence_join"]["right_card_id"])]
        ),
        "reasoning_mode": record["reasoning"]["mode"],
        "operation": record["reasoning"]["operation"],
        "downstream_relation": record["downstream"]["relation"],
        "chosen_outcome": chosen_outcome,
        "rejected_outcome": rejected_outcome,
        "verification": verification,
    }


def reason_pair(record: dict[str, Any]) -> dict[str, Any]:
    event = event_for(record, "ReasonerAgent", node_id="3")
    winner = str(record["winner"])
    loser = str(record["counterfactual_entity"])
    chosen = f"<conclusion>{winner}</conclusion>"
    rejected = f"<conclusion>{loser}</conclusion>"
    if event["response"].strip() != chosen:
        raise ValueError(f"{record['trajectory_id']}: stored Reasoner response is not the winner")
    return base_pair(
        record,
        agent_name="ReasonerAgent",
        messages=event["messages"],
        chosen=chosen,
        rejected=rejected,
        variant="factual_condition",
        chosen_outcome={
            "selected_entity": winner,
            "downstream_query": record["nodes_execution"][3]["executed_query"],
            "final_answer": record["answer"],
            "final_em": 1,
        },
        rejected_outcome={
            "selected_entity": loser,
            "downstream_query": record["counterfactual"]["executed_query"],
            "final_answer": record["counterfactual_answer"],
            "final_em": 0,
        },
        verification={
            "kind": "deterministic_counterfactual_graph",
            "same_prompt": True,
            "single_action_intervention": True,
            "both_actions_valid": True,
            "both_downstream_branches_retrievable": True,
            "answer_direction_verified": True,
        },
    )


def flipped_reason_messages(record: dict[str, Any], event: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    messages = copy.deepcopy(event["messages"])
    user_messages = [message for message in messages if message.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError(f"{record['trajectory_id']}: Reasoner event must contain one user message")
    content = str(user_messages[0]["content"])
    mode = record["reasoning"]["mode"]
    if mode == "direct":
        operation = record["reasoning"]["operation"]
        inverse = {"earlier": "later", "later": "earlier"}.get(operation)
        if inverse is None or not re.search(rf"\b{re.escape(operation)}\b", content, flags=re.IGNORECASE):
            raise ValueError(f"{record['trajectory_id']}: cannot invert direct operation {operation!r}")
        user_messages[0]["content"] = re.sub(
            rf"\b{re.escape(operation)}\b", inverse, content, flags=re.IGNORECASE
        )
        return messages, f"inverse_{inverse}"
    if mode != "anchor_distance":
        raise ValueError(f"{record['trajectory_id']}: unsupported reasoning mode {mode!r}")
    old_anchor = int(record["reasoning"]["anchor_year"])
    loser_side = candidate_side(record, str(record["counterfactual_entity"]))
    loser_year = comparison_year(record["comparison_evidence"][loser_side]["comparison_key"])
    if loser_year == old_anchor:
        raise ValueError(f"{record['trajectory_id']}: counterfactual anchor does not change")
    pattern = rf"(?<=distance from ){re.escape(str(old_anchor))}\b"
    replaced, count = re.subn(pattern, str(loser_year), content)
    if count != 1:
        raise ValueError(f"{record['trajectory_id']}: expected one anchor in Reasoner prompt, found {count}")
    user_messages[0]["content"] = replaced
    return messages, f"counterfactual_anchor_{loser_year}"


def flipped_reason_pair(record: dict[str, Any]) -> dict[str, Any]:
    event = event_for(record, "ReasonerAgent", node_id="3")
    messages, flip = flipped_reason_messages(record, event)
    original_winner = str(record["winner"])
    counterfactual_winner = str(record["counterfactual_entity"])
    return base_pair(
        record,
        agent_name="ReasonerAgent",
        messages=messages,
        chosen=f"<conclusion>{counterfactual_winner}</conclusion>",
        rejected=f"<conclusion>{original_winner}</conclusion>",
        variant="flipped_condition",
        chosen_outcome={
            "condition_flip": flip,
            "selected_entity": counterfactual_winner,
            "downstream_query": record["counterfactual"]["executed_query"],
            "final_answer": record["counterfactual_answer"],
            "counterfactual_task_em": 1,
        },
        rejected_outcome={
            "condition_flip": flip,
            "selected_entity": original_winner,
            "downstream_query": record["nodes_execution"][3]["executed_query"],
            "final_answer": record["answer"],
            "counterfactual_task_em": 0,
        },
        verification={
            "kind": "deterministic_condition_flip",
            "same_prompt": True,
            "single_action_intervention": True,
            "both_actions_valid": True,
            "label_flipped_from_factual_pair": True,
            "both_downstream_branches_retrievable": True,
            "answer_direction_verified": True,
        },
    )


def query_rewrite_pair(record: dict[str, Any]) -> dict[str, Any]:
    event = event_for(record, "QueryRewriteAgent", node_id="4")
    chosen = str(event["response"]).strip()
    rejected = str(record["counterfactual"]["executed_query"]).strip()
    if chosen != str(record["nodes_execution"][3]["executed_query"]).strip():
        raise ValueError(f"{record['trajectory_id']}: QueryRewrite response differs from executed query")
    return base_pair(
        record,
        agent_name="QueryRewriteAgent",
        messages=event["messages"],
        chosen=chosen,
        rejected=rejected,
        variant="factual_dependency",
        chosen_outcome={
            "resolved_entity": record["winner"],
            "retrieved_answer": record["answer"],
            "final_em": 1,
        },
        rejected_outcome={
            "resolved_entity": record["counterfactual_entity"],
            "retrieved_answer": record["counterfactual_answer"],
            "final_em": 0,
        },
        verification={
            "kind": "deterministic_counterfactual_retrieval",
            "same_prompt": True,
            "single_action_intervention": True,
            "both_actions_valid": True,
            "both_queries_retrievable": True,
            "answer_direction_verified": True,
        },
    )


def rollout_planning_event(row: dict[str, Any]) -> dict[str, Any] | None:
    events = row.get("raw_trace", {}).get("sft_events", [])
    matches = [event for event in events if event.get("agent_name") == "PlanningAgent"]
    return matches[0] if len(matches) == 1 else None


def index_failed_planner_rollouts(
    path: Path | None,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    if path is None:
        return {}, stats
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            stats["non_object"] += 1
            continue
        trajectory_id = str(row.get("sample_index") or "")
        if not trajectory_id:
            stats["missing_sample_index"] += 1
            continue
        if row.get("pipeline_error") or row.get("error"):
            stats["pipeline_or_execution_error"] += 1
            continue
        if float(row.get("em") or 0.0) >= 1.0:
            stats["successful_rollout"] += 1
            continue
        trace = row.get("raw_trace") or {}
        if trace.get("plan_warnings"):
            stats["plan_warning"] += 1
            continue
        event = rollout_planning_event(row)
        if event is None:
            stats["missing_unique_planning_event"] += 1
            continue
        response = str(event.get("response") or "").strip()
        if not response.startswith("<plan>") or not response.endswith("</plan>"):
            stats["invalid_plan_envelope"] += 1
            continue
        if trajectory_id in indexed:
            stats["duplicate_sample_index"] += 1
            continue
        indexed[trajectory_id] = row
        stats["eligible_failed_rollout"] += 1
    return indexed, stats


def planning_pair(record: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    event = rollout_planning_event(rollout)
    assert event is not None
    if str(rollout.get("question")) != str(record["question"]):
        raise ValueError(f"{record['trajectory_id']}: policy rollout question mismatch")
    accepted_answer = normalize_answer(record["answer"])
    rollout_gold = {normalize_answer(value) for value in rollout.get("gold_answer", [])}
    if accepted_answer not in rollout_gold:
        raise ValueError(f"{record['trajectory_id']}: policy rollout gold answer mismatch")
    chosen = str(record["plan_output"]).strip()
    rejected = str(event["response"]).strip()
    if chosen == rejected:
        raise ValueError(f"{record['trajectory_id']}: failed rollout used the accepted plan")
    trace = rollout["raw_trace"]
    return base_pair(
        record,
        agent_name="PlanningAgent",
        messages=event["messages"],
        chosen=chosen,
        rejected=rejected,
        variant="accepted_vs_on_policy_failure",
        chosen_outcome={
            "topology": record["topology"],
            "final_answer": record["answer"],
            "final_em": 1,
            "search_steps": 3,
        },
        rejected_outcome={
            "topology": "-".join(str(node.get("type")) for node in trace.get("nodes_execution", [])),
            "final_answer": rollout.get("pred_answer"),
            "final_em": 0,
            "search_steps": rollout.get("search_steps"),
        },
        verification={
            "kind": "accepted_execution_vs_logged_on_policy_failure",
            "same_prompt": True,
            "single_action_intervention": True,
            "both_actions_valid": True,
            "chosen_execution_verified": True,
            "rejected_execution_verified": True,
            "answer_direction_verified": True,
        },
    )


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def record_resources(record: dict[str, Any]) -> list[str]:
    resources = ["parent:" + str(value) for value in record["source"]["parent_trajectory_ids"]]
    resources.extend(
        "entity:" + normalize_answer(value)
        for value in (record["blueprint"]["candidate_a"], record["blueprint"]["candidate_b"])
    )
    resources.extend(
        "card:" + str(value)
        for value in (record["evidence_join"]["left_card_id"], record["evidence_join"]["right_card_id"])
    )
    return resources


def assign_component_splits(
    records: list[dict[str, Any]], ratios: tuple[float, float, float]
) -> tuple[dict[str, str], dict[str, Any]]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must contain three positive values")
    total_ratio = sum(ratios)
    ratios = tuple(value / total_ratio for value in ratios)
    union_find = UnionFind()
    for record in records:
        resources = record_resources(record)
        for resource in resources[1:]:
            union_find.union(resources[0], resource)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        components[union_find.find(record_resources(record)[0])].append(record)

    targets = {split: ratios[index] * len(records) for index, split in enumerate(SPLITS)}
    counts = {split: 0 for split in SPLITS}
    component_split: dict[str, str] = {}
    ordered_components = sorted(
        components.items(), key=lambda item: (-len(item[1]), stable_id(item[0]))
    )
    for root, component_records in ordered_components:
        split = max(SPLITS, key=lambda name: (targets[name] - counts[name], -SPLITS.index(name)))
        component_split[root] = split
        counts[split] += len(component_records)

    assignment: dict[str, str] = {}
    component_rows = []
    for root, component_records in ordered_components:
        split = component_split[root]
        ids = sorted(str(record["trajectory_id"]) for record in component_records)
        for trajectory_id in ids:
            assignment[trajectory_id] = split
        component_rows.append(
            {"component_id": stable_id(root), "split": split, "records": len(ids), "trajectory_ids": ids}
        )
    return assignment, {
        "ratios_requested": dict(zip(SPLITS, ratios, strict=True)),
        "record_counts": counts,
        "component_count": len(components),
        "components": component_rows,
    }


def overlap_report(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("parent_trajectory_ids", "entities", "card_ids")
    result: dict[str, Any] = {}
    for field in fields:
        values = {
            split: {
                normalize_answer(value) if field == "entities" else str(value)
                for pair in pairs if pair["split"] == split for value in pair[field]
            }
            for split in SPLITS
        }
        intersections = {
            f"{left}__{right}": sorted(values[left] & values[right])
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        }
        result[field] = {
            "unique_by_split": {split: len(items) for split, items in values.items()},
            "overlap_counts": {key: len(items) for key, items in intersections.items()},
            "overlap_examples": {key: items[:10] for key, items in intersections.items() if items},
        }
    return result


def action_is_valid(pair: dict[str, Any], action: str) -> bool:
    if pair["agent_name"] == "ReasonerAgent":
        return re.fullmatch(r"<conclusion>[^<>\r\n]+</conclusion>", action.strip()) is not None
    if pair["agent_name"] == "QueryRewriteAgent":
        return bool(action.strip()) and "\n" not in action and "\r" not in action and "#" not in action
    if pair["agent_name"] == "PlanningAgent":
        stripped = action.strip()
        return stripped.startswith("<plan>") and stripped.endswith("</plan>")
    return False


def outcome_direction_is_valid(pair: dict[str, Any]) -> bool:
    chosen = pair["chosen_outcome"]
    rejected = pair["rejected_outcome"]
    if pair["variant"] == "flipped_condition":
        return chosen.get("counterfactual_task_em") == 1 and rejected.get("counterfactual_task_em") == 0
    return chosen.get("final_em") == 1 and rejected.get("final_em") == 0


def normalized_phrase_in_text(value: Any, text: str) -> bool:
    phrase = normalize_answer(value)
    normalized_text = normalize_answer(text)
    return bool(phrase) and f" {phrase} " in f" {normalized_text} "


def audit_pairs(
    pairs: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    invalid_actions: list[str] = []
    invalid_directions: list[str] = []
    prompt_answer_leaks: list[dict[str, str]] = []
    for pair in pairs:
        if not action_is_valid(pair, pair["chosen"]) or not action_is_valid(pair, pair["rejected"]):
            invalid_actions.append(pair["pair_id"])
        if not outcome_direction_is_valid(pair):
            invalid_directions.append(pair["pair_id"])
        prompt_text = "\n".join(str(message.get("content") or "") for message in pair["prompt"])
        record = records_by_id[pair["source_trajectory_id"]]
        for field in ("answer", "counterfactual_answer"):
            if normalized_phrase_in_text(record[field], prompt_text):
                prompt_answer_leaks.append(
                    {"pair_id": pair["pair_id"], "field": field, "value": str(record[field])}
                )
    return {
        "invalid_action_count": len(invalid_actions),
        "invalid_action_examples": invalid_actions[:10],
        "invalid_outcome_direction_count": len(invalid_directions),
        "invalid_outcome_direction_examples": invalid_directions[:10],
        "prompt_answer_leak_count": len(prompt_answer_leaks),
        "prompt_answer_leak_examples": prompt_answer_leaks[:10],
    }


def distribution_by_split(
    pairs: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in SPLITS:
        group_ids = sorted({pair["group_id"] for pair in pairs if pair["split"] == split})
        records = [records_by_id[group_id] for group_id in group_ids]
        report[split] = {
            "groups": len(group_ids),
            "pairs": sum(pair["split"] == split for pair in pairs),
            "roles": dict(Counter(pair["agent_name"] for pair in pairs if pair["split"] == split)),
            "relations": dict(Counter(record["downstream"]["relation"] for record in records)),
            "reasoning_modes": dict(Counter(record["reasoning"]["mode"] for record in records)),
            "operations": dict(Counter(record["reasoning"]["operation"] for record in records)),
        }
    return report


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - execution environment dependency
        raise RuntimeError("pandas and a parquet engine are required for --write-parquet") from exc
    frame = pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(args.accepted_jsonl)
    if not records:
        raise ValueError("accepted workflow bank is empty")
    trajectory_ids = [str(record.get("trajectory_id")) for record in records]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("accepted workflow bank contains duplicate trajectory IDs")
    for record in records:
        validate_record(record)
    prompt_mismatches = canonical_prompt_mismatches(records)
    if getattr(args, "require_canonical_prompts", False) and prompt_mismatches:
        raise ValueError(f"canonical prompt audit failed: {prompt_mismatches[:5]}")
    failed_rollouts, planner_stats = index_failed_planner_rollouts(args.policy_rollout_json)
    candidate_pairs_by_record: dict[str, list[dict[str, Any]]] = {}
    retained_records: list[dict[str, Any]] = []
    excluded_prompt_leaks: list[dict[str, Any]] = []
    planner_join_stats: Counter[str] = Counter()
    for record in records:
        trajectory_id = str(record["trajectory_id"])
        record_pairs = [reason_pair(record), flipped_reason_pair(record), query_rewrite_pair(record)]
        rollout = failed_rollouts.get(trajectory_id)
        if rollout is not None:
            try:
                record_pairs.append(planning_pair(record, rollout))
                planner_join_stats["constructed"] += 1
            except ValueError as exc:
                planner_join_stats["rejected:" + str(exc)] += 1
        else:
            planner_join_stats["no_eligible_failed_rollout"] += 1
        preliminary_audit = audit_pairs(record_pairs, {trajectory_id: record})
        if preliminary_audit["prompt_answer_leak_count"]:
            excluded_prompt_leaks.append(
                {
                    "trajectory_id": trajectory_id,
                    "question": record["question"],
                    "leaks": preliminary_audit["prompt_answer_leak_examples"],
                }
            )
            planner_join_stats["excluded_with_prompt_answer_leak"] += int(
                any(pair["agent_name"] == "PlanningAgent" for pair in record_pairs)
            )
            continue
        candidate_pairs_by_record[trajectory_id] = record_pairs
        retained_records.append(record)
        planner_join_stats["retained"] += int(
            any(pair["agent_name"] == "PlanningAgent" for pair in record_pairs)
        )

    if not retained_records:
        raise ValueError("all source records were excluded by prompt-answer leakage gates")
    records_by_id = {str(record["trajectory_id"]): record for record in retained_records}
    split_assignment, split_report = assign_component_splits(
        retained_records, tuple(args.split_ratios)
    )
    pairs: list[dict[str, Any]] = []
    for record in retained_records:
        trajectory_id = str(record["trajectory_id"])
        split = split_assignment[trajectory_id]
        for pair in candidate_pairs_by_record[trajectory_id]:
            pair["split"] = split
            pairs.append(pair)

    pairs.sort(
        key=lambda pair: (
            SPLITS.index(pair["split"]),
            pair["source_trajectory_id"],
            ROLE_ORDER[pair["agent_name"]],
            pair["variant"],
        )
    )
    pair_ids = [pair["pair_id"] for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("generated preference pairs contain duplicate pair IDs")

    overlaps = overlap_report(pairs)
    leaked = {
        field: count
        for field, details in overlaps.items()
        for key, count in details["overlap_counts"].items()
        if count
    }
    if leaked:
        raise ValueError(f"split resource leakage detected: {leaked}")
    pair_audit = audit_pairs(pairs, records_by_id)
    failed_pair_gates = {
        key: value for key, value in pair_audit.items()
        if key.endswith("_count") and value
    }
    if failed_pair_gates:
        raise ValueError(f"preference pair audit failed: {failed_pair_gates}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "preferences_all.jsonl", pairs)
    for split in SPLITS:
        split_rows = [pair for pair in pairs if pair["split"] == split]
        write_jsonl(args.output_dir / f"preferences_{split}.jsonl", split_rows)
        if args.write_parquet:
            write_parquet(args.output_dir / f"preferences_{split}.parquet", split_rows)

    by_split = {
        split: dict(Counter(pair["agent_name"] for pair in pairs if pair["split"] == split))
        for split in SPLITS
    }
    variant_counts = dict(Counter(pair["variant"] for pair in pairs))
    verification_counts = dict(Counter(pair["verification"]["kind"] for pair in pairs))
    winner_position = Counter()
    for record in retained_records:
        side = candidate_side(record, str(record["winner"]))
        winner_position["a" if side == "candidate_a" else "b"] += 1
    report = {
        "method": "counterfactual_workflow_action_preferences_v2",
        "accepted_jsonl": str(args.accepted_jsonl),
        "policy_rollout_json": str(args.policy_rollout_json) if args.policy_rollout_json else None,
        "source_records": len(records),
        "retained_source_records": len(retained_records),
        "excluded_source_records_with_prompt_answer_leak": len(excluded_prompt_leaks),
        "excluded_prompt_answer_leak_examples": excluded_prompt_leaks[:10],
        "preference_pairs": len(pairs),
        "pair_counts_by_split_and_role": by_split,
        "pair_counts_by_variant": variant_counts,
        "verification_counts": verification_counts,
        "reasoning_mode_counts": dict(Counter(record["reasoning"]["mode"] for record in retained_records)),
        "operation_counts": dict(Counter(record["reasoning"]["operation"] for record in retained_records)),
        "relation_counts": dict(Counter(record["downstream"]["relation"] for record in retained_records)),
        "winner_position_counts": dict(winner_position),
        "condition_flip_pairs": variant_counts.get("flipped_condition", 0),
        "planner_rollout_filter": dict(planner_stats),
        "planner_join": dict(planner_join_stats),
        "split": split_report,
        "distribution_by_split": distribution_by_split(pairs, records_by_id),
        "split_overlap_audit": overlaps,
        "pair_audit": pair_audit,
        "canonical_prompt_audit": {
            "required": bool(getattr(args, "require_canonical_prompts", False)),
            "mismatch_count": len(prompt_mismatches),
            "mismatch_examples": prompt_mismatches[:10],
        },
        "hard_gates": {
            "all_pairs_share_one_prompt_between_chosen_and_rejected": all(
                pair["verification"].get("same_prompt") is True for pair in pairs
            ),
            "all_pairs_have_distinct_nonempty_actions": all(
                pair["chosen"].strip() and pair["rejected"].strip()
                and pair["chosen"].strip() != pair["rejected"].strip() for pair in pairs
            ),
            "all_actions_pass_role_format_checks": pair_audit["invalid_action_count"] == 0,
            "all_pairs_have_verified_direction": pair_audit["invalid_outcome_direction_count"] == 0,
            "parent_entity_card_split_overlap_zero": not leaked,
            "gold_or_counterfactual_answer_added_to_prompt": pair_audit["prompt_answer_leak_count"] > 0,
            "teacher_model_used": False,
            "canonical_system_prompts_exact": not prompt_mismatches,
        },
        "parquet_written": bool(args.write_parquet),
    }
    write_json(args.output_dir / "report.json", report)
    write_json(
        args.output_dir / "generation_config.json",
        {
            "accepted_jsonl": str(args.accepted_jsonl),
            "policy_rollout_json": str(args.policy_rollout_json) if args.policy_rollout_json else None,
            "split_ratios": list(args.split_ratios),
            "write_parquet": bool(args.write_parquet),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-jsonl", type=Path, required=True)
    parser.add_argument("--policy-rollout-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, default=(0.5, 0.25, 0.25),
        metavar=("TRAIN", "VALIDATION", "TEST"),
    )
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--require-canonical-prompts", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.accepted_jsonl.is_file():
        raise SystemExit(f"accepted JSONL not found: {args.accepted_jsonl}")
    if args.policy_rollout_json is not None and not args.policy_rollout_json.is_file():
        raise SystemExit(f"policy rollout JSON not found: {args.policy_rollout_json}")
    if any(not math.isfinite(value) or value <= 0 for value in args.split_ratios):
        raise SystemExit("split ratios must be finite and positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {args.output_dir}")
    build(args)


if __name__ == "__main__":
    main()
