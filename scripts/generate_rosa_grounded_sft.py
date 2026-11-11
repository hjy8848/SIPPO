#!/usr/bin/env python3
"""Collect execution-grounded ROSA trajectories and build multiturn SFT parquet files."""

import argparse
import json
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa_manager.normalize_answers import normalize_answer
from qa_manager.ROSA.protocol import (
    find_reference_ids,
    is_degenerate_text,
    is_invalid_conclusion,
    is_supported_answer,
    validate_plan_semantics,
    validate_plan_structure,
)


PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
NODE_RE = re.compile(r'<(search|reason) id="(\d+)">(.*?)</\1>', re.DOTALL | re.IGNORECASE)

ACTOR_AGENTS = {
    "PlanningAgent",
    "QueryRewriteAgent",
    "AnswerGenerationAgent",
    "ReasonerAgent",
    "AnswerSummarizationAgent",
}


def to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return to_python(value.as_py())
    if hasattr(value, "tolist"):
        return to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_python(item) for item in value]
    return value


def ensure_string_list(value: Any) -> list[str]:
    value = to_python(value)
    if value is None:
        return []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(ensure_string_list(item))
        return result
    text = str(value).strip()
    return [text] if text else []


def extract_question_and_answers(row: pd.Series) -> tuple[str, list[str]]:
    extra_info = to_python(row.get("extra_info", {}))
    reward_model = to_python(row.get("reward_model", {}))
    question = str(extra_info.get("question", "")).strip() if isinstance(extra_info, dict) else ""
    if not question and isinstance(reward_model, dict):
        question = str(reward_model.get("question", "")).strip()

    answers: list[str] = []
    if isinstance(extra_info, dict):
        answers = ensure_string_list(extra_info.get("answer"))
    if not answers and isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            answers = ensure_string_list(ground_truth.get("target"))
    if not question or not answers:
        raise ValueError("Every source row must contain a question and at least one gold answer.")
    return question, answers


def full_xml_value(tag: str, text: Any) -> str:
    match = re.fullmatch(
        rf"\s*<{tag}>(.*?)</{tag}>\s*",
        str(text or ""),
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def normalized_contains(container: str, value: str) -> bool:
    needle = normalize_answer(str(value))
    haystack = normalize_answer(str(container))
    return bool(needle) and needle in haystack


def validate_dag(nodes: list[tuple[str, str, str]]) -> list[str]:
    errors: list[str] = []
    ids = [int(node_id) for _, node_id, _ in nodes]
    if ids != list(range(1, len(nodes) + 1)):
        errors.append("node IDs are not consecutive from 1")
    for _, node_id, text in nodes:
        current = int(node_id)
        for ref in find_reference_ids(text):
            if int(ref) >= current:
                errors.append(f"node {node_id} has a future or self reference #{ref}#")
    return errors


def validate_trace(
    question: str,
    gold_answers: list[str],
    trace: dict[str, Any],
    max_nodes: int,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    plan_output = str(trace.get("plan_output", ""))
    plan_blocks = PLAN_RE.findall(plan_output)
    if len(plan_blocks) != 1:
        reasons.append(f"expected exactly one plan block, got {len(plan_blocks)}")
        nodes: list[tuple[str, str, str]] = []
    else:
        nodes = [(kind.lower(), node_id, text.strip()) for kind, node_id, text in NODE_RE.findall(plan_blocks[0])]
        if not full_xml_value("plan", plan_output):
            reasons.append("planner output contains text outside the plan block")

    if trace.get("evaluation_mode") != "training_protocol":
        reasons.append("trace was not generated in training_protocol mode")
    if trace.get("execution_error"):
        reasons.append(f"execution error: {trace['execution_error']}")
    if trace.get("plan_warnings"):
        reasons.append("plan warnings were emitted")
    reasons.extend(validate_plan_structure(nodes, max_nodes=max_nodes))
    reasons.extend(validate_plan_semantics(question, nodes))
    reasons.extend(validate_dag(nodes))

    ablation = trace.get("ablation_config", {})
    if isinstance(ablation, dict):
        if ablation.get("disable_query_rewrite"):
            reasons.append("query rewrite was disabled")
        if ablation.get("disable_final_summarizer"):
            reasons.append("final summarizer was disabled")
        if ablation.get("linearized_execution"):
            reasons.append("linearized execution was enabled")

    node_logs = trace.get("nodes_execution", [])
    if not isinstance(node_logs, list):
        node_logs = []
    if len(node_logs) != len(nodes):
        reasons.append(f"executed {len(node_logs)} of {len(nodes)} plan nodes")

    for index, (node_type, node_id, _) in enumerate(nodes):
        if index >= len(node_logs):
            break
        log = node_logs[index]
        if str(log.get("id")) != node_id or str(log.get("type")) != node_type:
            reasons.append(f"execution log does not match plan node {node_id}")
            continue
        conclusion = str(log.get("conclusion", "")).strip()
        if is_invalid_conclusion(conclusion):
            reasons.append(f"node {node_id} has an unusable conclusion")
        if node_type == "search":
            query = str(log.get("executed_query", "")).strip()
            context = str(log.get("search_context", "")).strip()
            if not query or is_degenerate_text(query):
                reasons.append(f"search node {node_id} has no concrete executed query")
            if not context or context.startswith("[No Documents") or context.startswith("[Search API Error"):
                reasons.append(f"search node {node_id} has no usable retrieved context")
            if normalize_answer(conclusion) in {"yes", "no", "noanswer"}:
                reasons.append(f"search node {node_id} returned a decision instead of an evidence fact")
            elif conclusion and context and not normalized_contains(context, conclusion):
                reasons.append(f"search node {node_id} conclusion is not stated in its retrieved context")

    final_answer = str(trace.get("final_answer", "")).strip()
    if not final_answer or final_answer.startswith("Execution Failed"):
        reasons.append("missing final answer")
    elif not any(normalize_answer(final_answer) == normalize_answer(gold) for gold in gold_answers):
        reasons.append("final answer does not exact-match gold")
    conclusions = [str(log.get("conclusion", "")) for log in node_logs]
    if final_answer and not is_supported_answer(final_answer, conclusions):
        reasons.append("final answer is not grounded by a node conclusion")

    verification = trace.get("verification", {})
    if isinstance(verification, dict):
        if verification.get("direct_fallback_used"):
            reasons.append("direct fallback was used")
        if verification.get("grounded_by_nodes") is False:
            reasons.append("server grounding check failed")

    events = trace.get("sft_events", [])
    if not isinstance(events, list) or not events:
        reasons.append("trace has no exact SFT events; restart the updated rosa_server.py")

    diagnostics = {
        "plan_nodes": len(nodes),
        "search_nodes": sum(kind == "search" for kind, _, _ in nodes),
        "reason_nodes": sum(kind == "reason" for kind, _, _ in nodes),
        "final_answer": final_answer,
    }
    return list(dict.fromkeys(reasons)), diagnostics


def event_response_is_valid(event: dict[str, Any], final_answer: str) -> bool:
    agent = str(event.get("agent_name", ""))
    response = str(event.get("response", "")).strip()
    if agent not in ACTOR_AGENTS or not response:
        return False
    if agent == "PlanningAgent":
        return bool(full_xml_value("plan", response))
    if agent == "QueryRewriteAgent":
        return (
            not is_degenerate_text(response)
            and "<" not in response
            and ">" not in response
            and "\n" not in response
        )
    if agent == "AnswerGenerationAgent":
        conclusion = full_xml_value("conclusion", response)
        rewrite = full_xml_value("rewrite", response)
        return (
            bool(conclusion) and not is_invalid_conclusion(conclusion) and not rewrite
        ) or (
            bool(rewrite) and not is_degenerate_text(rewrite) and not conclusion
        )
    if agent == "ReasonerAgent":
        conclusion = full_xml_value("conclusion", response)
        return bool(conclusion) and not is_invalid_conclusion(conclusion)
    if agent == "AnswerSummarizationAgent":
        answer = full_xml_value("answer", response)
        return bool(answer) and normalize_answer(answer) == normalize_answer(final_answer)
    return False


def selected_sft_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    final_answer = str(trace.get("final_answer", ""))
    selected = [
        event for event in trace.get("sft_events", [])
        if isinstance(event, dict) and event_response_is_valid(event, final_answer)
    ]
    # Only the final valid summarization call should supervise the final answer.
    final_indices = [
        index for index, event in enumerate(selected)
        if event.get("agent_name") == "AnswerSummarizationAgent"
    ]
    if len(final_indices) > 1:
        keep = final_indices[-1]
        selected = [
            event for index, event in enumerate(selected)
            if event.get("agent_name") != "AnswerSummarizationAgent" or index == keep
        ]
    return selected


def validate_event_coverage(trace: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    node_logs = trace.get("nodes_execution", [])
    by_agent = Counter(str(event.get("agent_name")) for event in events)
    if by_agent["PlanningAgent"] != 1:
        reasons.append("accepted events do not contain exactly one valid planner action")
    if by_agent["AnswerSummarizationAgent"] != 1:
        reasons.append("accepted events do not contain exactly one valid final action")
    search_node_ids = {str(node.get("id")) for node in node_logs if node.get("type") == "search"}
    reason_node_ids = {str(node.get("id")) for node in node_logs if node.get("type") == "reason"}
    event_search_ids = {
        str(event.get("metadata", {}).get("node_id"))
        for event in events if event.get("agent_name") == "AnswerGenerationAgent"
    }
    event_reason_ids = {
        str(event.get("metadata", {}).get("node_id"))
        for event in events if event.get("agent_name") == "ReasonerAgent"
    }
    dependent_search_ids = {
        str(node.get("id"))
        for node in node_logs
        if node.get("type") == "search" and find_reference_ids(str(node.get("query", "")))
    }
    event_rewrite_ids = {
        str(event.get("metadata", {}).get("node_id"))
        for event in events if event.get("agent_name") == "QueryRewriteAgent"
    }
    if not search_node_ids.issubset(event_search_ids):
        reasons.append("at least one search node has no valid SFT action")
    if not reason_node_ids.issubset(event_reason_ids):
        reasons.append("at least one reason node has no valid SFT action")
    if not dependent_search_ids.issubset(event_rewrite_ids):
        reasons.append("at least one dependent search has no valid rewrite SFT action")
    return reasons


def event_to_sft_row(
    event: dict[str, Any],
    trajectory_id: str,
    question: str,
    source_index: int,
) -> dict[str, Any]:
    messages = deepcopy(event["messages"])
    for message in messages:
        if message.get("role") == "assistant":
            message["loss_mask"] = 0
    messages.append({"role": "assistant", "content": str(event["response"]), "loss_mask": 1})
    metadata = to_python(event.get("metadata", {}))
    return {
        "messages": messages,
        "enable_thinking": False,
        "agent_name": str(event["agent_name"]),
        "trajectory_id": trajectory_id,
        "source_index": int(source_index),
        "question": question,
        "event_metadata": json.dumps(metadata, sort_keys=True, ensure_ascii=False),
    }


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(to_python(record), ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def collect_one(
    item: dict[str, Any],
    api_url: str,
    candidates: int,
    timeout: float,
    max_nodes: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    rejections: list[dict[str, Any]] = []
    for candidate_index in range(candidates):
        try:
            response = requests.post(api_url, json={"query": item["question"]}, timeout=timeout)
            response.raise_for_status()
            trace = response.json()
            reasons, diagnostics = validate_trace(item["question"], item["gold_answers"], trace, max_nodes)
            events = selected_sft_events(trace) if not reasons else []
            if not reasons:
                reasons.extend(validate_event_coverage(trace, events))
            if not reasons:
                trajectory_id = f"rosa_sft_{item['source_index']}_{candidate_index}"
                trace_without_events = dict(trace)
                trace_without_events.pop("sft_events", None)
                return ({
                    **item,
                    "trajectory_id": trajectory_id,
                    "candidate_index": candidate_index,
                    "diagnostics": diagnostics,
                    "selected_sft_events": events,
                    "trace": trace_without_events,
                }, rejections)
            rejections.append({
                "source_index": item["source_index"],
                "question": item["question"],
                "candidate_index": candidate_index,
                "reasons": reasons,
                "diagnostics": diagnostics,
            })
        except Exception as exc:
            rejections.append({
                "source_index": item["source_index"],
                "question": item["question"],
                "candidate_index": candidate_index,
                "reasons": [f"request failed: {type(exc).__name__}: {exc}"],
            })
    return None, rejections


def write_sft_outputs(
    accepted: list[dict[str, Any]],
    output_dir: Path,
    seed: int,
    val_ratio: float,
    role_repeats: dict[str, int] | None = None,
) -> dict[str, Any]:
    base_rows: list[dict[str, Any]] = []
    seen = set()
    for trajectory in accepted:
        for event in trajectory["selected_sft_events"]:
            row = event_to_sft_row(
                event,
                trajectory["trajectory_id"],
                trajectory["question"],
                trajectory["source_index"],
            )
            dedup_key = (
                trajectory["trajectory_id"],
                json.dumps(row["messages"], sort_keys=True, ensure_ascii=False),
            )
            if dedup_key not in seen:
                seen.add(dedup_key)
                base_rows.append(row)

    role_repeats = role_repeats or {}
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        repeats = max(1, int(role_repeats.get(row["agent_name"], 1)))
        for replica in range(repeats):
            expanded = dict(row)
            expanded["sample_replica"] = replica
            rows.append(expanded)

    trajectory_ids = [record["trajectory_id"] for record in accepted]
    rng = random.Random(seed)
    rng.shuffle(trajectory_ids)
    val_count = min(len(trajectory_ids), max(1, round(len(trajectory_ids) * val_ratio))) if trajectory_ids else 0
    val_ids = set(trajectory_ids[:val_count])
    train_rows = [row for row in rows if row["trajectory_id"] not in val_ids]
    val_rows = [row for row in rows if row["trajectory_id"] in val_ids]

    columns = [
        "messages", "enable_thinking", "agent_name", "trajectory_id",
        "source_index", "question", "event_metadata", "sample_replica",
    ]
    pd.DataFrame(rows, columns=columns).to_parquet(output_dir / "sft_all.parquet", index=False)
    pd.DataFrame(train_rows, columns=columns).to_parquet(output_dir / "sft_train.parquet", index=False)
    pd.DataFrame(val_rows, columns=columns).to_parquet(output_dir / "sft_val.parquet", index=False)
    return {
        "sft_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "role_counts": dict(Counter(row["agent_name"] for row in rows)),
        "train_trajectories": len(set(row["trajectory_id"] for row in train_rows)),
        "val_trajectories": len(val_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8084/rosa/chat")
    parser.add_argument("--num-questions", type=int, default=10000)
    parser.add_argument("--candidates-per-question", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-nodes", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--planner-repeat", type=int, default=2)
    parser.add_argument("--rewrite-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh", action="store_true", help="Refuse to reuse an existing output directory.")
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted_trajectories.jsonl"
    rejected_path = args.output_dir / "rejected_attempts.jsonl"
    if args.fresh and (accepted_path.exists() or rejected_path.exists()):
        raise FileExistsError(f"Output already contains collection state: {args.output_dir}")

    accepted_existing = read_jsonl(accepted_path)
    rejected_existing = read_jsonl(rejected_path)
    completed_indices = {
        int(record["source_index"])
        for record in accepted_existing
        if "source_index" in record
    }
    completed_indices.update(
        int(record["source_index"])
        for record in rejected_existing
        if record.get("all_candidates_exhausted") and "source_index" in record
    )

    frame = pd.read_parquet(args.input)
    indices = list(range(len(frame)))
    random.Random(args.seed).shuffle(indices)
    if args.num_questions > 0:
        indices = indices[: min(args.num_questions, len(indices))]
    items = []
    for source_index in indices:
        if source_index in completed_indices:
            continue
        question, gold_answers = extract_question_and_answers(frame.iloc[source_index])
        items.append({
            "source_index": source_index,
            "question": question,
            "gold_answers": gold_answers,
        })

    print(
        f"[COLLECT] source_rows={len(frame)} selected={len(indices)} pending={len(items)} "
        f"existing_accepted={len(accepted_existing)}"
    )
    write_lock = threading.Lock()
    newly_accepted = 0
    newly_rejected_questions = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                collect_one,
                item,
                args.api_url,
                max(1, args.candidates_per_question),
                args.timeout,
                args.max_nodes,
            ): item
            for item in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="grounded-sft"):
            item = futures[future]
            accepted_record, rejections = future.result()
            for rejection in rejections:
                append_jsonl(rejected_path, rejection, write_lock)
            if accepted_record is not None:
                append_jsonl(accepted_path, accepted_record, write_lock)
                newly_accepted += 1
            else:
                append_jsonl(rejected_path, {
                    "source_index": item["source_index"],
                    "question": item["question"],
                    "all_candidates_exhausted": True,
                    "reasons": ["all candidates rejected"],
                }, write_lock)
                newly_rejected_questions += 1

    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    output_stats = write_sft_outputs(
        accepted,
        args.output_dir,
        args.seed,
        args.val_ratio,
        role_repeats={
            "PlanningAgent": args.planner_repeat,
            "QueryRewriteAgent": args.rewrite_repeat,
        },
    )
    rejection_counts = Counter(
        reason
        for record in rejected
        for reason in record.get("reasons", [])
        if reason != "all candidates rejected"
    )
    report = {
        "input": str(args.input),
        "api_url": args.api_url,
        "selected_questions": len(indices),
        "accepted_trajectories": len(accepted),
        "acceptance_rate_percent": round(len(accepted) / max(len(indices), 1) * 100, 2),
        "newly_accepted": newly_accepted,
        "newly_rejected_questions": newly_rejected_questions,
        **output_stats,
        "top_rejection_reasons": rejection_counts.most_common(20),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[DONE] train={args.output_dir / 'sft_train.parquet'} val={args.output_dir / 'sft_val.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
