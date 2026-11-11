#!/usr/bin/env python3
"""Audit SOD-style step divergence on fixed DAGENT student trajectories.

The script has three stages:

1. ``prepare`` selects matched correct/incorrect sibling trajectories and
   materializes the exact policy actions that both models must score.
2. ``score`` computes chosen-token log probabilities for one frozen model.
3. ``analyze`` joins student and teacher scores, computes step-level divergence,
   and summarizes correctness, retrieval-repair, and recovery patterns.

No privileged workflow or gold answer is included in either model prompt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def stable_digest(*parts: Any) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_question(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def is_correct(row: dict[str, Any]) -> bool:
    try:
        return float(row.get("em") or 0.0) == 1.0
    except (TypeError, ValueError):
        return False


def trajectory_id(row: dict[str, Any]) -> str:
    return stable_digest(
        normalize_question(row.get("question")),
        row.get("sample_index"),
        row.get("rollout_index", 0),
    )[:24]


def event_is_retrieval_issue(event: dict[str, Any]) -> bool:
    response = str(event.get("response") or "").lstrip().lower()
    metadata = event.get("metadata") or {}
    return response.startswith("<rewrite>") or int(metadata.get("retry") or 0) > 0


def first_retrieval_issue_step(events: list[dict[str, Any]]) -> int | None:
    for index, event in enumerate(events, start=1):
        if event_is_retrieval_issue(event):
            return index
    return None


def choose_trajectory(
    candidates: list[dict[str, Any]], seed: int, outcome: int
) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda row: stable_digest(
            seed,
            outcome,
            trajectory_id(row),
        ),
    )


def prepare(args: argparse.Namespace) -> int:
    with args.rollout_log.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Rollout log must contain a JSON list")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question = normalize_question(row.get("question"))
        events = (row.get("raw_trace") or {}).get("sft_events") or []
        if question and events and not row.get("error") and not row.get("pipeline_error"):
            grouped[question].append(row)

    mixed: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for question, group in grouped.items():
        correct = [row for row in group if is_correct(row)]
        incorrect = [row for row in group if not is_correct(row)]
        if correct and incorrect:
            mixed.append((question, correct, incorrect))
    mixed.sort(key=lambda item: stable_digest(args.seed, item[0]))
    selected_groups = mixed[: args.question_pairs]
    if len(selected_groups) < args.question_pairs:
        raise RuntimeError(
            f"Requested {args.question_pairs} mixed-outcome questions, "
            f"but only {len(selected_groups)} are available"
        )

    trajectory_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    for question, correct, incorrect in selected_groups:
        pair_id = stable_digest(question)[:20]
        selected = (
            choose_trajectory(correct, args.seed, 1),
            choose_trajectory(incorrect, args.seed, 0),
        )
        for row in selected:
            events = (row.get("raw_trace") or {}).get("sft_events") or []
            tid = trajectory_id(row)
            issue_step = first_retrieval_issue_step(events)
            trajectory_rows.append(
                {
                    "trajectory_id": tid,
                    "pair_id": pair_id,
                    "question": row.get("question"),
                    "sample_index": row.get("sample_index"),
                    "rollout_index": row.get("rollout_index", 0),
                    "outcome": int(is_correct(row)),
                    "em": float(row.get("em") or 0.0),
                    "f1": float(row.get("f1") or 0.0),
                    "event_count": len(events),
                    "retrieval_issue_step": issue_step,
                    "has_explicit_retrieval_issue": issue_step is not None,
                    "search_steps": int(row.get("search_steps") or 0),
                    "retrieval_calls": int(row.get("retrieval_calls") or 0),
                }
            )
            for step_index, event in enumerate(events, start=1):
                response = str(event.get("response") or "")
                messages = event.get("messages") or []
                if not response.strip() or not messages:
                    continue
                role = str(event.get("agent_name") or "UnknownAgent")
                role_counts[role] += 1
                action_rows.append(
                    {
                        "action_id": stable_digest(tid, step_index, role)[:28],
                        "trajectory_id": tid,
                        "pair_id": pair_id,
                        "outcome": int(is_correct(row)),
                        "step_index": step_index,
                        "role": role,
                        "node_id": str((event.get("metadata") or {}).get("node_id") or ""),
                        "is_retrieval_issue_step": event_is_retrieval_issue(event),
                        "messages": messages,
                        "response": response,
                    }
                )

    write_jsonl(args.output_dir / "trajectories.jsonl", trajectory_rows)
    write_jsonl(args.output_dir / "actions.jsonl", action_rows)
    summary = {
        "rollout_log": str(args.rollout_log),
        "seed": args.seed,
        "available_rows": len(rows),
        "available_question_groups": len(grouped),
        "available_mixed_outcome_questions": len(mixed),
        "selected_question_pairs": len(selected_groups),
        "selected_trajectories": len(trajectory_rows),
        "selected_actions": len(action_rows),
        "correct_trajectories": sum(row["outcome"] for row in trajectory_rows),
        "incorrect_trajectories": sum(
            not row["outcome"] for row in trajectory_rows
        ),
        "explicit_retrieval_issue_trajectories": sum(
            row["has_explicit_retrieval_issue"] for row in trajectory_rows
        ),
        "role_counts": dict(role_counts),
        "retrieval_issue_definition": (
            "First policy event whose response starts with <rewrite> or whose "
            "metadata retry index is positive. This is an observable retrieval-"
            "repair proxy, not a gold semantic error label."
        ),
    }
    write_json(args.output_dir / "selection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def prompt_token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_scoring_items(
    tokenizer: Any,
    action_rows: list[dict[str, Any]],
    completed: set[str],
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in action_rows:
        if row["action_id"] in completed:
            continue
        prompt_ids = prompt_token_ids(tokenizer, row["messages"])
        response_ids = tokenizer(
            str(row["response"]), add_special_tokens=False
        )["input_ids"]
        reason = ""
        if not prompt_ids:
            reason = "empty_prompt"
        elif not response_ids:
            reason = "empty_response"
        elif len(prompt_ids) + len(response_ids) > max_length:
            reason = "too_long"
        if reason:
            rejected.append(
                {
                    "action_id": row["action_id"],
                    "error": reason,
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(response_ids),
                }
            )
            continue
        items.append(
            {
                "action_id": row["action_id"],
                "input_ids": prompt_ids + response_ids,
                "response_ids": response_ids,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids),
            }
        )
    items.sort(key=lambda item: (len(item["input_ids"]), item["response_tokens"]))
    return items, rejected


def make_batches(
    items: list[dict[str, Any]],
    batch_size: int,
    max_batch_tokens: int,
) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    max_length = 0
    for item in items:
        candidate_max = max(max_length, len(item["input_ids"]))
        candidate_size = len(batch) + 1
        if batch and (
            candidate_size > batch_size
            or candidate_max * candidate_size > max_batch_tokens
        ):
            yield batch
            batch = []
            max_length = 0
        batch.append(item)
        max_length = max(max_length, len(item["input_ids"]))
    if batch:
        yield batch


def score(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    action_rows = read_jsonl(args.actions)
    existing = read_jsonl(args.output)
    completed = {row["action_id"] for row in existing}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    items, rejected = build_scoring_items(
        tokenizer,
        action_rows,
        completed,
        args.max_length,
    )
    if rejected:
        append_jsonl(args.output, rejected)
        completed.update(row["action_id"] for row in rejected)

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation=args.attention,
            **model_kwargs,
        ).cuda()
    except (ImportError, ValueError) as error:
        if args.attention == "sdpa":
            raise
        print(f"[WARN] {args.attention} unavailable ({error}); using SDPA")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="sdpa",
            **model_kwargs,
        ).cuda()
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True

    batches = list(make_batches(items, args.batch_size, args.max_batch_tokens))
    processed = len(completed)
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches, start=1):
            maximum_length = max(len(item["input_ids"]) for item in batch)
            maximum_response = max(item["response_tokens"] for item in batch)
            input_ids = []
            attention_mask = []
            for item in batch:
                padding = maximum_length - len(item["input_ids"])
                input_ids.append(
                    [tokenizer.pad_token_id] * padding + item["input_ids"]
                )
                attention_mask.append(
                    [0] * padding + [1] * len(item["input_ids"])
                )
            input_tensor = torch.tensor(
                input_ids, dtype=torch.long, device="cuda"
            )
            mask_tensor = torch.tensor(
                attention_mask, dtype=torch.long, device="cuda"
            )
            position_ids = mask_tensor.cumsum(dim=-1) - 1
            position_ids.masked_fill_(mask_tensor == 0, 0)
            try:
                output = model(
                    input_ids=input_tensor,
                    attention_mask=mask_tensor,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=maximum_response + 1,
                )
            except TypeError:
                output = model(
                    input_ids=input_tensor,
                    attention_mask=mask_tensor,
                    position_ids=position_ids,
                    use_cache=False,
                )
            logits = output.logits
            cropped = logits.shape[1] <= maximum_response + 1
            output_rows = []
            for row_index, item in enumerate(batch):
                response_length = item["response_tokens"]
                if cropped:
                    start = maximum_response - response_length
                else:
                    start = maximum_length - response_length - 1
                token_logits = logits[
                    row_index,
                    start : start + response_length,
                    :,
                ].float()
                targets = torch.tensor(
                    item["response_ids"],
                    dtype=torch.long,
                    device=token_logits.device,
                )
                chosen = token_logits.gather(-1, targets[:, None]).squeeze(-1)
                log_probs = chosen - torch.logsumexp(token_logits, dim=-1)
                if log_probs.numel() != response_length:
                    raise RuntimeError(
                        f"Alignment error for {item['action_id']}: "
                        f"{log_probs.numel()} != {response_length}"
                    )
                output_rows.append(
                    {
                        "action_id": item["action_id"],
                        "model": str(args.model),
                        "prompt_tokens": item["prompt_tokens"],
                        "response_tokens": response_length,
                        "token_ids": item["response_ids"],
                        "token_logprobs": [
                            round(float(value), 8)
                            for value in log_probs.detach().cpu()
                        ],
                    }
                )
            append_jsonl(args.output, output_rows)
            processed += len(output_rows)
            if batch_index == 1 or batch_index % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "batches": len(batches),
                            "processed": processed,
                            "total_actions": len(action_rows),
                            "max_input_tokens": maximum_length,
                            "max_response_tokens": maximum_response,
                        }
                    ),
                    flush=True,
                )
            del output, logits, input_tensor, mask_tensor, position_ids

    final = read_jsonl(args.output)
    ids = [row["action_id"] for row in final]
    summary = {
        "model": str(args.model),
        "actions": len(action_rows),
        "score_rows": len(final),
        "unique_action_ids": len(set(ids)),
        "errors": sum(bool(row.get("error")) for row in final),
        "complete": len(final) == len(action_rows) == len(set(ids)),
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise RuntimeError(f"Incomplete scoring output: {summary}")
    return 0


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = (len(values) - 1) / 2
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    if not denominator:
        return 0.0
    return sum(
        (index - center) * (value - statistics.mean(values))
        for index, value in enumerate(values)
    ) / denominator


def bootstrap_ci(
    values: list[float], seed: int, samples: int = 5000
) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(
            statistics.mean(rng.choice(values) for _ in range(len(values)))
        )
    estimates.sort()
    return [
        estimates[round(0.025 * (samples - 1))],
        estimates[round(0.975 * (samples - 1))],
    ]


def summarize_values(values: list[float], seed: int) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "median": statistics.median(values) if values else None,
        "bootstrap_95ci": bootstrap_ci(values, seed),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze(args: argparse.Namespace) -> int:
    import numpy as np

    trajectories = {
        row["trajectory_id"]: row for row in read_jsonl(args.trajectories)
    }
    actions = {row["action_id"]: row for row in read_jsonl(args.actions)}
    student = {row["action_id"]: row for row in read_jsonl(args.student_scores)}
    teacher = {row["action_id"]: row for row in read_jsonl(args.teacher_scores)}
    missing = sorted(set(actions) - set(student) | (set(actions) - set(teacher)))
    if missing:
        raise RuntimeError(f"Missing model scores for {len(missing)} actions")

    scored_actions: list[dict[str, Any]] = []
    for action_id, action in actions.items():
        student_row = student[action_id]
        teacher_row = teacher[action_id]
        if student_row.get("error") or teacher_row.get("error"):
            continue
        if student_row["token_ids"] != teacher_row["token_ids"]:
            raise RuntimeError(f"Tokenizer mismatch for action {action_id}")
        student_logp = np.asarray(student_row["token_logprobs"], dtype=np.float64)
        teacher_logp = np.asarray(teacher_row["token_logprobs"], dtype=np.float64)
        if student_logp.shape != teacher_logp.shape or not student_logp.size:
            raise RuntimeError(f"Score shape mismatch for action {action_id}")
        divergence = float(np.abs(student_logp - teacher_logp).mean())
        signed_gap = float((teacher_logp - student_logp).mean())
        scored_actions.append(
            {
                **{
                    key: action[key]
                    for key in (
                        "action_id",
                        "trajectory_id",
                        "pair_id",
                        "outcome",
                        "step_index",
                        "role",
                        "node_id",
                        "is_retrieval_issue_step",
                    )
                },
                "token_count": int(student_logp.size),
                "d_k": divergence,
                "teacher_minus_student_chosen_logp": signed_gap,
            }
        )

    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_actions:
        by_trajectory[row["trajectory_id"]].append(row)
    trajectory_metrics: list[dict[str, Any]] = []
    for tid, rows in by_trajectory.items():
        rows.sort(key=lambda row: row["step_index"])
        values = [row["d_k"] for row in rows]
        if not values:
            continue
        first = values[0]
        for row in rows:
            row["sod_weight"] = min(
                (first + args.epsilon) / (row["d_k"] + args.epsilon),
                1.0 + args.delta,
            )
        meta = trajectories[tid]
        issue_step = meta.get("retrieval_issue_step")
        issue_position = next(
            (
                index
                for index, row in enumerate(rows)
                if row["step_index"] == issue_step
            ),
            None,
        )
        recovery_any = None
        recovery_final = None
        continued_rise = None
        post_issue_slope = None
        if issue_position is not None:
            post = values[issue_position:]
            post_issue_slope = linear_slope(post)
            peak_position = max(range(len(post)), key=post.__getitem__)
            peak = post[peak_position]
            later = post[peak_position + 1 :]
            recovery_any = bool(
                later and min(later) <= peak * (1.0 - args.recovery_drop)
            )
            recovery_final = bool(
                later and post[-1] <= peak * (1.0 - args.recovery_drop)
            )
            continued_rise = bool(
                len(post) >= 2
                and post[-1] >= post[0] * (1.0 + args.rise_threshold)
                and (post_issue_slope or 0.0) > 0
            )
        trajectory_metrics.append(
            {
                **meta,
                "scored_event_count": len(rows),
                "mean_d": statistics.mean(values),
                "max_d": max(values),
                "first_d": values[0],
                "final_d": values[-1],
                "trajectory_slope": linear_slope(values),
                "mean_sod_weight": statistics.mean(
                    row["sod_weight"] for row in rows
                ),
                "post_issue_slope": post_issue_slope,
                "recovery_any": recovery_any,
                "recovery_final": recovery_final,
                "continued_rise": continued_rise,
            }
        )

    outcome_summary: dict[str, Any] = {}
    for outcome in (0, 1):
        subset = [row for row in trajectory_metrics if row["outcome"] == outcome]
        label = "correct" if outcome else "incorrect"
        outcome_summary[label] = {
            "trajectories": len(subset),
            "mean_d": summarize_values(
                [row["mean_d"] for row in subset], args.seed + outcome
            ),
            "max_d": summarize_values(
                [row["max_d"] for row in subset], args.seed + 10 + outcome
            ),
            "final_d": summarize_values(
                [row["final_d"] for row in subset], args.seed + 20 + outcome
            ),
            "trajectory_slope": summarize_values(
                [
                    row["trajectory_slope"]
                    for row in subset
                    if row["trajectory_slope"] is not None
                ],
                args.seed + 30 + outcome,
            ),
            "mean_sod_weight": summarize_values(
                [row["mean_sod_weight"] for row in subset],
                args.seed + 40 + outcome,
            ),
        }

    by_pair: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in trajectory_metrics:
        by_pair[row["pair_id"]][row["outcome"]] = row
    paired_differences = [
        pair[0]["mean_d"] - pair[1]["mean_d"]
        for pair in by_pair.values()
        if 0 in pair and 1 in pair
    ]

    step_rows: list[dict[str, Any]] = []
    grouped_steps: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in scored_actions:
        grouped_steps[(row["outcome"], row["step_index"])].append(row["d_k"])
    for (outcome, step_index), values in sorted(grouped_steps.items()):
        step_rows.append(
            {
                "outcome": outcome,
                "step_index": step_index,
                "n": len(values),
                "mean_d": statistics.mean(values),
                "median_d": statistics.median(values),
            }
        )

    role_rows: list[dict[str, Any]] = []
    grouped_roles: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in scored_actions:
        grouped_roles[(row["outcome"], row["role"])].append(row["d_k"])
    for (outcome, role), values in sorted(grouped_roles.items()):
        role_rows.append(
            {
                "outcome": outcome,
                "role": role,
                "n": len(values),
                "mean_d": statistics.mean(values),
                "median_d": statistics.median(values),
            }
        )

    relative_rows_raw: dict[tuple[int, int], list[float]] = defaultdict(list)
    for tid, rows in by_trajectory.items():
        issue_step = trajectories[tid].get("retrieval_issue_step")
        if issue_step is None:
            continue
        for row in rows:
            offset = row["step_index"] - issue_step
            if args.relative_min <= offset <= args.relative_max:
                relative_rows_raw[(row["outcome"], offset)].append(row["d_k"])
    relative_rows = [
        {
            "outcome": outcome,
            "relative_step": offset,
            "n": len(values),
            "mean_d": statistics.mean(values),
            "median_d": statistics.median(values),
        }
        for (outcome, offset), values in sorted(relative_rows_raw.items())
    ]

    issue_metrics: dict[str, Any] = {}
    for outcome in (0, 1):
        subset = [
            row
            for row in trajectory_metrics
            if row["outcome"] == outcome
            and row["retrieval_issue_step"] is not None
            and row["recovery_any"] is not None
        ]
        label = "correct" if outcome else "incorrect"
        issue_metrics[label] = {
            "trajectories": len(subset),
            "mean_post_issue_slope": mean(
                [
                    row["post_issue_slope"]
                    for row in subset
                    if row["post_issue_slope"] is not None
                ]
            ),
            "continued_rise_fraction": mean(
                [float(row["continued_rise"]) for row in subset]
            ),
            "recovery_any_fraction": mean(
                [float(row["recovery_any"]) for row in subset]
            ),
            "recovery_final_fraction": mean(
                [float(row["recovery_final"]) for row in subset]
            ),
        }

    summary = {
        "definition": {
            "d_k": (
                "Mean absolute difference between Student and Teacher chosen-token "
                "log probabilities within one model-generated Agent action."
            ),
            "student_context": "Original on-policy event messages only.",
            "teacher_context": "The exact same event messages; no privileged reference.",
            "retrieval_issue": (
                "First <rewrite> response or positive retry index. This is an "
                "explicit retrieval-repair proxy rather than a gold semantic label."
            ),
            "recovery": (
                f"After an explicit retrieval issue, a later step drops at least "
                f"{args.recovery_drop:.0%} from the post-issue divergence peak."
            ),
        },
        "student_scores": str(args.student_scores),
        "teacher_scores": str(args.teacher_scores),
        "scored_actions": len(scored_actions),
        "scored_trajectories": len(trajectory_metrics),
        "outcome": outcome_summary,
        "matched_question_analysis": {
            "pairs": len(paired_differences),
            "incorrect_minus_correct_mean_d": summarize_values(
                paired_differences, args.seed + 100
            ),
            "fraction_incorrect_greater": mean(
                [float(value > 0) for value in paired_differences]
            ),
        },
        "explicit_retrieval_issue": issue_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scored_actions.jsonl", scored_actions)
    write_jsonl(args.output_dir / "trajectory_metrics.jsonl", trajectory_metrics)
    write_csv(args.output_dir / "step_by_outcome.csv", step_rows)
    write_csv(args.output_dir / "role_by_outcome.csv", role_rows)
    write_csv(args.output_dir / "retrieval_issue_relative.csv", relative_rows)
    write_json(args.output_dir / "summary.json", summary)

    report_lines = [
        "# DAGENT SOD Step-Divergence Audit",
        "",
        f"- Scored trajectories: {len(trajectory_metrics)}",
        f"- Scored actions: {len(scored_actions)}",
        f"- Matched question pairs: {len(paired_differences)}",
        "",
        "## Correct vs. Incorrect",
        "",
    ]
    for label in ("correct", "incorrect"):
        item = outcome_summary[label]
        report_lines.append(
            f"- {label}: n={item['trajectories']}, "
            f"mean trajectory d={item['mean_d']['mean']:.6f}, "
            f"final d={item['final_d']['mean']:.6f}, "
            f"mean SOD weight={item['mean_sod_weight']['mean']:.6f}"
        )
    pair_summary = summary["matched_question_analysis"][
        "incorrect_minus_correct_mean_d"
    ]
    report_lines.extend(
        [
            "",
            "## Matched Sibling Difference",
            "",
            (
                "- Incorrect minus correct mean divergence: "
                f"{pair_summary['mean']:.6f} "
                f"(95% bootstrap CI {pair_summary['bootstrap_95ci']})"
            ),
            (
                "- Fraction of questions where incorrect sibling has larger "
                f"divergence: {summary['matched_question_analysis']['fraction_incorrect_greater']:.4f}"
            ),
            "",
            "## Explicit Retrieval-Issue Proxy",
            "",
        ]
    )
    for label in ("correct", "incorrect"):
        item = issue_metrics[label]
        report_lines.append(
            f"- {label}: n={item['trajectories']}, "
            f"continued rise={item['continued_rise_fraction']}, "
            f"any recovery={item['recovery_any_fraction']}, "
            f"final recovery={item['recovery_final_fraction']}"
        )
    report_lines.extend(
        [
            "",
            "The retrieval-issue label is an explicit repair signal, not a gold "
            "semantic judgment of every retrieved document.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--rollout-log", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--question-pairs", type=int, default=400)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.set_defaults(function=prepare)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--actions", type=Path, required=True)
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--max-length", type=int, default=8192)
    score_parser.add_argument("--batch-size", type=int, default=2)
    score_parser.add_argument("--max-batch-tokens", type=int, default=10000)
    score_parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    score_parser.add_argument("--log-every", type=int, default=25)
    score_parser.set_defaults(function=score)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--trajectories", type=Path, required=True)
    analyze_parser.add_argument("--actions", type=Path, required=True)
    analyze_parser.add_argument("--student-scores", type=Path, required=True)
    analyze_parser.add_argument("--teacher-scores", type=Path, required=True)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--epsilon", type=float, default=1e-6)
    analyze_parser.add_argument("--delta", type=float, default=0.2)
    analyze_parser.add_argument("--recovery-drop", type=float, default=0.2)
    analyze_parser.add_argument("--rise-threshold", type=float, default=0.2)
    analyze_parser.add_argument("--relative-min", type=int, default=-2)
    analyze_parser.add_argument("--relative-max", type=int, default=4)
    analyze_parser.add_argument("--seed", type=int, default=42)
    analyze_parser.set_defaults(function=analyze)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
