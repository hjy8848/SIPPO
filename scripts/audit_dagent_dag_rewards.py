#!/usr/bin/env python3
"""Audit DAG-specific frozen-model rewards without an external judge model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_manager.normalize_answers import normalize_answer
from qa_manager.ROSA.attribution_reward import (
    build_query_binding_messages,
    build_search_attribution_messages,
    build_summary_attribution_messages,
    descendant_closure,
    history_as_search_context,
    node_dependency_ids,
    render_workflow_history,
    tagged_attribution_target,
)
from qa_manager.ROSA.protocol import (
    ANSWER_GENERATION_SYSTEM_MSG,
    FINAL_SYSTEM_MSG,
    QUERY_REWRITE_SYSTEM_MSG,
)


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "from", "was", "were", "is", "are", "who", "what", "when", "where",
    "which", "did", "does", "do", "with", "by", "both", "same",
}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalized(value: Any) -> str:
    return normalize_answer(str(value or ""))


def exact_match(value: Any, answers: Iterable[Any]) -> bool:
    candidate = normalized(value)
    return bool(candidate) and any(candidate == normalized(answer) for answer in answers)


def supported(value: Any, evidence: Iterable[Any]) -> bool:
    candidate = normalized(value)
    return bool(candidate) and any(candidate in normalized(item) for item in evidence)


def split_docs(context: Any) -> list[str]:
    text = str(context or "").strip()
    if not text:
        return []
    starts = list(re.finditer(r"(?m)^\[\d+_\d+\]:", text))
    if not starts:
        return [text]
    docs = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        docs.append(text[match.start():end].strip())
    return docs


def source_name(value: Any) -> str:
    text = str(value or "").casefold()
    if "2wiki" in text:
        return "2wiki"
    if "hotpot" in text:
        return "hotpotqa"
    if "musique" in text:
        return "musique"
    if "bamboogle" in text:
        return "bamboogle"
    return text or "unknown"


def informative_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized(value))
        if len(token) >= 3 and token not in STOPWORDS
    }


def binding_is_expressed(values: Sequence[str], rewritten_query: str) -> bool:
    query_tokens = informative_tokens(rewritten_query)
    if not values or "#" in str(rewritten_query):
        return False
    for value in values:
        value_tokens = informative_tokens(value)
        if value_tokens and not value_tokens.intersection(query_tokens):
            return False
        if not value_tokens and normalized(value) not in normalized(rewritten_query):
            return False
    return True


def standardize_nodes(trace: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for raw in trace.get("nodes_execution") or []:
        node_type = str(raw.get("type", "")).casefold()
        if node_type not in {"search", "reason"}:
            continue
        nodes.append(
            {
                "id": str(raw.get("id", "")),
                "type": node_type,
                "task": str(raw.get("task", raw.get("query", ""))),
                "query": str(raw.get("query", raw.get("task", ""))),
                "resolved_task": str(raw.get("resolved_task", raw.get("resolved_query", ""))),
                "executed_query": str(raw.get("executed_query", raw.get("resolved_query", ""))),
                "conclusion": str(raw.get("conclusion", "")).strip(),
                "docs": split_docs(raw.get("search_context", "")),
            }
        )
    return nodes


def load_trajectories(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    valid = []
    for row_index, row in enumerate(rows):
        trace = row.get("trace") or row.get("raw_trace") or {}
        if row.get("status") not in {None, "ok"} or trace.get("execution_error"):
            continue
        nodes = standardize_nodes(trace)
        if not nodes:
            continue
        answers = [str(item) for item in as_list(row.get("gold_answers", row.get("gold_answer")))]
        final_answer = str(row.get("final_answer", trace.get("final_answer", ""))).strip()
        source = source_name(row.get("data_source", row.get("source_file", "")))
        valid.append(
            {
                "trajectory_id": f"{source}:{row.get('row_index', row_index)}",
                "source": source,
                "question": str(row.get("question", trace.get("question", ""))),
                "answers": answers,
                "final_answer": final_answer,
                "em": int(row.get("em", exact_match(final_answer, answers))),
                "nodes": nodes,
            }
        )
    if limit > 0 and len(valid) > limit:
        rng = random.Random(seed)
        rng.shuffle(valid)
        valid = valid[:limit]
    return valid


def choose_foreign_bindings(cases: Sequence[dict[str, Any]], index: int) -> dict[str, str]:
    target = cases[index]
    references = list(target["spec"]["bindings"])
    candidates = []
    for candidate_index, candidate in enumerate(cases):
        if candidate_index == index or candidate["source"] != target["source"]:
            continue
        values = list(candidate["spec"]["bindings"].values())
        if len(values) != len(references):
            continue
        if any(supported(target["spec"]["output"], [value]) for value in values):
            continue
        distance = sum(
            abs(len(str(left)) - len(str(right)))
            for left, right in zip(target["spec"]["bindings"].values(), values)
        )
        candidates.append((distance, candidate["case_id"], values))
    if not candidates:
        return {reference: "UNKNOWN" for reference in references}
    values = min(candidates)[2]
    return dict(zip(references, values, strict=True))


def choose_subgraph_replacements(
    trajectory: dict[str, Any],
    closure: set[str],
    node_pool: Sequence[dict[str, str]],
) -> dict[str, str]:
    answers = trajectory["answers"]
    replacements = {}
    for node in trajectory["nodes"]:
        if node["id"] not in closure:
            continue
        candidates = []
        for candidate in node_pool:
            if candidate["trajectory_id"] == trajectory["trajectory_id"]:
                continue
            if candidate["source"] != trajectory["source"] or candidate["type"] != node["type"]:
                continue
            value = candidate["conclusion"]
            if not value or any(supported(answer, [value]) for answer in answers):
                continue
            candidates.append(
                (abs(len(value) - len(node["conclusion"])), candidate["trajectory_id"], value)
            )
        replacements[node["id"]] = min(candidates)[2] if candidates else "unavailable"
    return replacements


def build_cases(trajectories: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    incremental = []
    bindings = []
    subgraphs = []
    node_pool = [
        {
            "trajectory_id": trajectory["trajectory_id"],
            "source": trajectory["source"],
            "type": node["type"],
            "conclusion": node["conclusion"],
        }
        for trajectory in trajectories
        for node in trajectory["nodes"]
        if node["conclusion"]
    ]

    for trajectory in trajectories:
        completed = []
        by_id = {node["id"]: node for node in trajectory["nodes"]}
        for node in trajectory["nodes"]:
            history = render_workflow_history(completed)
            if node["type"] == "search" and node["docs"] and node["conclusion"] and completed:
                grounded = supported(node["conclusion"], node["docs"])
                novel = not supported(
                    node["conclusion"], [item["conclusion"] for item in completed]
                )
                incremental.append(
                    {
                        "case_id": f"incremental:{trajectory['trajectory_id']}:{node['id']}",
                        "candidate": "search_incremental_attribution",
                        "source": trajectory["source"],
                        "trajectory_id": trajectory["trajectory_id"],
                        "em": trajectory["em"],
                        "label": int(grounded and novel),
                        "label_components": {"grounded": grounded, "novel": novel},
                        "spec": {
                            "node_query": node["executed_query"] or node["resolved_task"] or node["task"],
                            "docs": node["docs"],
                            "history": history,
                            "output": node["conclusion"],
                        },
                    }
                )

            references = node_dependency_ids(node)
            if node["type"] == "search" and references and node["executed_query"]:
                bound = {reference: by_id.get(reference, {}).get("conclusion", "") for reference in references}
                values = list(bound.values())
                bindings.append(
                    {
                        "case_id": f"binding:{trajectory['trajectory_id']}:{node['id']}",
                        "candidate": "query_binding_attribution",
                        "source": trajectory["source"],
                        "trajectory_id": trajectory["trajectory_id"],
                        "em": trajectory["em"],
                        "label": int(binding_is_expressed(values, node["executed_query"])),
                        "spec": {
                            "question": trajectory["question"],
                            "draft_query": node["task"],
                            "bindings": bound,
                            "output": node["executed_query"],
                        },
                    }
                )
            completed.append(node)

        if trajectory["em"] and trajectory["final_answer"]:
            full_history = render_workflow_history(trajectory["nodes"])
            for node in trajectory["nodes"]:
                closure = descendant_closure(trajectory["nodes"], node["id"])
                replacements = choose_subgraph_replacements(trajectory, closure, node_pool)
                counterfactual_history = render_workflow_history(
                    trajectory["nodes"], conclusion_overrides=replacements
                )
                closure_conclusions = [
                    item["conclusion"] for item in trajectory["nodes"] if item["id"] in closure
                ]
                remaining_conclusions = [
                    item["conclusion"] for item in trajectory["nodes"] if item["id"] not in closure
                ]
                subgraphs.append(
                    {
                        "case_id": f"subgraph:{trajectory['trajectory_id']}:{node['id']}",
                        "candidate": "dag_subgraph_attribution",
                        "source": trajectory["source"],
                        "trajectory_id": trajectory["trajectory_id"],
                        "em": trajectory["em"],
                        "label": None,
                        "support_removed": int(
                            supported(trajectory["final_answer"], closure_conclusions)
                            and not supported(trajectory["final_answer"], remaining_conclusions)
                        ),
                        "node_type": node["type"],
                        "closure_size": len(closure),
                        "spec": {
                            "question": trajectory["question"],
                            "answers": trajectory["answers"],
                            "history": full_history,
                            "counterfactual_history": counterfactual_history,
                            "output": trajectory["final_answer"],
                        },
                    }
                )

    for index, case in enumerate(bindings):
        case["spec"]["counterfactual_bindings"] = choose_foreign_bindings(bindings, index)
    return incremental + bindings + subgraphs


@dataclass
class Probe:
    case_index: int
    variant: str
    input_ids: list[int]
    response_start: int
    response_length: int


def encode_probe(
    tokenizer: Any,
    case_index: int,
    variant: str,
    messages: list[dict[str, str]],
    prefix: str,
    response: str,
    max_prompt_tokens: int,
) -> Probe:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_tokens:
        raise ValueError(f"prompt has {len(prompt_ids)} tokens")
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not response_ids:
        raise ValueError("empty response")
    return Probe(case_index, variant, prompt_ids + response_ids, len(prompt_ids), len(response_ids))


def case_messages(case: dict[str, Any]) -> tuple[dict[str, list[dict[str, str]]], str, str]:
    spec = case["spec"]
    if case["candidate"] == "search_incremental_attribution":
        variants = {
            "factual": build_search_attribution_messages(
                ANSWER_GENERATION_SYSTEM_MSG, spec["node_query"], spec["docs"]
            ),
            "counterfactual": build_search_attribution_messages(
                ANSWER_GENERATION_SYSTEM_MSG,
                spec["node_query"],
                history_as_search_context(spec["history"]),
            ),
        }
        prefix, response = tagged_attribution_target("conclusion", spec["output"])
    elif case["candidate"] == "query_binding_attribution":
        variants = {
            "factual": build_query_binding_messages(
                QUERY_REWRITE_SYSTEM_MSG,
                spec["question"],
                spec["draft_query"],
                spec["bindings"],
            ),
            "counterfactual": build_query_binding_messages(
                QUERY_REWRITE_SYSTEM_MSG,
                spec["question"],
                spec["draft_query"],
                spec["counterfactual_bindings"],
            ),
        }
        prefix, response = "", spec["output"]
    else:
        variants = {
            "factual": build_summary_attribution_messages(
                FINAL_SYSTEM_MSG, spec["question"], spec["history"]
            ),
            "counterfactual": build_summary_attribution_messages(
                FINAL_SYSTEM_MSG, spec["question"], spec["counterfactual_history"]
            ),
        }
        prefix, response = tagged_attribution_target("answer", spec["output"])
    return variants, prefix, response


def score_cases(
    cases: list[dict[str, Any]],
    model_path: str,
    batch_size: int,
    max_prompt_tokens: int,
    generation_batch_size: int,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, fix_mistral_regex=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    probes = []
    overflow = Counter()
    messages_by_case = {}
    for index, case in enumerate(cases):
        variants, prefix, response = case_messages(case)
        messages_by_case[index] = variants
        for variant, messages in variants.items():
            try:
                probes.append(
                    encode_probe(
                        tokenizer, index, variant, messages, prefix, response, max_prompt_tokens
                    )
                )
            except ValueError:
                overflow[index] += 1

    values: dict[int, dict[str, float]] = defaultdict(dict)
    pad_id = int(tokenizer.pad_token_id)
    for offset in range(0, len(probes), batch_size):
        chunk = probes[offset:offset + batch_size]
        width = max(len(probe.input_ids) for probe in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device="cuda")
        attention = torch.zeros((len(chunk), width), dtype=torch.long, device="cuda")
        labels = torch.full((len(chunk), width), -100, dtype=torch.long, device="cuda")
        for row_index, probe in enumerate(chunk):
            padding = width - len(probe.input_ids)
            ids = torch.tensor(probe.input_ids, dtype=torch.long, device="cuda")
            input_ids[row_index, padding:] = ids
            attention[row_index, padding:] = 1
            start = padding + probe.response_start
            labels[row_index, start:start + probe.response_length] = ids[
                probe.response_start:probe.response_start + probe.response_length
            ]
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
            shifted = labels[:, 1:]
            valid = shifted.ne(-100)
            logprobs = torch.log_softmax(logits, dim=-1).gather(
                -1, shifted.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            means = (logprobs * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        for probe, score in zip(chunk, means.tolist(), strict=True):
            values[probe.case_index][probe.variant] = float(score)
        if offset % max(batch_size, batch_size * 20) == 0 or offset + len(chunk) == len(probes):
            print(f"[SCORE] {min(offset + len(chunk), len(probes))}/{len(probes)}")

    subgraph_indices = [
        index
        for index, case in enumerate(cases)
        if case["candidate"] == "dag_subgraph_attribution" and not overflow[index]
    ]
    for offset in range(0, len(subgraph_indices), generation_batch_size):
        indices = subgraph_indices[offset:offset + generation_batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                messages_by_case[index]["counterfactual"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for index in indices
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
            add_special_tokens=False,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for index, output in zip(indices, outputs, strict=True):
            match = re.search(r"<answer>(.*?)</answer>", output, flags=re.I | re.S)
            answer = match.group(1).strip() if match else output.strip()
            cases[index]["counterfactual_generated_answer"] = answer
            cases[index]["label"] = int(not exact_match(answer, cases[index]["spec"]["answers"]))
        print(f"[GENERATE] {min(offset + len(indices), len(subgraph_indices))}/{len(subgraph_indices)}")

    output = []
    for index, case in enumerate(cases):
        row = dict(case)
        scores = values.get(index, {})
        if overflow[index] or not {"factual", "counterfactual"}.issubset(scores):
            row.update({"status": "prompt_overflow", "variant_logprobs": scores})
        elif row.get("label") is None:
            row.update({"status": "generation_failed", "variant_logprobs": scores})
        else:
            margin = scores["factual"] - scores["counterfactual"]
            row.update(
                {
                    "status": "ok",
                    "score": margin,
                    "bounded_score": math.tanh(margin),
                    "variant_logprobs": scores,
                }
            )
        output.append(row)
    return output


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def stable_split(case_id: str) -> str:
    value = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16)
    return "calibration" if value % 2 == 0 else "test"


def threshold_metrics(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, float]:
    tp = sum(row["label"] == 1 and row["score"] > threshold for row in rows)
    fp = sum(row["label"] == 0 and row["score"] > threshold for row in rows)
    fn = sum(row["label"] == 1 and row["score"] <= threshold for row in rows)
    tn = sum(row["label"] == 0 and row["score"] <= threshold for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "count": len(rows),
        "threshold": threshold,
        "selected": tp + fp,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def select_threshold(rows: Sequence[dict[str, Any]]) -> float:
    values = sorted({float(row["score"]) for row in rows})
    candidates = [values[0] - 1e-6] + values if values else [0.0]
    eligible = []
    for threshold in candidates:
        metrics = threshold_metrics(rows, threshold)
        if metrics["precision"] >= 0.85 and metrics["recall"] >= 0.25:
            eligible.append((metrics["balanced_accuracy"], metrics["precision"], threshold))
    if eligible:
        return max(eligible)[2]
    return max(
        ((threshold_metrics(rows, value)["balanced_accuracy"], value) for value in candidates),
        default=(0.0, 0.0),
    )[1]


def bootstrap_gap(rows: Sequence[dict[str, Any]], seed: int) -> tuple[float, float, float]:
    positives = [row["score"] for row in rows if row["label"] == 1]
    negatives = [row["score"] for row in rows if row["label"] == 0]
    if not positives or not negatives:
        return 0.0, 0.0, 0.0
    gap = sum(positives) / len(positives) - sum(negatives) / len(negatives)
    rng = random.Random(seed)
    samples = []
    for _ in range(2000):
        pos = [rng.choice(positives) for _ in positives]
        neg = [rng.choice(negatives) for _ in negatives]
        samples.append(sum(pos) / len(pos) - sum(neg) / len(neg))
    samples.sort()
    return gap, samples[49], samples[1949]


def summarize_candidate(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    calibration = [row for row in valid if stable_split(row["case_id"]) == "calibration"]
    test = [row for row in valid if stable_split(row["case_id"]) == "test"]
    threshold = select_threshold(calibration)
    gap, low, high = bootstrap_gap(valid, seed)
    by_source = {}
    for source in sorted({row["source"] for row in test}):
        subset = [row for row in test if row["source"] == source]
        by_source[source] = {
            "auc": roc_auc([row["label"] for row in subset], [row["score"] for row in subset]),
            "operating_point": threshold_metrics(subset, threshold),
        }
    return {
        "count": len(valid),
        "positive_rate": sum(row["label"] for row in valid) / max(1, len(valid)),
        "auc": roc_auc([row["label"] for row in valid], [row["score"] for row in valid]),
        "mean_gap": gap,
        "mean_gap_bootstrap_95ci": [low, high],
        "threshold_selected_on_calibration": threshold,
        "calibration": threshold_metrics(calibration, threshold),
        "test": threshold_metrics(test, threshold),
        "by_source_test": by_source,
        "status_counts": dict(Counter(row.get("status", "unknown") for row in rows)),
    }


def summarize(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
    report = {
        candidate: summarize_candidate(
            [row for row in rows if row["candidate"] == candidate], seed
        )
        for candidate in sorted({row["candidate"] for row in rows})
    }
    subgraphs = [
        row
        for row in rows
        if row["candidate"] == "dag_subgraph_attribution" and row.get("status") == "ok"
    ]
    if subgraphs:
        report["dag_subgraph_attribution"]["support_removed_auc"] = roc_auc(
            [row["support_removed"] for row in subgraphs], [row["score"] for row in subgraphs]
        )
        report["dag_subgraph_attribution"]["by_node_type"] = {
            node_type: summarize_candidate(
                [row for row in subgraphs if row["node_type"] == node_type], seed
            )
            for node_type in sorted({row["node_type"] for row in subgraphs})
        }
    return report


def self_test() -> None:
    nodes = [
        {"id": "1", "type": "search", "query": "Who directed Film A?", "conclusion": "Jane Doe"},
        {"id": "2", "type": "search", "query": "Where was #1# born?", "conclusion": "Paris"},
        {"id": "3", "type": "reason", "task": "Compare #1# and #2#", "conclusion": "done"},
    ]
    assert descendant_closure(nodes, "1") == {"1", "2", "3"}
    assert descendant_closure(nodes, "2") == {"2", "3"}
    assert binding_is_expressed(["Jane Doe"], "Where was Jane Doe born?")
    assert not binding_is_expressed(["Jane Doe"], "Where was John Smith born?")
    messages = build_query_binding_messages("system", "goal", "Where was #1# born?", {"1": "Jane Doe"})
    assert "# 1 = Jane Doe" in messages[1]["content"]
    history = render_workflow_history(nodes, {"2": "London"})
    assert "Conclusion: London" in history and "Task 3 [REASON]" in history
    metrics = threshold_metrics(
        [{"label": 1, "score": 2.0}, {"label": 0, "score": -1.0}], 0.0
    )
    assert metrics["precision"] == 1.0 and metrics["recall"] == 1.0
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--model-path")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.trajectories or not args.output_dir or not args.model_path:
        raise SystemExit("--trajectories, --output-dir, and --model-path are required")

    trajectories = load_trajectories(args.trajectories, args.limit, args.seed)
    cases = build_cases(trajectories)
    print(f"[CONFIG] trajectories={len(trajectories)} cases={len(cases)}")
    print(f"[CONFIG] candidates={dict(Counter(case['candidate'] for case in cases))}")
    rows = score_cases(
        cases,
        args.model_path,
        args.batch_size,
        args.max_prompt_tokens,
        args.generation_batch_size,
    )
    report = summarize(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
