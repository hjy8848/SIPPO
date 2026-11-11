#!/usr/bin/env python3
"""Audit gold-document-free counterfactual confidence rewards for DAGENT."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_manager.ROSA.protocol import (  # noqa: E402
    ANSWER_GENERATION_SYSTEM_MSG,
    FINAL_SYSTEM_MSG,
    REASON_SYSTEM_MSG,
)
from qa_manager.ROSA.attribution_reward import (  # noqa: E402
    build_search_attribution_messages,
    build_summary_attribution_messages,
    SEARCH_EMPTY_ATTRIBUTION_THRESHOLD,
    SEARCH_MATCHED_ATTRIBUTION_THRESHOLD,
    SUMMARY_ATTRIBUTION_THRESHOLD,
    select_matched_counterfactual_docs,
    tagged_attribution_target,
)


NO_EVIDENCE = "No relevant external evidence was provided."


def normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def supported(value: str, evidence: Iterable[str]) -> bool:
    needle = normalize(value)
    if not needle or needle in {"unknown", "insufficient information"}:
        return False
    return any(needle in normalize(item) for item in evidence)


def exact_match(value: str, answers: Iterable[str]) -> bool:
    candidate = normalize(value)
    return bool(candidate) and any(candidate == normalize(answer) for answer in answers)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def split_docs(context: str) -> list[str]:
    text = str(context or "").strip()
    starts = list(re.finditer(r"(?m)^\[\d+_\d+\]:", text))
    if not starts:
        return [text] if text else []
    return [
        text[start.start() : starts[index + 1].start() if index + 1 < len(starts) else len(text)].strip()
        for index, start in enumerate(starts)
    ]


def source_name(value: str) -> str:
    text = str(value).casefold()
    if "2wiki" in text:
        return "2wiki"
    if "hotpot" in text:
        return "hotpotqa"
    if "musique" in text:
        return "musique"
    if "bamboogle" in text:
        return "bamboogle"
    return "unknown"


def build_natural_cases(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build unmodified node examples from held-out workflow traces."""
    cases = []
    for row_index, row in enumerate(rows):
        trace = row.get("trace") or row.get("raw_trace") or {}
        if row.get("status") not in {None, "ok"} or trace.get("execution_error"):
            continue
        source = str(row.get("data_source", source_name(row.get("source_file", ""))))
        trajectory_id = f"{source}:{row.get('row_index', row_index)}"
        question = str(row.get("question", trace.get("question", "")))
        answers = [
            str(value)
            for value in as_list(row.get("gold_answers", row.get("gold_answer")))
        ]
        final_answer = str(row.get("final_answer", trace.get("final_answer", "")))
        outcome_em = int(row.get("em", exact_match(final_answer, answers)))
        question_type = str(row.get("question_type", "")).casefold()
        nodes = list(trace.get("nodes_execution") or [])
        by_id = {str(node.get("id", "")): node for node in nodes}
        completed = []
        for node in nodes:
            node_id = str(node.get("id", ""))
            conclusion = str(node.get("conclusion", "")).strip()
            if not conclusion or normalize(conclusion) in {"unknown", "insufficient information"}:
                completed.append(node)
                continue
            if node.get("type") == "search":
                docs = split_docs(node.get("search_context", ""))
                if docs:
                    label = int(supported(conclusion, docs))
                    cases.append(
                        {
                            "case_id": f"natural-search:{trajectory_id}:{node_id}",
                            "kind": "answer_generation_critic",
                            "label": label,
                            "subtype": "natural_grounded" if label else "natural_unsupported",
                            "source_file": source,
                            "trajectory_id": trajectory_id,
                            "em": outcome_em,
                            "spec": {
                                "question": question,
                                "node_query": str(
                                    node.get("executed_query", node.get("resolved_query", node.get("query", "")))
                                ),
                                "docs": docs,
                                "output": conclusion,
                            },
                        }
                    )
            elif (
                node.get("type") == "reason"
                and source_name(source) == "2wiki"
                and question_type in {"comparison", "bridge_comparison"}
            ):
                raw_task = str(node.get("task", node.get("query", "")))
                dependency_ids = re.findall(r"#(\d+)#", raw_task)
                premises = [str(by_id[value].get("conclusion", "")) for value in dependency_ids if value in by_id]
                if len(set(dependency_ids)) >= 2 and premises:
                    label = int(exact_match(conclusion, answers))
                    cases.append(
                        {
                            "case_id": f"natural-reason:{trajectory_id}:{node_id}",
                            "kind": "reasoner_critic",
                            "label": label,
                            "subtype": "natural_gold_exact" if label else "natural_wrong",
                            "source_file": source,
                            "trajectory_id": trajectory_id,
                            "em": outcome_em,
                            "spec": {
                                "task": str(
                                    node.get("resolved_task", node.get("resolved_query", raw_task))
                                ),
                                "premises": premises,
                                "output": conclusion,
                            },
                        }
                    )
            completed.append(node)
        conclusions = [str(node.get("conclusion", "")) for node in completed]
        if final_answer and normalize(final_answer) not in {"unknown", "insufficient information"}:
            label = int(supported(final_answer, conclusions))
            history = str(trace.get("history_summary", ""))
            cases.append(
                {
                    "case_id": f"natural-summary:{trajectory_id}",
                    "kind": "summary_faithfulness",
                    "label": label,
                    "subtype": "natural_supported" if label else "natural_unsupported",
                    "source_file": source,
                    "trajectory_id": trajectory_id,
                    "em": outcome_em,
                    "spec": {"question": question, "history": history, "output": final_answer},
                }
            )
    return cases


def search_messages(spec: dict[str, Any], docs: Sequence[str]) -> list[dict[str, str]]:
    return build_search_attribution_messages(
        ANSWER_GENERATION_SYSTEM_MSG, spec.get("node_query", ""), docs
    )


def reason_messages(task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REASON_SYSTEM_MSG},
        {"role": "user", "content": str(task)},
    ]


def summary_messages(spec: dict[str, Any], history: str) -> list[dict[str, str]]:
    return build_summary_attribution_messages(
        FINAL_SYSTEM_MSG, spec.get("question", ""), history
    )


def mask_reason_task(task: str, premises: Iterable[str]) -> tuple[str, bool]:
    masked = str(task)
    replacements = 0
    for premise in sorted({str(value).strip() for value in premises if str(value).strip()}, key=len, reverse=True):
        masked, count = re.subn(re.escape(premise), "[unknown]", masked, flags=re.I)
        replacements += count
    if replacements == 0:
        return "Complete the reasoning operation using unknown inputs.", False
    return masked, True


def matched_foreign_docs(cases: Sequence[dict[str, Any]], index: int) -> list[str]:
    search_indices = [
        case_index
        for case_index, case in enumerate(cases)
        if case.get("kind") == "answer_generation_critic"
    ]
    position = search_indices.index(index)
    steps = []
    for case_index in search_indices:
        case = cases[case_index]
        spec = case.get("spec") or {}
        steps.append(
            {
                "question": spec.get("question", case.get("case_id", "")),
                "data_source": source_name(case.get("source_file", "")),
                "docs": list(spec.get("docs") or []),
                "output": spec.get("output", ""),
            }
        )
    docs, _ = select_matched_counterfactual_docs(steps, position)
    return docs


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


def build_probes(cases: Sequence[dict[str, Any]], tokenizer: Any, max_prompt_tokens: int) -> tuple[list[Probe], Counter]:
    probes = []
    overflow = Counter()
    for index, case in enumerate(cases):
        spec = case["spec"]
        kind = case["kind"]
        output = str(spec.get("output", "")).strip()
        if not output:
            overflow[index] += 1
            continue
        if kind == "answer_generation_critic":
            variants = {
                "factual": search_messages(spec, spec.get("docs") or []),
                "matched_counterfactual": search_messages(spec, matched_foreign_docs(cases, index)),
                "empty_counterfactual": search_messages(spec, [NO_EVIDENCE]),
            }
            prefix, response = tagged_attribution_target("conclusion", output)
        elif kind == "reasoner_critic":
            masked_task, replacement_ok = mask_reason_task(spec.get("task", ""), spec.get("premises") or [])
            variants = {
                "factual": reason_messages(spec.get("task", "")),
                "masked_counterfactual": reason_messages(masked_task),
            }
            case["premise_mask_replaced"] = replacement_ok
            prefix, response = tagged_attribution_target("conclusion", output)
        elif kind == "summary_faithfulness":
            variants = {
                "factual": summary_messages(spec, spec.get("history", "")),
                "empty_counterfactual": summary_messages(spec, ""),
            }
            prefix, response = tagged_attribution_target("answer", output)
        else:
            continue
        for variant, messages in variants.items():
            try:
                probes.append(
                    encode_probe(
                        tokenizer, index, variant, messages, prefix, response, max_prompt_tokens
                    )
                )
            except ValueError:
                overflow[index] += 1
    return probes, overflow


def score_cases(
    cases: list[dict[str, Any]],
    model_path: str,
    batch_size: int,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, fix_mistral_regex=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    probes, overflow = build_probes(cases, tokenizer, max_prompt_tokens)

    values: dict[int, dict[str, float]] = defaultdict(dict)
    pad_id = int(tokenizer.pad_token_id)
    for offset in range(0, len(probes), batch_size):
        chunk = probes[offset : offset + batch_size]
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
            labels[row_index, start : start + probe.response_length] = ids[
                probe.response_start : probe.response_start + probe.response_length
            ]
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
            shifted = labels[:, 1:]
            valid = shifted.ne(-100)
            token_logprobs = torch.log_softmax(logits, dim=-1).gather(
                -1, shifted.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            mean_logprobs = (token_logprobs * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        for probe, score in zip(chunk, mean_logprobs.tolist(), strict=True):
            values[probe.case_index][probe.variant] = float(score)
        if (offset // batch_size + 1) % 20 == 0 or offset + len(chunk) == len(probes):
            print(f"[SCORE] {min(offset + len(chunk), len(probes))}/{len(probes)}")

    output = []
    for index, case in enumerate(cases):
        row = dict(case)
        scores = values.get(index, {})
        required = {
            "answer_generation_critic": {"factual", "matched_counterfactual", "empty_counterfactual"},
            "reasoner_critic": {"factual", "masked_counterfactual"},
            "summary_faithfulness": {"factual", "empty_counterfactual"},
        }[case["kind"]]
        if overflow[index] or not required.issubset(scores):
            row.update({"status": "prompt_overflow", "scores": scores})
        else:
            if case["kind"] == "answer_generation_critic":
                matched_margin = scores["factual"] - scores["matched_counterfactual"]
                empty_margin = scores["factual"] - scores["empty_counterfactual"]
                primary_margin = matched_margin
                row["candidate_scores"] = {
                    "search_matched_attribution": matched_margin,
                    "search_empty_attribution": empty_margin,
                    "factual_confidence": scores["factual"],
                }
            elif case["kind"] == "reasoner_critic":
                primary_margin = scores["factual"] - scores["masked_counterfactual"]
                row["candidate_scores"] = {
                    "reason_premise_attribution": primary_margin,
                    "factual_confidence": scores["factual"],
                }
            else:
                primary_margin = scores["factual"] - scores["empty_counterfactual"]
                row["candidate_scores"] = {
                    "summary_evidence_attribution": primary_margin,
                    "factual_confidence": scores["factual"],
                }
            row.update(
                {
                    "status": "ok",
                    "variant_logprobs": scores,
                    "primary_margin": primary_margin,
                    "bounded_score": math.tanh(primary_margin),
                }
            )
        output.append(row)
    return output


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = sum(pos > neg for pos in positives for neg in negatives)
    ties = sum(pos == neg for pos in positives for neg in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def stable_split(case_id: str) -> str:
    value = int(hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()[:8], 16)
    return "calibration" if value % 2 == 0 else "test"


def balanced_accuracy(rows: Sequence[dict[str, Any]], threshold: float, key: str) -> float:
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    tpr = sum(row["candidate_scores"][key] > threshold for row in positives) / max(1, len(positives))
    tnr = sum(row["candidate_scores"][key] <= threshold for row in negatives) / max(1, len(negatives))
    return 0.5 * (tpr + tnr)


def select_threshold(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = sorted({float(row["candidate_scores"][key]) for row in rows})
    candidates = [values[0] - 1e-6] + values if values else [0.0]
    return max(candidates, key=lambda threshold: (balanced_accuracy(rows, threshold, key), threshold))


def select_high_precision_threshold(
    rows: Sequence[dict[str, Any]], key: str, minimum_precision: float = 0.9
) -> float | None:
    values = sorted({float(row["candidate_scores"][key]) for row in rows})
    best = None
    for threshold in values:
        selected = [row for row in rows if row["candidate_scores"][key] > threshold]
        if not selected:
            continue
        precision = sum(row["label"] == 1 for row in selected) / len(selected)
        recall = sum(row["label"] == 1 for row in selected) / max(
            1, sum(row["label"] == 1 for row in rows)
        )
        if precision >= minimum_precision and (best is None or recall > best[0]):
            best = (recall, threshold)
    return None if best is None else best[1]


def threshold_metrics(rows: Sequence[dict[str, Any]], threshold: float, key: str) -> dict[str, float]:
    predicted = [row for row in rows if row["candidate_scores"][key] > threshold]
    true_positive = sum(row["label"] == 1 for row in predicted)
    positives = sum(row["label"] == 1 for row in rows)
    negatives = sum(row["label"] == 0 for row in rows)
    true_negative = sum(
        row["label"] == 0 and row["candidate_scores"][key] <= threshold for row in rows
    )
    return {
        "threshold": threshold,
        "balanced_accuracy": 0.5
        * (true_positive / max(1, positives) + true_negative / max(1, negatives)),
        "positive_precision": true_positive / max(1, len(predicted)),
        "positive_recall": true_positive / max(1, positives),
        "positive_rate": len(predicted) / max(1, len(rows)),
    }


def bootstrap_mean_gap(rows: Sequence[dict[str, Any]], key: str, seed: int) -> tuple[float, float, float]:
    positives = [row["candidate_scores"][key] for row in rows if row["label"] == 1]
    negatives = [row["candidate_scores"][key] for row in rows if row["label"] == 0]
    if not positives or not negatives:
        return 0.0, 0.0, 0.0
    observed = sum(positives) / len(positives) - sum(negatives) / len(negatives)
    rng = random.Random(seed)
    samples = []
    for _ in range(2000):
        pos = [rng.choice(positives) for _ in positives]
        neg = [rng.choice(negatives) for _ in negatives]
        samples.append(sum(pos) / len(pos) - sum(neg) / len(neg))
    samples.sort()
    return observed, samples[49], samples[1949]


def summarize_candidate(rows: Sequence[dict[str, Any]], key: str, seed: int) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok" and key in row["candidate_scores"]]
    calibration = [row for row in valid if stable_split(row["case_id"]) == "calibration"]
    test = [row for row in valid if stable_split(row["case_id"]) == "test"]
    threshold = select_threshold(calibration, key)
    high_precision_threshold = select_high_precision_threshold(calibration, key)
    mean_gap, ci_low, ci_high = bootstrap_mean_gap(valid, key, seed)
    report = {
        "count": len(valid),
        "auc": roc_auc([row["label"] for row in valid], [row["candidate_scores"][key] for row in valid]),
        "positive_mean": sum(row["candidate_scores"][key] for row in valid if row["label"] == 1)
        / max(1, sum(row["label"] == 1 for row in valid)),
        "negative_mean": sum(row["candidate_scores"][key] for row in valid if row["label"] == 0)
        / max(1, sum(row["label"] == 0 for row in valid)),
        "mean_gap": mean_gap,
        "mean_gap_bootstrap_95ci": [ci_low, ci_high],
        "calibrated_test": threshold_metrics(test, threshold, key),
        "high_precision_test": (
            threshold_metrics(test, high_precision_threshold, key)
            if high_precision_threshold is not None
            else None
        ),
        "by_source": {},
        "by_subtype": {},
    }
    for source in sorted({source_name(row.get("source_file", "")) for row in valid}):
        subset = [row for row in valid if source_name(row.get("source_file", "")) == source]
        report["by_source"][source] = {
            "count": len(subset),
            "auc": roc_auc(
                [row["label"] for row in subset],
                [row["candidate_scores"][key] for row in subset],
            ),
            "test_at_global_threshold": threshold_metrics(
                [row for row in subset if stable_split(row["case_id"]) == "test"], threshold, key
            ),
        }
    for subtype in sorted({row.get("subtype", "") for row in valid}):
        subset = [row for row in valid if row.get("subtype", "") == subtype]
        report["by_subtype"][subtype] = {
            "count": len(subset),
            "mean": sum(row["candidate_scores"][key] for row in subset) / max(1, len(subset)),
            "positive_rate_at_threshold": sum(
                row["candidate_scores"][key] > threshold for row in subset
            )
            / max(1, len(subset)),
        }
    return report


def add_external_threshold_metrics(
    report: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    threshold_report: dict[str, Any] | None,
) -> None:
    if not threshold_report:
        return
    previous = threshold_report.get("candidates", {})
    for key, candidate_report in report.get("candidates", {}).items():
        source = previous.get(key, {})
        calibrated = source.get("calibrated_test") or {}
        high_precision = source.get("high_precision_test") or {}
        valid = [
            row
            for row in rows
            if row.get("status") == "ok" and key in row.get("candidate_scores", {})
        ]
        if calibrated.get("threshold") is not None:
            candidate_report["external_balanced_threshold"] = threshold_metrics(
                valid, float(calibrated["threshold"]), key
            )
        if high_precision.get("threshold") is not None:
            candidate_report["external_high_precision_threshold"] = threshold_metrics(
                valid, float(high_precision["threshold"]), key
            )


def add_runtime_operating_points(
    report: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> None:
    thresholds = {
        "search_matched_attribution": SEARCH_MATCHED_ATTRIBUTION_THRESHOLD,
        "search_empty_attribution": SEARCH_EMPTY_ATTRIBUTION_THRESHOLD,
        "summary_evidence_attribution": SUMMARY_ATTRIBUTION_THRESHOLD,
    }
    output = {}
    for key, threshold in thresholds.items():
        valid = [
            row
            for row in rows
            if row.get("status") == "ok" and key in row.get("candidate_scores", {})
        ]
        if not valid:
            continue
        test = [row for row in valid if stable_split(row["case_id"]) == "test"]
        by_source = {}
        for source in sorted({source_name(row.get("source_file", "")) for row in test}):
            subset = [
                row
                for row in test
                if source_name(row.get("source_file", "")) == source
            ]
            by_source[source] = threshold_metrics(subset, threshold, key)
        output[key] = {
            "threshold": threshold,
            "all": threshold_metrics(valid, threshold, key),
            "test": threshold_metrics(test, threshold, key),
            "test_by_source": by_source,
        }
    report["runtime_operating_points"] = output


def summarize_planner_hindsight(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
    trajectories: dict[str, dict[str, Any]] = {}
    child_keys = {
        "search_matched_attribution",
        "reason_premise_attribution",
    }
    for row in rows:
        trajectory_id = row.get("trajectory_id")
        if not trajectory_id or row.get("status") != "ok":
            continue
        candidate_scores = row.get("candidate_scores", {})
        child_scores = [
            math.tanh(float(candidate_scores[key]))
            for key in child_keys
            if key in candidate_scores
        ]
        if not child_scores:
            continue
        entry = trajectories.setdefault(
            str(trajectory_id),
            {
                "case_id": f"planner:{trajectory_id}",
                "label": int(row.get("em", 0)),
                "source_file": row.get("source_file", ""),
                "scores": [],
            },
        )
        entry["scores"].extend(child_scores)
    planner_rows = []
    for entry in trajectories.values():
        planner_rows.append(
            {
                "case_id": entry["case_id"],
                "label": entry["label"],
                "source_file": entry["source_file"],
                "status": "ok",
                "candidate_scores": {
                    "planner_child_attribution": sum(entry["scores"]) / len(entry["scores"])
                },
            }
        )
    if not planner_rows or len({row["label"] for row in planner_rows}) < 2:
        return {"count": len(planner_rows), "status": "insufficient_labels"}
    return summarize_candidate(planner_rows, "planner_child_attribution", seed)


def summarize(
    rows: Sequence[dict[str, Any]],
    seed: int,
    threshold_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for row in rows
            if row.get("status") == "ok"
            for key in row.get("candidate_scores", {})
        }
    )
    report = {
        "candidates": {key: summarize_candidate(rows, key, seed) for key in keys},
        "status": dict(Counter(row.get("status", "missing") for row in rows)),
        "acceptance_gate": {
            "dense_reward_auc_min": 0.70,
            "selective_reward_test_precision_min": 0.85,
            "selective_reward_test_recall_min": 0.50,
            "mean_gap_ci_must_exclude_zero": True,
            "must_not_reverse_on_any_dataset_with_both_labels": True,
        },
    }
    add_external_threshold_metrics(report, rows, threshold_report)
    add_runtime_operating_points(report, rows)
    if any(row.get("trajectory_id") for row in rows):
        report["planner_hindsight"] = summarize_planner_hindsight(rows, seed)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--benchmark", type=Path)
    source.add_argument("--trajectories", type=Path)
    source.add_argument(
        "--predictions",
        type=Path,
        help="Rebuild a report from an existing predictions.jsonl without model scoring.",
    )
    parser.add_argument(
        "--threshold-report",
        type=Path,
        help="Controlled-case report whose thresholds are applied unchanged to this audit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.predictions:
        scored = [
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        audit_mode = "resummarized"
    elif args.benchmark:
        cases = json.loads(args.benchmark.read_text(encoding="utf-8"))
        audit_mode = "controlled"
    else:
        rows = json.loads(args.trajectories.read_text(encoding="utf-8"))
        cases = build_natural_cases(rows)
        audit_mode = "natural"
    threshold_report = (
        json.loads(args.threshold_report.read_text(encoding="utf-8"))
        if args.threshold_report
        else None
    )
    if not args.predictions:
        if not args.model_path:
            parser.error("--model-path is required with --benchmark or --trajectories")
        print(f"[CONFIG] mode={audit_mode} cases={len(cases)} model={args.model_path}")
        scored = score_cases(cases, args.model_path, args.batch_size, args.max_prompt_tokens)
    else:
        print(f"[CONFIG] mode={audit_mode} predictions={len(scored)}")
    report = summarize(scored, args.seed, threshold_report)
    report["audit_mode"] = audit_mode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
