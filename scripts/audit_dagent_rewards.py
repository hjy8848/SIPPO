#!/usr/bin/env python3
"""Build and score contrastive audits for DAGENT's candidate rewards.

The semantic critics are evaluated exactly as they are used during RL: the
frozen SFT model scores the fixed continuations ``yes`` and ``no`` and the
higher normalized probability becomes the verdict.  The benchmark is built
from held-out inference traces and uses high-precision silver labels plus
controlled corruptions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import glob
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalize(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _as_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _exact_match(answer: str, gold: Sequence[str]) -> bool:
    normalized = _normalize(answer)
    return bool(normalized) and any(normalized == _normalize(item) for item in gold)


def _supported(answer: str, evidence: Iterable[str]) -> bool:
    answer_norm = _normalize(answer)
    if not answer_norm or answer_norm in {"unknown", "insufficient information"}:
        return False
    for item in evidence:
        item_norm = _normalize(item)
        if answer_norm == item_norm:
            return True
        if len(answer_norm) >= 4 and re.search(rf"(?<!\w){re.escape(answer_norm)}(?!\w)", item_norm):
            return True
    return False


def _xml_value(tag: str, text: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", str(text), flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def _history(nodes: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for node in nodes:
        task = node.get("resolved_task", node.get("resolved_query", node.get("task", node.get("query", ""))))
        blocks.append(
            f"Task {node.get('id', '')} [{str(node.get('type', '')).upper()}]: {task}\n"
            f"Conclusion: {node.get('conclusion', '')}"
        )
    return "\n\n".join(blocks)


def _split_docs(context: str) -> list[str]:
    context = str(context or "").strip()
    if not context:
        return []
    starts = list(re.finditer(r"(?m)^\[\d+_\d+\]:", context))
    if not starts:
        return [context]
    docs = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(context)
        docs.append(context[start.start():end].strip())
    return docs


def _node_dependencies(node: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    raw = str(node.get("task", node.get("query", "")))
    return [
        str(by_id[match].get("conclusion", ""))
        for match in re.findall(r"#(\d+)#", raw)
        if match in by_id
    ]


def build_critic_prompt(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Mirror RayPPOTrainer._node_reward_critic_prompt exactly."""
    kind = spec.get("kind")
    later_history = str(spec.get("full_history", ""))[-3000:]
    retrospective = (
        f"Final answer produced: {spec.get('final_answer', '')}\n"
        f"Gold answer: {', '.join(map(str, spec.get('answers', [])))}\n"
        f"Later workflow context:\n{later_history}\n"
    )
    if kind == "answer_generation_critic":
        docs_text = "\n".join(str(doc) for doc in spec.get("docs", []))[:6000]
        if spec.get("action") == "rewrite":
            criterion = (
                "Are the documents insufficient for the current sub-question, so that issuing the proposed "
                "rewrite and searching again is justified?"
            )
        else:
            criterion = (
                "Is the proposed conclusion directly supported by the documents and does it answer the "
                "current sub-question?"
            )
        body = (
            f"Current sub-question: {spec.get('node_query', '')}\n"
            f"Documents:\n{docs_text}\n"
            f"Agent action: {spec.get('action', '')}\n"
            f"Agent output: {spec.get('output', '')}\n"
            f"{retrospective}{criterion}"
        )
    elif kind == "reasoner_critic":
        premises = "\n".join(str(value) for value in spec.get("premises", []))
        body = (
            f"Reasoning task: {spec.get('task', '')}\n"
            f"Available premises:\n{premises}\n"
            f"Reasoner conclusion: {spec.get('output', '')}\n"
            f"{retrospective}Is the conclusion entailed by the premises and does it complete the task?"
        )
    elif kind == "summary_faithfulness":
        body = (
            f"Question: {spec.get('question', '')}\n"
            f"Intermediate conclusions:\n{str(spec.get('history', ''))[-6000:]}\n"
            f"Final answer: {spec.get('output', '')}\n"
            "Is the final answer supported by the intermediate conclusions?"
        )
    else:
        raise ValueError(f"unsupported critic kind: {kind}")
    return [
        {"role": "system", "content": "You are a strict factual verifier. Answer exactly yes or no."},
        {"role": "user", "content": body},
    ]


def _load_trace_rows(patterns: Sequence[str]) -> list[dict[str, Any]]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    rows = []
    for path in sorted(set(paths)):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in payload:
            row = dict(row)
            row["_source_file"] = Path(path).name
            rows.append(row)
    return rows


def _base_spec(row: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": str(row.get("question", trace.get("question", ""))),
        "answers": _as_answers(row.get("gold_answer", trace.get("gold_answers", []))),
        "final_answer": str(row.get("pred_answer", trace.get("final_answer", ""))),
        "full_history": str(trace.get("history_summary", "")),
    }


def _case(kind: str, label: int, subtype: str, spec: dict[str, Any], row: dict[str, Any], serial: int) -> dict[str, Any]:
    return {
        "case_id": f"{kind}-{serial:05d}",
        "kind": kind,
        "label": int(label),
        "subtype": subtype,
        "source_file": row.get("_source_file", ""),
        "sample_index": row.get("sample_index", row.get("source_index")),
        "spec": spec,
    }


def build_cases(rows: Sequence[dict[str, Any]], max_per_label: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pools: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    grounding_positives: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    reason_positives: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    summary_positives: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for row in rows:
        trace = row.get("raw_trace") or row.get("trace") or {}
        if trace.get("execution_error") or row.get("pipeline_error") or row.get("error"):
            continue
        nodes = list(trace.get("nodes_execution") or [])
        if not nodes:
            continue
        base = _base_spec(row, trace)
        by_id = {str(node.get("id")): node for node in nodes}
        completed: list[dict[str, Any]] = []
        for node in nodes:
            conclusion = str(node.get("conclusion", "")).strip()
            if not conclusion:
                continue
            if node.get("type") == "search":
                docs = _split_docs(node.get("search_context", ""))
                if docs and _supported(conclusion, docs):
                    spec = dict(base)
                    spec.update(
                        {
                            "kind": "answer_generation_critic",
                            "history": _history(completed),
                            "node_query": str(node.get("resolved_task", node.get("resolved_query", node.get("task", node.get("query", ""))))),
                            "docs": docs,
                            "action": "conclusion",
                            "output": conclusion,
                        }
                    )
                    grounding_positives.append((row, node, spec))
            elif node.get("type") == "reason":
                gold = base["answers"]
                verifier_ok = str(node.get("reason_verifier", "")).lower() == "supported"
                outcome_ok = _exact_match(base["final_answer"], gold) and _supported(base["final_answer"], [conclusion])
                if verifier_ok or outcome_ok:
                    spec = dict(base)
                    spec.update(
                        {
                            "kind": "reasoner_critic",
                            "history": _history(completed),
                            "task": str(node.get("resolved_task", node.get("resolved_query", node.get("task", node.get("query", ""))))),
                            "premises": _node_dependencies(node, by_id),
                            "output": conclusion,
                        }
                    )
                    reason_positives.append((row, node, spec))
            completed.append(node)

        conclusions = [str(node.get("conclusion", "")) for node in nodes]
        if _exact_match(base["final_answer"], base["answers"]) and _supported(base["final_answer"], conclusions):
            spec = {
                "kind": "summary_faithfulness",
                "question": base["question"],
                "history": _history(nodes),
                "output": base["final_answer"],
            }
            summary_positives.append((row, spec))

    rng.shuffle(grounding_positives)
    rng.shuffle(reason_positives)
    rng.shuffle(summary_positives)

    def foreign_output(current: str, forbidden_text: str, candidates: Sequence[str]) -> str | None:
        options = list(candidates)
        rng.shuffle(options)
        for candidate in options:
            if _normalize(candidate) != _normalize(current) and not _supported(candidate, [forbidden_text]):
                return candidate
        return None

    grounding_outputs = [item[2]["output"] for item in grounding_positives]
    reason_outputs = [item[2]["output"] for item in reason_positives]
    summary_outputs = [item[1]["output"] for item in summary_positives]
    serial = 0

    for row, _node, spec in grounding_positives:
        serial += 1
        pools[("answer_generation_critic", 1)].append(_case("answer_generation_critic", 1, "supported_original", spec, row, serial))
        docs_text = "\n".join(spec["docs"])
        wrong = foreign_output(spec["output"], docs_text, grounding_outputs + summary_outputs)
        if wrong:
            negative = dict(spec, output=wrong)
            serial += 1
            pools[("answer_generation_critic", 0)].append(_case("answer_generation_critic", 0, "foreign_conclusion", negative, row, serial))
        gold = next((answer for answer in spec["answers"] if not _supported(answer, spec["docs"])), None)
        if gold and not _exact_match(gold, [spec["output"]]):
            negative = dict(spec, output=gold)
            serial += 1
            pools[("answer_generation_critic", 0)].append(_case("answer_generation_critic", 0, "gold_leak_trap", negative, row, serial))

    for row, _node, spec in reason_positives:
        serial += 1
        pools[("reasoner_critic", 1)].append(_case("reasoner_critic", 1, "verified_original", spec, row, serial))
        output_norm = _normalize(spec["output"])
        if output_norm == "yes":
            wrong = "no"
        elif output_norm == "no":
            wrong = "yes"
        else:
            wrong = foreign_output(spec["output"], "\n".join(spec["premises"]), reason_outputs + summary_outputs)
        if wrong:
            negative = dict(spec, output=wrong)
            serial += 1
            pools[("reasoner_critic", 0)].append(_case("reasoner_critic", 0, "wrong_conclusion", negative, row, serial))

    for row, spec in summary_positives:
        serial += 1
        pools[("summary_faithfulness", 1)].append(_case("summary_faithfulness", 1, "grounded_original", spec, row, serial))
        wrong = foreign_output(spec["output"], spec["history"], summary_outputs + grounding_outputs)
        if wrong:
            negative = dict(spec, output=wrong)
            serial += 1
            pools[("summary_faithfulness", 0)].append(_case("summary_faithfulness", 0, "foreign_answer", negative, row, serial))
        empty = dict(spec, history="")
        serial += 1
        pools[("summary_faithfulness", 0)].append(_case("summary_faithfulness", 0, "empty_evidence_trap", empty, row, serial))

    selected = []
    for key in sorted(pools):
        values = pools[key]
        rng.shuffle(values)
        selected.extend(values[:max_per_label])
    rng.shuffle(selected)
    return selected


def deterministic_audits(allow_parser_skip: bool = False) -> dict[str, Any]:
    from qa_manager.compute_metrics import compute_scores

    em_cases = [
        ("The Beatles", ["Beatles"], 1),
        ("April 3, 1885", ["April 3 1885"], 1),
        ("yes", ["Yes"], 1),
        ("no", ["yes"], 0),
        ("Paris, France", ["Paris"], 0),
        ("", ["Paris"], 0),
    ]
    em_pass = 0
    em_details = []
    for prediction, answers, expected in em_cases:
        actual = int(compute_scores([prediction], [answers])["em"])
        em_pass += actual == expected
        em_details.append({"prediction": prediction, "answers": answers, "expected": expected, "actual": actual})

    try:
        from qa_manager.ROSA.rosa_agent_manager import ROSA_AgentManager
    except ModuleNotFoundError:
        if not allow_parser_skip:
            raise
        return {
            "final_em": {"passed": em_pass, "total": len(em_cases), "accuracy": em_pass / len(em_cases), "details": em_details},
            "invalid_action": {"passed": 0, "total": 0, "accuracy": 0.0, "details": [], "status": "deferred_until_score_host"},
        }

    manager = ROSA_AgentManager.__new__(ROSA_AgentManager)
    parser_cases = [
        ("PlanningAgent", '<plan><search id="1">Who directed Film A?</search><search id="2">When was #1# born?</search></plan>', True),
        ("PlanningAgent", '<search id="1">Who directed Film A?</search>', False),
        ("PlanningAgent", '<plan><search id="1">Who directed Film A?</search><search id="1">When was #1# born?</search></plan>', False),
        ("PlanningAgent", '<plan><search id="1">Who directed Film A?</search><search id="2">When was #3# born?</search></plan>', False),
        ("PlanningAgent", '<plan><reason id="1">Compare values</reason></plan>', False),
        ("PlanningAgent", 'explanation <plan><search id="1">Who directed Film A?</search></plan>', False),
        ("AnswerGenerationAgent", '<conclusion>John Smith</conclusion>', True),
        ("AnswerGenerationAgent", 'explanation <conclusion>John Smith</conclusion>', False),
        ("AnswerGenerationAgent", '<conclusion>John</conclusion><rewrite>John biography</rewrite>', False),
        ("ReasonerAgent", '<conclusion>yes</conclusion>', True),
        ("ReasonerAgent", 'because facts <conclusion>yes</conclusion>', False),
        ("AnswerSummarizationAgent", '<answer>Paris</answer>', True),
        ("AnswerSummarizationAgent", 'The answer is <answer>Paris</answer>', False),
        ("QueryRewriteAgent", 'Ada Lovelace husband', True),
        ("QueryRewriteAgent", '#1# husband', False),
        ("QueryRewriteAgent", 'search query', False),
    ]
    parser_pass = 0
    parser_details = []
    for agent, response, expected in parser_cases:
        kwargs = {"max_nodes": 5, "question": "When was the director of Film A born?"} if agent == "PlanningAgent" else {}
        actual = bool(manager.parse_response(agent, response, **kwargs).get("is_valid"))
        parser_pass += actual == expected
        parser_details.append({"agent": agent, "response": response, "expected": expected, "actual": actual})
    return {
        "final_em": {"passed": em_pass, "total": len(em_cases), "accuracy": em_pass / len(em_cases), "details": em_details},
        "invalid_action": {"passed": parser_pass, "total": len(parser_cases), "accuracy": parser_pass / len(parser_cases), "details": parser_details},
    }


@dataclass
class EncodedProbe:
    input_ids: list[int]
    response_start: int
    response_length: int


def _encode_probe(tokenizer: Any, messages: list[dict[str, str]], answer: str, max_prompt_tokens: int) -> EncodedProbe:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_tokens:
        raise ValueError(f"critic prompt has {len(prompt_ids)} tokens; training limit is {max_prompt_tokens}")
    response_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not response_ids:
        raise ValueError(f"empty tokenization for critic answer {answer!r}")
    return EncodedProbe(prompt_ids + response_ids, len(prompt_ids), len(response_ids))


def score_cases(cases: list[dict[str, Any]], model_path: str, batch_size: int, max_prompt_tokens: int) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    probes: list[tuple[int, str, EncodedProbe]] = []
    overflow = Counter()
    for index, case in enumerate(cases):
        messages = build_critic_prompt(case["spec"])
        for answer in ("yes", "no"):
            try:
                probes.append((index, answer, _encode_probe(tokenizer, messages, answer, max_prompt_tokens)))
            except ValueError:
                overflow[index] += 1

    scores: dict[int, dict[str, float]] = defaultdict(dict)
    pad_id = int(tokenizer.pad_token_id)
    for offset in range(0, len(probes), batch_size):
        chunk = probes[offset:offset + batch_size]
        width = max(len(item.input_ids) for _, _, item in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device="cuda")
        attention = torch.zeros((len(chunk), width), dtype=torch.long, device="cuda")
        labels = torch.full((len(chunk), width), -100, dtype=torch.long, device="cuda")
        for row_index, (_case_index, _answer, item) in enumerate(chunk):
            padding = width - len(item.input_ids)
            values = torch.tensor(item.input_ids, dtype=torch.long, device="cuda")
            input_ids[row_index, padding:] = values
            attention[row_index, padding:] = 1
            start = padding + item.response_start
            labels[row_index, start:start + item.response_length] = values[item.response_start:]
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
            shifted_labels = labels[:, 1:]
            valid = shifted_labels.ne(-100)
            gathered = torch.log_softmax(logits, dim=-1).gather(
                -1, shifted_labels.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            sequence_scores = (gathered * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        for (case_index, answer, _item), value in zip(chunk, sequence_scores.tolist(), strict=True):
            scores[case_index][answer] = float(value)

    output = []
    for index, case in enumerate(cases):
        row = dict(case)
        if overflow[index] or set(scores[index]) != {"yes", "no"}:
            row.update({"status": "prompt_overflow", "prediction": None, "confidence": None})
        else:
            yes_score = scores[index]["yes"]
            no_score = scores[index]["no"]
            maximum = max(yes_score, no_score)
            yes_prob = math.exp(yes_score - maximum)
            no_prob = math.exp(no_score - maximum)
            probability_yes = yes_prob / (yes_prob + no_prob)
            row.update(
                {
                    "status": "ok",
                    "yes_logprob": yes_score,
                    "no_logprob": no_score,
                    "probability_yes": probability_yes,
                    "prediction": int(probability_yes >= 0.5),
                    "confidence": max(probability_yes, 1.0 - probability_yes),
                    "raw_score": 2.0 * probability_yes - 1.0,
                }
            )
        output.append(row)
    return output


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def summarize_scored(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind and row.get("status") == "ok"]
        confident = [row for row in subset if float(row["confidence"]) >= threshold]
        correct = [row for row in confident if row["prediction"] == row["label"]]
        predicted_no = [row for row in confident if row["prediction"] == 0]
        actual_no = [row for row in confident if row["label"] == 0]
        actual_yes = [row for row in confident if row["label"] == 1]
        by_subtype = {}
        for subtype in sorted({row["subtype"] for row in subset}):
            values = [row for row in subset if row["subtype"] == subtype]
            selected = [row for row in values if row["confidence"] >= threshold]
            by_subtype[subtype] = {
                "count": len(values),
                "coverage": _safe_div(len(selected), len(values)),
                "selective_accuracy": _safe_div(sum(row["prediction"] == row["label"] for row in selected), len(selected)),
                "predicted_yes_rate": _safe_div(sum(row["prediction"] == 1 for row in selected), len(selected)),
            }
        summary[kind] = {
            "count": len(subset),
            "coverage": _safe_div(len(confident), len(subset)),
            "selective_accuracy": _safe_div(len(correct), len(confident)),
            "positive_recall": _safe_div(sum(row["prediction"] == 1 for row in actual_yes), len(actual_yes)),
            "negative_recall": _safe_div(sum(row["prediction"] == 0 for row in actual_no), len(actual_no)),
            "penalty_precision": _safe_div(sum(row["label"] == 0 for row in predicted_no), len(predicted_no)),
            "mean_confidence": _safe_div(sum(row["confidence"] for row in subset), len(subset)),
            "by_subtype": by_subtype,
        }
    return summary


def write_report(path: Path, deterministic: dict[str, Any], semantic: dict[str, Any] | None, case_count: int, threshold: float) -> None:
    lines = ["# DAGENT Reward Audit", "", f"Semantic cases: {case_count}", f"Critic confidence threshold: {threshold:.2f}", "", "## Deterministic rewards", ""]
    for name, result in deterministic.items():
        lines.append(f"- **{name}**: {result['passed']}/{result['total']} ({result['accuracy']:.2%})")
        failures = [item for item in result["details"] if item["actual"] != item["expected"]]
        for failure in failures:
            lines.append(f"  - FAIL: `{failure}`")
    if semantic is not None:
        lines.extend(["", "## Semantic critics", "", "| Reward | Cases | Coverage | Selective accuracy | Positive recall | Negative recall | Penalty precision |", "|---|---:|---:|---:|---:|---:|---:|"])
        for kind, result in semantic.items():
            lines.append(
                f"| {kind} | {result['count']} | {result['coverage']:.2%} | {result['selective_accuracy']:.2%} | "
                f"{result['positive_recall']:.2%} | {result['negative_recall']:.2%} | {result['penalty_precision']:.2%} |"
            )
        lines.extend(["", "### Stress cases", ""])
        for kind, result in semantic.items():
            lines.append(f"**{kind}**")
            for subtype, values in result["by_subtype"].items():
                lines.append(
                    f"- `{subtype}`: n={values['count']}, coverage={values['coverage']:.2%}, "
                    f"selective_accuracy={values['selective_accuracy']:.2%}, predicted_yes={values['predicted_yes_rate']:.2%}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "score", "all"), default="all")
    parser.add_argument("--trace-glob", action="append", default=[])
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--max-per-label", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deterministic = deterministic_audits(allow_parser_skip=args.mode == "prepare")
    cases: list[dict[str, Any]]
    if args.mode in {"prepare", "all"}:
        if not args.trace_glob:
            raise ValueError("--trace-glob is required in prepare/all mode")
        trace_rows = _load_trace_rows(args.trace_glob)
        cases = build_cases(trace_rows, args.max_per_label, args.seed)
        args.benchmark.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[PREPARE] traces={len(trace_rows)} cases={len(cases)} benchmark={args.benchmark}")
    else:
        cases = json.loads(args.benchmark.read_text(encoding="utf-8"))

    semantic = None
    if args.mode in {"score", "all"}:
        if not args.model_path:
            raise ValueError("--model-path is required in score/all mode")
        scored = score_cases(cases, args.model_path, args.batch_size, args.max_prompt_tokens)
        semantic = summarize_scored(scored, args.confidence_threshold)
        (args.output_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8"
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps({"deterministic": deterministic, "semantic": semantic}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(semantic, ensure_ascii=False, indent=2))
    write_report(args.output_dir / "report.md", deterministic, semantic, len(cases), args.confidence_threshold)
    print(f"[DONE] {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
