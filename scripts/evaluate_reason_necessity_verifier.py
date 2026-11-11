#!/usr/bin/env python3
"""Validate an SFT model as a verifier for whether a DAG needs a reason node."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import string
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests


REASON_REQUIRED_TYPES = {"comparison", "bridge_comparison"}
REASON_NOT_REQUIRED_TYPES = {"compositional", "inference"}
VALID_PREDICTIONS = {"reason_required", "not_required", "insufficient"}
_SESSION_LOCAL = threading.local()


def _session() -> requests.Session:
    session = getattr(_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _SESSION_LOCAL.session = session
    return session


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
    return [str(value)]


def _extract_question(row: Any) -> str:
    reward_model = row.reward_model if isinstance(row.reward_model, dict) else {}
    question = reward_model.get("question")
    if question:
        return str(question)
    prompt = row.prompt
    if isinstance(prompt, (list, tuple)) or hasattr(prompt, "tolist"):
        messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
        if messages:
            return str(messages[-1].get("content", ""))
    return ""


def load_cases(data_file: Path, samples_per_class: int, seed: int) -> List[Dict[str, Any]]:
    frame = pd.read_parquet(data_file)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        if str(row.data_source).lower() not in {"2wiki", "2wikimultihopqa"}:
            continue
        reward_model = row.reward_model if isinstance(row.reward_model, dict) else {}
        ground_truth = reward_model.get("ground_truth", {})
        question_type = str(ground_truth.get("type", ""))
        if question_type in REASON_REQUIRED_TYPES:
            expected = "reason_required"
        elif question_type in REASON_NOT_REQUIRED_TYPES:
            expected = "not_required"
        else:
            continue
        question = _extract_question(row).strip()
        answers = _as_list(ground_truth.get("target"))
        if not question or not answers:
            continue
        buckets[expected].append(
            {
                "row_index": row_index,
                "question": question,
                "gold_answers": answers,
                "question_type": question_type,
                "topology_type": str(ground_truth.get("topology_type", "")),
                "expected": expected,
            }
        )

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for label in ("reason_required", "not_required"):
        candidates = buckets[label]
        rng.shuffle(candidates)
        selected.extend(candidates[: min(samples_per_class, len(candidates))])
    rng.shuffle(selected)
    return selected


def _search_nodes(trace: Dict[str, Any]) -> List[Dict[str, str]]:
    nodes = []
    for node in trace.get("nodes_execution", []):
        if node.get("type") != "search":
            continue
        conclusion = str(node.get("conclusion", "")).strip()
        if not conclusion:
            continue
        nodes.append(
            {
                "id": str(node.get("id", "")),
                "task": str(node.get("task", node.get("query", ""))),
                "resolved_task": str(
                    node.get("resolved_task", node.get("resolved_query", ""))
                ),
                "conclusion": conclusion,
            }
        )
    return nodes


def _xml_value(tag: str, text: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def normalize_answer(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_matches(response: str, gold_answers: Sequence[str]) -> Tuple[bool, str]:
    answer = _xml_value("answer", response)
    normalized = normalize_answer(answer)
    return any(normalized == normalize_answer(gold) for gold in gold_answers), answer


def build_terminal_branches(nodes: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_id = {str(node["id"]): node for node in nodes}
    dependencies = {
        node_id: [dep for dep in re.findall(r"#\s*(\d+)\s*#", node.get("task", "")) if dep in by_id]
        for node_id, node in by_id.items()
    }
    referenced = {dependency for values in dependencies.values() for dependency in values}
    terminal_ids = [node_id for node_id in by_id if node_id not in referenced]

    def closure(node_id: str, visiting: Optional[set] = None) -> set:
        visiting = set() if visiting is None else set(visiting)
        if node_id in visiting:
            return {node_id}
        visiting.add(node_id)
        result = {node_id}
        for dependency in dependencies.get(node_id, []):
            result.update(closure(dependency, visiting))
        return result

    def sort_key(node_id: str) -> Tuple[int, str]:
        return (int(node_id), node_id) if node_id.isdigit() else (10**9, node_id)

    branches = []
    for terminal_id in sorted(terminal_ids, key=sort_key):
        node_ids = sorted(closure(terminal_id), key=sort_key)
        branches.append(
            {
                "id": f"branch_{terminal_id}",
                "node_ids": node_ids,
                "nodes": [by_id[node_id] for node_id in node_ids],
            }
        )
    return branches


def parse_verifier_response(text: str) -> Dict[str, Any]:
    ids_text = _xml_value("single_sufficient_ids", text).lower()
    single_ids = [] if ids_text in {"", "none", "no", "n/a"} else re.findall(r"\d+", ids_text)
    joint_text = _xml_value("joint_sufficient", text).lower()
    declared = _xml_value("verdict", text).lower()
    confidence_text = _xml_value("confidence", text)
    try:
        confidence = min(1.0, max(0.0, float(confidence_text)))
    except ValueError:
        confidence = None

    joint_sufficient: Optional[bool]
    if joint_text == "yes":
        joint_sufficient = True
    elif joint_text == "no":
        joint_sufficient = False
    else:
        joint_sufficient = None

    if joint_sufficient is False:
        derived = "insufficient"
    elif joint_sufficient is True and single_ids:
        derived = "not_required"
    elif joint_sufficient is True:
        derived = "reason_required"
    else:
        derived = None

    return {
        "single_sufficient_ids": single_ids,
        "joint_sufficient": joint_sufficient,
        "declared_prediction": declared if declared in VALID_PREDICTIONS else None,
        "prediction": derived,
        "confidence": confidence,
        "internally_consistent": derived is not None and declared == derived,
        "parse_ok": derived is not None,
    }


def evaluate_case(
    case: Dict[str, Any], api_base: str, timeout: float, method: str
) -> Dict[str, Any]:
    result = dict(case)
    try:
        pipeline_response = _session().post(
            f"{api_base.rstrip('/')}/rosa/chat",
            json={"query": case["question"]},
            timeout=timeout,
        )
        pipeline_response.raise_for_status()
        trace = pipeline_response.json()
        nodes = _search_nodes(trace)
        result.update(
            {
                "pipeline_error": trace.get("error"),
                "final_answer": trace.get("final_answer", ""),
                "plan_output": trace.get("plan_output", ""),
                "search_nodes": nodes,
            }
        )
        if trace.get("error") or not nodes:
            result.update({"status": "pipeline_failed", "prediction": None})
            return result

        if method == "direct":
            verifier_response = _session().post(
                f"{api_base.rstrip('/')}/rosa/verify-reason-necessity",
                json={
                    "question": case["question"],
                    "gold_answers": case["gold_answers"],
                    "nodes": nodes,
                },
                timeout=timeout,
            )
            verifier_response.raise_for_status()
            raw_verifier = str(verifier_response.json().get("response", ""))
            parsed = parse_verifier_response(raw_verifier)
            result.update(parsed)
            result.update(
                {
                    "status": "ok" if parsed["parse_ok"] else "parse_failed",
                    "raw_verifier_response": raw_verifier,
                    "method": method,
                }
            )
        else:
            branches = build_terminal_branches(nodes)
            evidence_sets = [{"id": "empty", "nodes": []}]
            evidence_sets.extend({"id": item["id"], "nodes": item["nodes"]} for item in branches)
            evidence_sets.append({"id": "joint", "nodes": nodes})
            verifier_response = _session().post(
                f"{api_base.rstrip('/')}/rosa/verify-counterfactual-support",
                json={"question": case["question"], "evidence_sets": evidence_sets},
                timeout=timeout,
            )
            verifier_response.raise_for_status()
            outputs = verifier_response.json().get("outputs", [])
            scored = {}
            for output in outputs:
                matched, answer = answer_matches(str(output.get("response", "")), case["gold_answers"])
                scored[str(output.get("id", ""))] = {
                    "matched_gold": matched,
                    "answer": answer,
                    "response": str(output.get("response", "")),
                }
            empty_matches = bool(scored.get("empty", {}).get("matched_gold"))
            joint_matches = bool(scored.get("joint", {}).get("matched_gold"))
            sufficient_branches = [
                item["id"] for item in branches if scored.get(item["id"], {}).get("matched_gold")
            ]
            if empty_matches:
                prediction = "parametric_leakage"
            elif not joint_matches:
                prediction = "insufficient"
            elif sufficient_branches:
                prediction = "not_required"
            else:
                prediction = "reason_required"
            result.update(
                {
                    "status": "ok",
                    "method": method,
                    "prediction": prediction,
                    "terminal_branches": branches,
                    "sufficient_branches": sufficient_branches,
                    "counterfactual_outputs": scored,
                    "parse_ok": True,
                    "internally_consistent": True,
                }
            )
        return result
    except Exception as exc:
        result.update({"status": "request_failed", "error": str(exc), "prediction": None})
        return result


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [row for row in results if row.get("status") == "ok"]
    binary = [
        row for row in evaluated if row.get("prediction") in {"reason_required", "not_required"}
    ]
    correct = sum(row["prediction"] == row["expected"] for row in binary)
    tp = sum(row["prediction"] == "reason_required" and row["expected"] == "reason_required" for row in binary)
    fp = sum(row["prediction"] == "reason_required" and row["expected"] == "not_required" for row in binary)
    fn = sum(row["prediction"] == "not_required" and row["expected"] == "reason_required" for row in binary)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    by_type = {}
    for question_type in sorted({row["question_type"] for row in results}):
        rows = [row for row in binary if row["question_type"] == question_type]
        by_type[question_type] = {
            "count": len(rows),
            "accuracy": _safe_div(sum(row["prediction"] == row["expected"] for row in rows), len(rows)),
        }

    return {
        "selected_cases": len(results),
        "status_counts": dict(Counter(row.get("status", "unknown") for row in results)),
        "parsed_coverage": _safe_div(len(evaluated), len(results)),
        "insufficient_count": sum(row.get("prediction") == "insufficient" for row in evaluated),
        "parametric_leakage_count": sum(
            row.get("prediction") == "parametric_leakage" for row in evaluated
        ),
        "binary_evaluated": len(binary),
        "accuracy": _safe_div(correct, len(binary)),
        "reason_required_precision": precision,
        "reason_required_recall": recall,
        "reason_required_f1": _safe_div(2 * precision * recall, precision + recall),
        "confusion": {
            "required_as_required": tp,
            "required_as_not_required": fn,
            "not_required_as_required": fp,
            "not_required_as_not_required": sum(
                row["prediction"] == "not_required" and row["expected"] == "not_required" for row in binary
            ),
        },
        "internal_consistency_rate": _safe_div(
            sum(bool(row.get("internally_consistent")) for row in evaluated), len(evaluated)
        ),
        "by_question_type": by_type,
    }


def write_outputs(output_dir: Path, results: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    disagreements = [
        row
        for row in results
        if row.get("status") == "ok"
        and (row.get("prediction") == "insufficient" or row.get("prediction") != row.get("expected"))
    ]
    with (output_dir / "disagreements.jsonl").open("w", encoding="utf-8") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    confusion = summary["confusion"]
    lines = [
        "# SFT Reason-Necessity Verifier Validation",
        "",
        "> This is a silver-label validation. Labels come from held-out 2Wiki official question types, not from the SFT model's generated plan.",
        "",
        f"- Selected cases: {summary['selected_cases']}",
        f"- Parsed coverage: {summary['parsed_coverage']:.2%}",
        f"- Binary evaluated: {summary['binary_evaluated']}",
        f"- Accuracy: {summary['accuracy']:.2%}",
        f"- Required precision: {summary['reason_required_precision']:.2%}",
        f"- Required recall: {summary['reason_required_recall']:.2%}",
        f"- Required F1: {summary['reason_required_f1']:.2%}",
        f"- Insufficient evidence: {summary['insufficient_count']}",
        f"- Empty-evidence/parametric leakage: {summary['parametric_leakage_count']}",
        f"- XML/internal consistency: {summary['internal_consistency_rate']:.2%}",
        "",
        "| Expected / Predicted | Reason required | Not required |",
        "|---|---:|---:|",
        f"| Reason required | {confusion['required_as_required']} | {confusion['required_as_not_required']} |",
        f"| Not required | {confusion['not_required_as_required']} | {confusion['not_required_as_not_required']} |",
        "",
        "## Per Type",
        "",
        "| Question type | Count | Accuracy |",
        "|---|---:|---:|",
    ]
    for question_type, metrics in summary["by_question_type"].items():
        lines.append(f"| {question_type} | {metrics['count']} | {metrics['accuracy']:.2%} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test() -> None:
    nodes = _search_nodes(
        {
            "nodes_execution": [
                {
                    "id": 1,
                    "type": "search",
                    "query": "Who directed Film A?",
                    "resolved_query": "Who directed Film A?",
                    "conclusion": "Director A",
                }
            ]
        }
    )
    assert nodes[0]["task"] == "Who directed Film A?"
    assert nodes[0]["resolved_task"] == "Who directed Film A?"
    branches = build_terminal_branches(
        [
            {"id": "1", "task": "Who directed Film A?", "resolved_task": "", "conclusion": "A"},
            {"id": "2", "task": "When was #1# born?", "resolved_task": "", "conclusion": "1900"},
        ]
    )
    assert len(branches) == 1
    assert branches[0]["node_ids"] == ["1", "2"]
    parsed = parse_verifier_response(
        "<single_sufficient_ids>none</single_sufficient_ids>"
        "<joint_sufficient>yes</joint_sufficient>"
        "<verdict>reason_required</verdict><confidence>0.9</confidence>"
    )
    assert parsed["prediction"] == "reason_required"
    assert parsed["internally_consistent"]
    parsed = parse_verifier_response(
        "<single_sufficient_ids>2</single_sufficient_ids>"
        "<joint_sufficient>yes</joint_sufficient>"
        "<verdict>not_required</verdict><confidence>0.8</confidence>"
    )
    assert parsed["prediction"] == "not_required"
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("exp/analysis_outputs_sft_reason_verifier"))
    parser.add_argument("--api-base", default="http://127.0.0.1:8086")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--method", choices=["direct", "counterfactual"], default="counterfactual")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.data_file is None:
        raise SystemExit("--data-file is required unless --self-test is used")

    cases = load_cases(args.data_file, args.samples_per_class, args.seed)
    print(
        f"[CONFIG] selected={len(cases)} concurrency={args.concurrency} "
        f"method={args.method} api={args.api_base}"
    )
    print(f"[CONFIG] labels={dict(Counter(case['expected'] for case in cases))}")
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(evaluate_case, case, args.api_base, args.timeout, args.method): case
            for case in cases
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 20 == 0 or completed == len(futures):
                print(f"[PROGRESS] {completed}/{len(futures)}")

    results.sort(key=lambda row: row["row_index"])
    summary = summarize(results)
    write_outputs(args.output_dir, results, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] report={args.output_dir / 'report.md'}")
    print(f"[DONE] disagreements={args.output_dir / 'disagreements.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
