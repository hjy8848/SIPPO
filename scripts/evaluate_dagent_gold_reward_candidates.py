#!/usr/bin/env python3
"""Collect held-out DAGENT traces and evaluate deterministic reward candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import random
import re
import string
import threading
from typing import Any, Sequence

import requests


_SESSION_LOCAL = threading.local()


def session() -> requests.Session:
    value = getattr(_SESSION_LOCAL, "value", None)
    if value is None:
        value = requests.Session()
        _SESSION_LOCAL.value = value
    return value


def normalize(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(answer: str, gold: Sequence[str]) -> bool:
    answer_norm = normalize(answer)
    return bool(answer_norm) and any(answer_norm == normalize(item) for item in gold)


def extract_titles(context: str) -> list[str]:
    return [
        match.strip().strip('"')
        for match in re.findall(r"\(Title:\s*(.*?)\)\s", str(context or ""), flags=re.I)
        if match.strip()
    ]


def contains(container: str, value: str) -> bool:
    needle = normalize(value)
    return bool(needle) and needle in normalize(container)


def evaluate_case(case: dict[str, Any], api_base: str, timeout: float) -> dict[str, Any]:
    result = dict(case)
    try:
        response = session().post(
            f"{api_base.rstrip('/')}/rosa/chat",
            json={"query": case["question"]},
            timeout=timeout,
        )
        response.raise_for_status()
        trace = response.json()
        final_answer = str(trace.get("final_answer", ""))
        result.update(
            {
                "status": "ok" if not trace.get("error") else "pipeline_error",
                "final_answer": final_answer,
                "em": int(exact_match(final_answer, case["gold_answers"])),
                "trace": trace,
            }
        )
    except Exception as exc:
        result.update({"status": "request_error", "error": str(exc), "em": 0})
    return result


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace", {})
    gold = {normalize(title) for title in row.get("gold_titles", []) if normalize(title)}
    covered_gold: set[str] = set()
    seen_titles: set[str] = set()
    search_diagnostics = []
    for node in trace.get("nodes_execution", []):
        if node.get("type") != "search":
            continue
        titles = {normalize(title) for title in extract_titles(node.get("search_context", "")) if normalize(title)}
        new_gold = (titles & gold) - covered_gold
        repeated_only = bool(titles) and not (titles - seen_titles)
        conclusion = str(node.get("conclusion", ""))
        search_diagnostics.append(
            {
                "node_id": str(node.get("id", "")),
                "query": str(node.get("executed_query", node.get("resolved_query", node.get("query", "")))),
                "retrieved_titles": sorted(titles),
                "new_gold_titles": sorted(new_gold),
                "gold_title_delta": len(new_gold) / max(1, len(gold)),
                "repeated_titles_only": repeated_only,
                "conclusion": conclusion,
                "span_grounded": contains(node.get("search_context", ""), conclusion),
            }
        )
        covered_gold.update(titles & gold)
        seen_titles.update(titles)
    row = dict(row)
    executed_nodes = trace.get("nodes_execution", [])
    reason_nodes = [node for node in executed_nodes if node.get("type") == "reason"]
    row["reward_diagnostics"] = {
        "gold_title_available": bool(gold),
        "gold_title_coverage": len(covered_gold) / max(1, len(gold)),
        "gold_titles_covered": sorted(covered_gold),
        "search_steps": search_diagnostics,
        "search_count": len(search_diagnostics),
        "positive_gold_delta_steps": sum(step["gold_title_delta"] > 0 for step in search_diagnostics),
        "span_grounded_steps": sum(step["span_grounded"] for step in search_diagnostics),
        "repeated_only_steps": sum(step["repeated_titles_only"] for step in search_diagnostics),
        "reason_count": len(reason_nodes),
        "has_reason": bool(reason_nodes),
    }
    return row


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for source in sorted({row["data_source"] for row in rows}):
        values = [row for row in rows if row["data_source"] == source and row.get("status") == "ok"]
        coverages = [row["reward_diagnostics"]["gold_title_coverage"] for row in values]
        outcomes = [float(row["em"]) for row in values]
        full = [row for row in values if row["reward_diagnostics"]["gold_title_coverage"] >= 0.999]
        partial = [row for row in values if 0 < row["reward_diagnostics"]["gold_title_coverage"] < 0.999]
        zero = [row for row in values if row["reward_diagnostics"]["gold_title_coverage"] == 0]
        steps = [step for row in values for step in row["reward_diagnostics"]["search_steps"]]
        positive_steps = [step for step in steps if step["gold_title_delta"] > 0]
        correct = [row for row in values if row["em"]]
        wrong = [row for row in values if not row["em"]]
        correct_steps = [step for row in correct for step in row["reward_diagnostics"]["search_steps"]]
        wrong_steps = [step for row in wrong for step in row["reward_diagnostics"]["search_steps"]]
        composite_steps = [
            step for step in steps if step["gold_title_delta"] > 0 and step["span_grounded"]
        ]
        reason_by_type = {}
        for question_type in sorted({str(row.get("question_type", "")) for row in values}):
            typed = [row for row in values if str(row.get("question_type", "")) == question_type]
            with_reason = [row for row in typed if row["reward_diagnostics"]["has_reason"]]
            without_reason = [row for row in typed if not row["reward_diagnostics"]["has_reason"]]
            reason_by_type[question_type] = {
                "count": len(typed),
                "reason_rate": safe_div(len(with_reason), len(typed)),
                "with_reason_em": safe_div(sum(row["em"] for row in with_reason), len(with_reason)),
                "without_reason_em": safe_div(sum(row["em"] for row in without_reason), len(without_reason)),
            }
        report[source] = {
            "requested": sum(row["data_source"] == source for row in rows),
            "completed": len(values),
            "em": safe_div(sum(row["em"] for row in values), len(values)),
            "mean_gold_title_coverage": safe_div(sum(coverages), len(coverages)),
            "coverage_em_pearson": pearson(coverages, outcomes),
            "correct_mean_coverage": safe_div(
                sum(row["reward_diagnostics"]["gold_title_coverage"] for row in correct), len(correct)
            ),
            "wrong_mean_coverage": safe_div(
                sum(row["reward_diagnostics"]["gold_title_coverage"] for row in wrong), len(wrong)
            ),
            "full_coverage_count": len(full),
            "full_coverage_em": safe_div(sum(row["em"] for row in full), len(full)),
            "partial_coverage_count": len(partial),
            "partial_coverage_em": safe_div(sum(row["em"] for row in partial), len(partial)),
            "zero_coverage_count": len(zero),
            "zero_coverage_em": safe_div(sum(row["em"] for row in zero), len(zero)),
            "search_steps": len(steps),
            "positive_gold_delta_steps": len(positive_steps),
            "positive_gold_delta_rate": safe_div(len(positive_steps), len(steps)),
            "correct_positive_gold_delta_rate": safe_div(
                sum(step["gold_title_delta"] > 0 for step in correct_steps), len(correct_steps)
            ),
            "wrong_positive_gold_delta_rate": safe_div(
                sum(step["gold_title_delta"] > 0 for step in wrong_steps), len(wrong_steps)
            ),
            "gold_delta_and_span_grounded_rate": safe_div(len(composite_steps), len(steps)),
            "span_grounded_rate": safe_div(sum(step["span_grounded"] for step in steps), len(steps)),
            "repeated_only_rate": safe_div(sum(step["repeated_titles_only"] for step in steps), len(steps)),
            "reason_rate": safe_div(
                sum(row["reward_diagnostics"]["has_reason"] for row in values), len(values)
            ),
            "reason_by_question_type": reason_by_type,
        }
    report["errors"] = dict(Counter(row.get("status", "missing") for row in rows))
    return report


def load_cases(path: Path, sources: set[str], max_per_source: int, seed: int) -> list[dict[str, Any]]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        if not str(row.get("file", "")).startswith("validation_"):
            continue
        source = str(row.get("data_source", ""))
        if source not in sources or not row.get("gold_titles"):
            continue
        buckets[source].append(
            {
                "data_source": source,
                "row_index": row.get("row_index"),
                "question": row.get("question", ""),
                "gold_answers": row.get("gold_answers", []),
                "gold_titles": row.get("gold_titles", []),
                "question_type": row.get("question_type", ""),
            }
        )
    # Older metadata exports did not include answers. Reload them from the
    # source is preferable, but fail loudly instead of producing fake EM.
    missing_answers = sum(not case["gold_answers"] for cases in buckets.values() for case in cases)
    if missing_answers:
        raise ValueError(
            f"gold metadata has {missing_answers} rows without gold_answers; rerun the metadata audit"
        )
    rng = random.Random(seed)
    selected = []
    for source in sorted(buckets):
        rng.shuffle(buckets[source])
        selected.extend(buckets[source][:max_per_source])
    rng.shuffle(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8087")
    parser.add_argument("--sources", default="hotpotqa,musique")
    parser.add_argument("--max-per-source", type=int, default=160)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Recompute diagnostics from output-dir/trajectories.json without calling the model server.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = args.output_dir / "trajectories.json"
    if args.rebuild_only:
        results = [enrich(row) for row in json.loads(trajectories_path.read_text(encoding="utf-8"))]
        report = summarize(results)
        trajectories_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    cases = load_cases(args.metadata, set(args.sources.split(",")), args.max_per_source, args.seed)
    print(f"[CONFIG] cases={len(cases)} concurrency={args.concurrency} sources={args.sources}")
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(evaluate_case, case, args.api_base, args.timeout): case for case in cases}
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(enrich(future.result()))
            if completed % 20 == 0 or completed == len(futures):
                print(f"[PROGRESS] {completed}/{len(futures)}")
    results.sort(key=lambda row: (row["data_source"], int(row.get("row_index", 0))))
    report = summarize(results)
    (args.output_dir / "trajectories.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
