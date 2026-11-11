"""Score a stable official ShoppingBench JSONL against a frozen ASC report."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_stable_jsonl(
    path: Path,
    *,
    expected_rows: int,
    settle_seconds: float = 3.0,
    timeout_seconds: float = 60.0,
) -> list[Any]:
    deadline = time.monotonic() + timeout_seconds
    previous: tuple[int, int] | None = None
    while time.monotonic() < deadline:
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").split("\n")
                if line
            ]
        except json.JSONDecodeError:
            rows = []
        if signature == previous and len(rows) == expected_rows:
            return rows
        previous = signature
        time.sleep(settle_seconds)
    raise RuntimeError(
        f"JSONL did not become stable with {expected_rows} valid rows: {path}"
    )


def product_ids(value: object) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def raw_tool_calls(content: object) -> list[dict[str, Any]]:
    marker = "<tool_call>"
    if not isinstance(content, str) or marker not in content:
        return []
    payload = content.split(marker, 1)[1].lstrip()
    start = payload.find("[")
    if start < 0:
        return []
    try:
        calls, _ = json.JSONDecoder().raw_decode(payload[start:])
    except json.JSONDecodeError:
        return []
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def collect_product_shops(trajectory: Iterable[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            product_id = str(value.get("product_id") or "")
            shop_id = str(value.get("shop_id") or "")
            if product_id.isdigit() and shop_id.isdigit():
                mapping[product_id] = shop_id
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for step in trajectory:
        for message in step.get("prompt") or []:
            content = message.get("content") if isinstance(message, dict) else ""
            for match in re.finditer(r"<obs>(.*?)</obs>", content or "", re.S):
                try:
                    visit(json.loads(match.group(1)))
                except json.JSONDecodeError:
                    continue
    return mapping


def score(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    matched = sum(
        left == right for left, right in itertools.zip_longest(expected, predicted)
    )
    return {
        "matched_product_ids": matched,
        "expected_product_count": len(expected),
        "exact_id_fraction": matched / len(expected) if expected else 0.0,
        "exact_sequence_success": predicted == expected,
    }


def summarize(
    rows: list[dict[str, Any]], ids_key: str, score_key: str
) -> dict[str, Any]:
    exact = sum(row[score_key]["exact_sequence_success"] for row in rows)
    return {
        "questions": len(rows),
        "exact_sequences": exact,
        "exact_sequence_rate": exact / len(rows),
        "mean_exact_id_fraction": sum(
            row[score_key]["exact_id_fraction"] for row in rows
        )
        / len(rows),
        "nonempty_predictions": sum(bool(row[ids_key]) for row in rows),
    }


def analyze(
    trajectories: list[list[dict[str, Any]]],
    references: dict[str, dict[str, Any]],
    *,
    retry_count: int,
) -> dict[str, Any]:
    rows = []
    for trajectory in trajectories:
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("empty official trajectory")
        query = trajectory[0]["extra_info"]["query"]
        if query not in references:
            raise ValueError(f"unexpected or duplicate official query: {query}")
        reference = references[query]
        counts: collections.Counter[str] = collections.Counter()
        calls = []
        strict_ids: list[str] = []
        recovered_ids: list[str] = []
        missing_page_calls = 0
        raw_final_content = ""
        for step in trajectory:
            completion = step.get("completion") or {}
            content = completion.get("content") or ""
            if content:
                raw_final_content = content
            message = completion.get("message") or {}
            structured = message.get("tool_call") or []
            parsed_calls = structured or raw_tool_calls(content)
            for call in parsed_calls:
                name = call.get("name")
                params = call.get("parameters") or {}
                counts[name] += 1
                calls.append(
                    {
                        "step": step["extra_info"]["step"],
                        "name": name,
                        "parameters": params,
                        "strictly_parsed": bool(structured),
                    }
                )
                if name == "find_product" and "page" not in params:
                    missing_page_calls += 1
                if name == "recommend_product":
                    ids = product_ids(params.get("product_ids"))
                    if structured:
                        strict_ids = ids
                    else:
                        recovered_ids = ids
        visible_shops = collect_product_shops(trajectory)
        diagnostic_ids = strict_ids or recovered_ids
        diagnostic_shops = [visible_shops.get(product_id) for product_id in diagnostic_ids]
        same_shop = (
            bool(diagnostic_shops)
            and None not in diagnostic_shops
            and len(set(diagnostic_shops)) == 1
        )
        expected = [str(value) for value in reference["expected_product_ids"]]
        rows.append(
            {
                "sample_id": reference["sample_id"],
                "query": query,
                "expected_product_ids": expected,
                "cp_shop_product_ids": reference["baseline_product_ids"],
                "cp_shop_score": reference["baseline_score"],
                "asc_shop_product_ids": reference["probe"]["predicted_product_ids"],
                "asc_shop_score": reference["probe_score"],
                "official_strict_product_ids": strict_ids,
                "official_strict_score": score(expected, strict_ids),
                "official_diagnostic_recovered_product_ids": recovered_ids,
                "official_diagnostic_recovered_score": score(expected, recovered_ids),
                "official_format_recovery_needed": bool(recovered_ids and not strict_ids),
                "official_diagnostic_shop_ids": diagnostic_shops,
                "official_diagnostic_same_shop": same_shop,
                "official_missing_page_calls": missing_page_calls,
                "official_steps": len(trajectory),
                "official_tool_counts": dict(sorted(counts.items())),
                "official_tool_calls": calls,
                "official_raw_final_content": raw_final_content,
            }
        )
    rows.sort(key=lambda row: row["sample_id"])
    missing_page_questions = sum(bool(row["official_missing_page_calls"]) for row in rows)
    summary = {
        "cp_shop": summarize(rows, "cp_shop_product_ids", "cp_shop_score"),
        "asc_shop": summarize(rows, "asc_shop_product_ids", "asc_shop_score"),
        "official_strict": summarize(
            rows, "official_strict_product_ids", "official_strict_score"
        ),
        "official_diagnostic_recovered": summarize(
            rows,
            "official_diagnostic_recovered_product_ids",
            "official_diagnostic_recovered_score",
        ),
        "official_format_recovery_needed": sum(
            row["official_format_recovery_needed"] for row in rows
        ),
        "official_diagnostic_same_shop": sum(
            row["official_diagnostic_same_shop"] for row in rows
        ),
        "official_missing_page_calls": sum(
            row["official_missing_page_calls"] for row in rows
        ),
        "official_missing_page_questions": missing_page_questions,
        "official_max_step_questions": sum(row["official_steps"] >= 30 for row in rows),
        "official_tool_retry_log_count": retry_count,
        "pipeline_materialization_errors": 0,
        "tool_protocol_error_questions": missing_page_questions,
        "strict_scoring_rule": "only parsed official recommend_product calls count",
        "diagnostic_rule": (
            "unterminated raw tool-call recovery is separate and never promoted "
            "to strict success"
        ),
    }
    return {
        "schema": "shoppingbench-shop16-official-vs-asc-qwen25-v2",
        "summary": summary,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=16)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    trajectories = read_stable_jsonl(
        args.run_dir / "rollout_shop.jsonl",
        expected_rows=args.expected_rows,
        settle_seconds=args.settle_seconds,
    )
    reference_rows = [
        json.loads(line)
        for line in (args.run_dir / "shop16_scoring_reference.jsonl")
        .read_text(encoding="utf-8")
        .split("\n")
        if line
    ]
    references = {row["question"]: row for row in reference_rows}
    if len(references) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} unique references")
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    report = analyze(
        trajectories,
        references,
        retry_count=log_text.count("Parse lazada search response error"),
    )
    report["evidence"] = {
        "rollout_sha256": sha256(args.run_dir / "rollout_shop.jsonl"),
        "reference_sha256": sha256(args.run_dir / "shop16_scoring_reference.jsonl"),
        "asc_report_sha256": sha256(args.run_dir / "asc" / "report.json"),
        "log_sha256": sha256(args.log),
        "analyzer_sha256": sha256(Path(__file__)),
        "stable_row_count": len(trajectories),
        "external_shopping_api_calls": 0,
        "gold_runtime_access": False,
        "recovery_reason": (
            "official concurrent runner returned before the final JSONL append "
            "became parse-stable; no trajectory was rerun"
        ),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
