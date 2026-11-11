#!/usr/bin/env python3
"""Audit retrieval coverage lost by truncating voucher plans to one Search."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from qa_manager.ROSA.shoppingbench_protocol import extract_atomic_search_queries


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compiler_queries(row: Mapping[str, Any]) -> list[str]:
    trace = row.get("raw_trace") or {}
    protocol = trace.get("contract_protocol") or {}
    compiler = protocol.get("plan_compiler") or {}
    return [str(value) for value in compiler.get("queries") or [] if str(value).strip()]


def raw_queries(row: Mapping[str, Any]) -> list[str]:
    trace = row.get("raw_trace") or {}
    return extract_atomic_search_queries(str(trace.get("raw_plan_output") or ""))


def _counts() -> dict[str, Any]:
    return {
        "rows": 0,
        "expected_positions": 0,
        "current_queries": 0,
        "expanded_queries": 0,
        "expanded_rows": 0,
        "current_corresponding_hits": 0,
        "expanded_corresponding_hits": 0,
        "current_any_hits": 0,
        "expanded_any_hits": 0,
        "current_all_corresponding": 0,
        "expanded_all_corresponding": 0,
        "current_all_any": 0,
        "expanded_all_any": 0,
    }


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in target:
        target[key] += int(source[key])


def _finalize(counts: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(counts["rows"])
    positions = int(counts["expected_positions"])
    result = dict(counts)
    for name in ("corresponding_hits", "any_hits"):
        for arm in ("current", "expanded"):
            result[f"{arm}_{name}_rate"] = counts[f"{arm}_{name}"] / positions if positions else 0.0
    for name in ("all_corresponding", "all_any"):
        for arm in ("current", "expanded"):
            result[f"{arm}_{name}_rate"] = counts[f"{arm}_{name}"] / rows if rows else 0.0
    result["corresponding_recall_delta"] = (
        result["expanded_corresponding_hits_rate"] - result["current_corresponding_hits_rate"]
    )
    result["any_recall_delta"] = result["expanded_any_hits_rate"] - result["current_any_hits_rate"]
    return result


def audit_rows(
    searcher: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    topk: int = 5,
    max_searches: int = 4,
) -> tuple[dict[str, Any], bool]:
    if topk <= 0 or max_searches <= 1:
        raise ValueError("topk must be positive and max_searches must exceed one")
    overall = _counts()
    by_cardinality: dict[int, dict[str, Any]] = defaultdict(_counts)
    raw_count_histogram: Counter[int] = Counter()
    current_count_histogram: Counter[int] = Counter()
    expanded_count_histogram: Counter[int] = Counter()
    prefix_control_passed = True
    identity_control_passed = True
    expanded_from_raw_only = True

    for row in rows:
        expected = [str(value) for value in row.get("expected_product_ids") or []]
        if not expected:
            raise ValueError(f"{row.get('sample_id')}: missing expected product IDs")
        current = compiler_queries(row)
        raw = raw_queries(row)
        expanded = raw[:max_searches]
        raw_count_histogram[len(raw)] += 1
        current_count_histogram[len(current)] += 1
        expanded_count_histogram[len(expanded)] += 1
        prefix_control_passed &= expanded[: len(current)] == current
        expanded_from_raw_only &= expanded == raw[:max_searches]

        current_results = [
            [str(hit.docid) for hit in searcher.search(query, topk)] for query in current
        ]
        expanded_results = [
            [str(hit.docid) for hit in searcher.search(query, topk)] for query in expanded
        ]
        identity_control_passed &= current_results == expanded_results[: len(current_results)]

        counts = _counts()
        counts["rows"] = 1
        counts["expected_positions"] = len(expected)
        counts["current_queries"] = len(current)
        counts["expanded_queries"] = len(expanded)
        counts["expanded_rows"] = int(len(expanded) > len(current))
        current_corresponding = {
            product_id
            for product_id, retrieved in zip(expected, current_results)
            if product_id in retrieved
        }
        expanded_corresponding = {
            product_id
            for product_id, retrieved in zip(expected, expanded_results)
            if product_id in retrieved
        }
        current_union = {value for retrieved in current_results for value in retrieved}
        expanded_union = {value for retrieved in expanded_results for value in retrieved}
        current_any = set(expected) & current_union
        expanded_any = set(expected) & expanded_union
        counts["current_corresponding_hits"] = len(current_corresponding)
        counts["expanded_corresponding_hits"] = len(expanded_corresponding)
        counts["current_any_hits"] = len(current_any)
        counts["expanded_any_hits"] = len(expanded_any)
        counts["current_all_corresponding"] = int(len(current_corresponding) == len(expected))
        counts["expanded_all_corresponding"] = int(len(expanded_corresponding) == len(expected))
        counts["current_all_any"] = int(len(current_any) == len(expected))
        counts["expanded_all_any"] = int(len(expanded_any) == len(expected))
        _merge(overall, counts)
        _merge(by_cardinality[len(expected)], counts)

    finalized = _finalize(overall)
    gates = {
        "shared_query_identity_control": identity_control_passed,
        "current_queries_are_expanded_prefix": prefix_control_passed,
        "expanded_uses_only_model_raw_queries": expanded_from_raw_only,
        "expanded_rows_nonzero": finalized["expanded_rows"] > 0,
        "corresponding_recall_improves": finalized["corresponding_recall_delta"] > 0,
        "any_recall_nonregression": finalized["any_recall_delta"] >= 0,
    }
    report = {
        "schema": "shoppingbench-voucher-compiler-coverage-audit-v1",
        "evidence_level": "offline deterministic retrieval intervention",
        "intervention": "retain up to four model-generated raw Search queries instead of compiler truncation to one",
        "unchanged": ["model_raw_plan", "query_order", "Lucene_index", "topk", "retrieval_code"],
        "gold_usage": "post-hoc scoring only; no query is generated, selected, or reordered using gold",
        "topk": topk,
        "max_searches": max_searches,
        "overall": finalized,
        "by_expected_cardinality": {
            str(key): _finalize(value) for key, value in sorted(by_cardinality.items())
        },
        "query_count_histograms": {
            "raw": {str(key): value for key, value in sorted(raw_count_histogram.items())},
            "current_compiled": {
                str(key): value for key, value in sorted(current_count_histogram.items())
            },
            "expanded": {
                str(key): value for key, value in sorted(expanded_count_histogram.items())
            },
        },
        "gates": gates,
        "passed": all(gates.values()),
        "scope_limit": "retrieval coverage only; descendants were not re-executed and no answer/EM claim is made",
    }
    return report, bool(report["passed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-searches", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    os.environ.setdefault("OPENAI_API_KEY", "unused-local-lucene-import")
    from pyserini.search.lucene import LuceneSearcher

    report, passed = audit_rows(
        LuceneSearcher(str(args.index_path)),
        load_jsonl(args.predictions),
        topk=args.topk,
        max_searches=args.max_searches,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
