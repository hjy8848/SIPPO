#!/usr/bin/env python3
"""Audit query-only retrieval coverage for disjoint ShoppingBench SFT specs."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.audit_shoppingbench_local_retrieval import gold_product_ids, load_rows


TASKS = ("product", "shop", "voucher")
SPLITS = ("train", "val")
_SHOP_ITEM_RE = re.compile(r"\((\d+)\)\s*(.*?)(?=;\s*\(\d+\)\s|\.\s*$)")
_PRODUCT_DETAILS_MARKER = "Product details: "


def atomic_queries_from_visible_query(row: Mapping[str, Any]) -> list[str]:
    """Recover template-level product descriptions without reading reward metadata."""
    task = str(row.get("task") or "").strip()
    query = " ".join(str(row.get("query") or "").split()).strip()
    if _PRODUCT_DETAILS_MARKER in query:
        details = query.split(_PRODUCT_DETAILS_MARKER, 1)[1]
        if task == "voucher":
            details = details.split(". Purchase terms: ", 1)[0]
            if details.startswith("(1) "):
                details += "."
        if task == "shop" or (task == "voucher" and details.startswith("(1) ")):
            values = [match.group(2) for match in _SHOP_ITEM_RE.finditer(details)]
        elif task in {"product", "voucher"}:
            values = [details.removesuffix(".")]
        else:
            raise ValueError(f"unsupported ShoppingBench task: {task!r}")
    elif task == "product":
        prefix = "Find this product in the catalog: "
        values = [query[len(prefix) :].removesuffix(".")] if query.startswith(prefix) else []
    elif task == "voucher":
        marker = ". My budget is "
        if query.startswith("Find this product: ") and marker in query:
            prefix = "Find this product: "
            values = [query[len(prefix) :].split(marker, 1)[0]]
        elif query.startswith("Find these products: ") and marker in query:
            prefix = "Find these products: "
            details = query[len(prefix) :].split(marker, 1)[0] + "."
            values = [match.group(2) for match in _SHOP_ITEM_RE.finditer(details)]
        else:
            values = []
    elif task == "shop":
        prefix = "Find one shop that sells all of these items: "
        values = (
            [match.group(2) for match in _SHOP_ITEM_RE.finditer(query[len(prefix) :])]
            if query.startswith(prefix)
            else []
        )
    else:
        raise ValueError(f"unsupported ShoppingBench task: {task!r}")
    return [" ".join(value.split()).strip() for value in values if value.strip()]


def _new_counts(topks: Sequence[int]) -> dict[str, Any]:
    return {
        "rows": 0,
        "gold_products": 0,
        "retrieval_calls": 0,
        "direct_docs": 0,
        "corresponding_hits": defaultdict(int),
        "any_node_hits": defaultdict(int),
        "rows_with_all_corresponding": defaultdict(int),
        "rows_with_all_any_node": defaultdict(int),
        "topks": tuple(topks),
    }


def _merge_counts(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in ("rows", "gold_products", "retrieval_calls", "direct_docs"):
        target[key] += int(source[key])
    for key in (
        "corresponding_hits",
        "any_node_hits",
        "rows_with_all_corresponding",
        "rows_with_all_any_node",
    ):
        for topk in target["topks"]:
            target[key][topk] += int(source[key][topk])


def _finalize(counts: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(counts["rows"])
    products = int(counts["gold_products"])
    report: dict[str, Any] = {
        "rows": rows,
        "gold_products": products,
        "retrieval_calls": int(counts["retrieval_calls"]),
        "direct_gold_doc_coverage": counts["direct_docs"] / products if products else 0.0,
    }
    for topk in counts["topks"]:
        report[f"corresponding_gold_recall@{topk}"] = (
            counts["corresponding_hits"][topk] / products if products else 0.0
        )
        report[f"any_node_gold_recall@{topk}"] = (
            counts["any_node_hits"][topk] / products if products else 0.0
        )
        report[f"rows_with_all_corresponding@{topk}"] = (
            counts["rows_with_all_corresponding"][topk] / rows if rows else 0.0
        )
        report[f"rows_with_all_any_node@{topk}"] = (
            counts["rows_with_all_any_node"][topk] / rows if rows else 0.0
        )
    return report


def _validate_spec(row: Mapping[str, Any], expected_split: str) -> tuple[list[str], list[str]]:
    if row.get("split") != expected_split:
        raise ValueError(f"{row.get('spec_id')}: split mismatch")
    if row.get("task") not in TASKS:
        raise ValueError(f"{row.get('spec_id')}: invalid task")
    if row.get("planner_input_fields") != ["query"] or row.get("gold_runtime_access") is not False:
        raise ValueError(f"{row.get('spec_id')}: runtime leakage contract failed")
    source_ids = [str(value) for value in row.get("source_product_ids") or []]
    reward_ids = gold_product_ids(row)
    if not source_ids or source_ids != reward_ids or not all(value.isdigit() for value in source_ids):
        raise ValueError(f"{row.get('spec_id')}: source/reward product IDs disagree")
    queries = atomic_queries_from_visible_query(row)
    if len(queries) != len(source_ids):
        raise ValueError(
            f"{row.get('spec_id')}: {len(queries)} query atoms for {len(source_ids)} products"
        )
    if any(product_id in query for product_id in source_ids for query in queries):
        raise ValueError(f"{row.get('spec_id')}: product ID leaked into a query atom")
    return queries, source_ids


def audit_rows(
    searcher: Any,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    topks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    topks = tuple(sorted(set(int(value) for value in topks)))
    if not topks or topks[0] < 1:
        raise ValueError("topks must contain positive integers")
    overall = _new_counts(topks)
    split_reports: dict[str, Any] = {}
    seen_spec_ids: set[str] = set()

    for split in SPLITS:
        task_counts = {task: _new_counts(topks) for task in TASKS}
        for row in rows_by_split.get(split, []):
            spec_id = str(row.get("spec_id") or "")
            if not spec_id or spec_id in seen_spec_ids:
                raise ValueError(f"missing or duplicate spec_id: {spec_id!r}")
            seen_spec_ids.add(spec_id)
            queries, gold_ids = _validate_spec(row, split)
            counts = task_counts[str(row["task"])]
            counts["rows"] += 1
            counts["gold_products"] += len(gold_ids)
            counts["direct_docs"] += sum(searcher.doc(value) is not None for value in gold_ids)
            ranked_ids: list[list[str]] = []
            for query in queries:
                counts["retrieval_calls"] += 1
                ranked_ids.append(
                    [str(hit.docid) for hit in searcher.search(query, topks[-1])]
                )
            for topk in topks:
                corresponding = {
                    gold_id
                    for gold_id, retrieved in zip(gold_ids, ranked_ids, strict=True)
                    if gold_id in retrieved[:topk]
                }
                union = {value for retrieved in ranked_ids for value in retrieved[:topk]}
                any_node = set(gold_ids) & union
                counts["corresponding_hits"][topk] += len(corresponding)
                counts["any_node_hits"][topk] += len(any_node)
                counts["rows_with_all_corresponding"][topk] += len(corresponding) == len(gold_ids)
                counts["rows_with_all_any_node"][topk] += len(any_node) == len(gold_ids)

        split_total = _new_counts(topks)
        for counts in task_counts.values():
            _merge_counts(split_total, counts)
            _merge_counts(overall, counts)
        split_reports[split] = {
            "overall": _finalize(split_total),
            "by_task": {task: _finalize(task_counts[task]) for task in TASKS},
        }

    return {
        "contract": {
            "schema": "shoppingbench-cp-sft-spec-retrieval-audit-v1",
            "query_builder_reads": ["task", "query"],
            "query_builder_reads_gold": False,
            "gold_used_for_scoring_only": True,
            "external_api_calls": 0,
            "topks": list(topks),
        },
        "splits": split_reports,
        "overall": _finalize(overall),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 3, 5, 10])
    return parser.parse_args()


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows_by_split = {
        split: load_rows(args.spec_dir / f"{split}.jsonl") for split in SPLITS
    }
    os.environ.setdefault("OPENAI_API_KEY", "unused-local-lucene-import")
    from pyserini.search.lucene import LuceneSearcher

    report = audit_rows(LuceneSearcher(str(args.index_path)), rows_by_split, args.topks)
    report["inputs"] = {
        "spec_dir": str(args.spec_dir),
        "index_path": str(args.index_path),
    }
    write_report(args.output, report)
    print(json.dumps(report["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
