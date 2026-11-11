"""Audit ShoppingBench gold coverage in the local Lucene index."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_FILES = {
    "product": "synthesize_product_test.jsonl",
    "shop": "synthesize_shop_test.jsonl",
    "voucher": "synthesize_voucher_test.jsonl",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def reward_items(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    reward = row.get("reward")
    values = reward if isinstance(reward, list) else [reward]
    return [value for value in values if isinstance(value, Mapping)]


def gold_product_ids(row: Mapping[str, Any]) -> list[str]:
    values = []
    for reward in reward_items(row):
        product_id = str(reward.get("product_id") or "").strip()
        if product_id.isdigit():
            values.append(product_id)
    return list(dict.fromkeys(values))


def raw_question_queries(row: Mapping[str, Any]) -> list[str]:
    query = " ".join(str(row.get("query") or "").split()).strip()
    return [query] if query else []


def gold_title_oracle_queries(row: Mapping[str, Any]) -> list[str]:
    """Return privileged title queries used only as an index-addressability bound."""
    queries = []
    for reward in reward_items(row):
        titles = reward.get("title")
        if isinstance(titles, str):
            titles = [titles]
        if not isinstance(titles, list):
            continue
        for title in titles:
            normalized = " ".join(str(title or "").split()).strip()
            if normalized:
                queries.append(normalized)
    return list(dict.fromkeys(queries))


def retrieved_ids_by_k(
    searcher: Any, queries: Iterable[str], topks: Sequence[int]
) -> tuple[dict[int, set[str]], int]:
    product_ids = {topk: set() for topk in topks}
    calls = 0
    for query in queries:
        calls += 1
        hits = [str(hit.docid) for hit in searcher.search(query, topks[-1])]
        for topk in topks:
            product_ids[topk].update(hits[:topk])
    return product_ids, calls


def _new_mode_counts() -> dict[str, Any]:
    return {
        "rows": 0,
        "gold_products": 0,
        "retrieval_calls": 0,
        "empty_query_rows": 0,
        "hits": defaultdict(int),
        "rows_with_any": defaultdict(int),
        "rows_with_all": defaultdict(int),
    }


def _finalize_mode(counts: Mapping[str, Any], topks: Sequence[int]) -> dict[str, Any]:
    rows = int(counts["rows"])
    gold_products = int(counts["gold_products"])
    metrics = {
        "rows": rows,
        "gold_products": gold_products,
        "retrieval_calls": int(counts["retrieval_calls"]),
        "empty_query_rows": int(counts["empty_query_rows"]),
    }
    for topk in topks:
        suffix = f"@{topk}"
        metrics[f"gold_recall{suffix}"] = (
            counts["hits"][topk] / gold_products if gold_products else 0.0
        )
        metrics[f"rows_with_any_gold{suffix}"] = (
            counts["rows_with_any"][topk] / rows if rows else 0.0
        )
        metrics[f"rows_with_all_gold{suffix}"] = (
            counts["rows_with_all"][topk] / rows if rows else 0.0
        )
    return metrics


def audit_rows(
    searcher: Any,
    rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    topks: Sequence[int] = (3, 5, 10),
) -> dict[str, Any]:
    topks = tuple(sorted(set(int(value) for value in topks)))
    if not topks or topks[0] < 1:
        raise ValueError("topks must contain positive integers")

    modes = {
        "raw_question": raw_question_queries,
        "gold_title_oracle": gold_title_oracle_queries,
    }
    dataset_reports: dict[str, Any] = {}
    overall_mode_counts = {mode: _new_mode_counts() for mode in modes}
    direct_docs = 0
    total_gold = 0

    for dataset, rows in rows_by_dataset.items():
        mode_counts = {mode: _new_mode_counts() for mode in modes}
        dataset_direct_docs = 0
        dataset_gold = 0
        for row in rows:
            gold = set(gold_product_ids(row))
            if not gold:
                raise ValueError(f"{dataset}: row without a numeric gold product ID")
            dataset_gold += len(gold)
            dataset_direct_docs += sum(searcher.doc(product_id) is not None for product_id in gold)

            for mode, query_builder in modes.items():
                queries = query_builder(row)
                counts = mode_counts[mode]
                counts["rows"] += 1
                counts["gold_products"] += len(gold)
                counts["empty_query_rows"] += not queries
                retrieved_by_k, calls = retrieved_ids_by_k(searcher, queries, topks)
                counts["retrieval_calls"] += calls
                for topk, retrieved in retrieved_by_k.items():
                    found = gold & retrieved
                    counts["hits"][topk] += len(found)
                    counts["rows_with_any"][topk] += bool(found)
                    counts["rows_with_all"][topk] += found == gold

        dataset_reports[dataset] = {
            "rows": len(rows),
            "gold_products": dataset_gold,
            "direct_gold_doc_coverage": dataset_direct_docs / dataset_gold,
            "modes": {
                mode: _finalize_mode(counts, topks)
                for mode, counts in mode_counts.items()
            },
        }
        direct_docs += dataset_direct_docs
        total_gold += dataset_gold
        for mode in modes:
            target = overall_mode_counts[mode]
            source = mode_counts[mode]
            for key in (
                "rows",
                "gold_products",
                "retrieval_calls",
                "empty_query_rows",
            ):
                target[key] += source[key]
            for key in ("hits", "rows_with_any", "rows_with_all"):
                for topk in topks:
                    target[key][topk] += source[key][topk]

    return {
        "contract": {
            "raw_question_uses_gold": False,
            "gold_title_oracle_uses_gold": True,
            "gold_title_oracle_interpretation": "index addressability only",
            "external_api_calls": 0,
            "topks": list(topks),
        },
        "datasets": dataset_reports,
        "overall": {
            "rows": sum(len(rows) for rows in rows_by_dataset.values()),
            "gold_products": total_gold,
            "direct_gold_doc_coverage": direct_docs / total_gold if total_gold else 0.0,
            "modes": {
                mode: _finalize_mode(counts, topks)
                for mode, counts in overall_mode_counts.items()
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topks", type=int, nargs="+", default=[3, 5, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {args.output}")
    rows_by_dataset = {
        dataset: load_rows(args.data_dir / filename)
        for dataset, filename in DATASET_FILES.items()
    }

    # Pyserini imports optional OpenAI encoders eagerly even for local BM25.
    os.environ.setdefault("OPENAI_API_KEY", "unused-local-lucene-import")
    from pyserini.search.lucene import LuceneSearcher

    report = audit_rows(LuceneSearcher(str(args.index_path)), rows_by_dataset, args.topks)
    report["inputs"] = {
        "data_dir": str(args.data_dir),
        "index_path": str(args.index_path),
        "dataset_files": DATASET_FILES,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
