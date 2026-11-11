#!/usr/bin/env python3
"""Audit surface-template regularity in ShoppingBench Shop questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


VISIBLE_PRODUCT_ITEM_RE = re.compile(
    r"\((\d+)\)\s*(.*?)(?=;\s*\(\d+\)\s|\.\s*(?:Purchase terms:|\?|$))",
    re.I,
)

OPENING_FAMILIES = (
    ("Find shops offering", re.compile(r"^Find shops offering\b", re.I)),
    ("Find a shop that offers", re.compile(r"^Find a shop that offers\b", re.I)),
    ("Find a shop offering", re.compile(r"^Find a shop offering\b", re.I)),
    ("Find a shop that sells", re.compile(r"^Find a shop that sells\b", re.I)),
    ("Find a shop selling", re.compile(r"^Find a shop selling\b", re.I)),
    ("Looking for a shop that offers", re.compile(r"^Looking for a shop that offers\b", re.I)),
    ("Looking for a shop offering", re.compile(r"^Looking for a shop offering\b", re.I)),
    ("Looking for a shop that sells", re.compile(r"^Looking for a shop that sells\b", re.I)),
    ("I'm looking for a shop that offers", re.compile(r"^I'm looking for a shop that offers\b", re.I)),
    ("I'm looking for a shop that sells", re.compile(r"^I'm looking for a shop that sells\b", re.I)),
    ("I'm looking for a shop", re.compile(r"^I'm looking for a shop\b", re.I)),
    ("Please find one store", re.compile(r"^Please find one store\b", re.I)),
)

WORD_ORDINAL_RE = re.compile(
    r"\b(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth(?:ly)?|lastly)\s*[,.:]",
    re.I,
)
SHARED_MERCHANT_RE = re.compile(
    r"(?:\b(?:one|single|same|common)\s+(?:shop|store|seller)\b|"
    r"\bfrom\s+(?:the\s+)?same\s+(?:shop|store|seller)\b)",
    re.I,
)
COLLECTIVE_MERCHANT_RE = re.compile(
    r"(?:\b(?:shop|store|seller)s?\b.{0,80}\b(?:all|both|together)\b|"
    r"\b(?:all|both)\b.{0,80}\b(?:shop|store|seller)s?\b)",
    re.I,
)


def load_rows(path: Path, task: str | None = None) -> tuple[list[dict[str, Any]], int]:
    lines = path.read_text(encoding="utf-8").split("\n")
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows or any("query" not in row or "reward" not in row for row in rows):
        raise ValueError("expected non-empty JSONL rows with query and reward fields")
    source_rows = len(rows)
    if task is not None:
        rows = [row for row in rows if row.get("task") == task]
        if not rows:
            raise ValueError(f"no rows found for task={task!r}")
    return rows, source_rows


def opening_family(query: str) -> str:
    for label, pattern in OPENING_FAMILIES:
        if pattern.search(query):
            return label
    return "Other"


def slot_style(query: str) -> str:
    if re.search(r"\(1\)", query):
        return "(1)/(2) 编号"
    if WORD_ORDINAL_RE.search(query):
        return "First/Second 序词"
    if re.search(r"\bboth\b", query, re.I):
        return "both 双商品"
    return "无显式 slot 标记"


def current_visible_parser(query: str) -> list[str]:
    text = " ".join(str(query or "").split())
    if "Product details: " not in text:
        return []
    details = text.split("Product details: ", 1)[1]
    return [
        match.group(2).strip()
        for match in VISIBLE_PRODUCT_ITEM_RE.finditer(details)
        if match.group(2).strip()
    ]


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def count_table(counter: Counter[Any], total: int) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count, "percent": round(100 * count / total, 2)}
        for value, count in counter.most_common()
    ]


def audit(path: Path, task: str | None = None) -> dict[str, Any]:
    rows, source_rows = load_rows(path, task)
    queries = [str(row["query"]) for row in rows]
    item_counts = [len(row["reward"]) for row in rows]
    total = len(rows)
    parser_counts = [len(current_visible_parser(query)) for query in queries]
    parenthesized_numbers = [
        [
            int(value)
            for value in re.findall(
                r"(?:Product details:\s*|;\s*)\((\d+)\)\s", query, re.I
            )
        ]
        for query in queries
    ]

    openings = Counter(opening_family(query) for query in queries)
    fixed_prefixes = Counter(
        query.split(" Product details:", 1)[0]
        for query in queries
        if " Product details:" in query
    )
    styles = Counter(slot_style(query) for query in queries)
    explicit_shared = [bool(SHARED_MERCHANT_RE.search(query)) for query in queries]
    collective_shared = [bool(COLLECTIVE_MERCHANT_RE.search(query)) for query in queries]
    signatures = Counter(
        (
            opening_family(query),
            slot_style(query),
            item_count,
            bool(SHARED_MERCHANT_RE.search(query)),
            "Purchase terms:" in query,
        )
        for query, item_count in zip(queries, item_counts)
    )

    chars = [len(query) for query in queries]
    words = [len(query.split()) for query in queries]
    feature_counts = {
        "contains_product_details": sum("Product details:" in query for query in queries),
        "contains_purchase_terms": sum("Purchase terms:" in query for query in queries),
        "contains_parenthesized_numbering": sum(bool(values) for values in parenthesized_numbers),
        "parenthesized_numbering_matches_gold_count": sum(
            values == list(range(1, item_count + 1))
            for values, item_count in zip(parenthesized_numbers, item_counts)
        ),
        "contains_word_ordinals": sum(bool(WORD_ORDINAL_RE.search(query)) for query in queries),
        "contains_both": sum(bool(re.search(r"\bboth\b", query, re.I)) for query in queries),
        "explicit_same_merchant_phrase": sum(explicit_shared),
        "collective_merchant_phrase": sum(collective_shared),
        "current_parser_nonempty": sum(count > 0 for count in parser_counts),
        "current_parser_matches_gold_count": sum(
            parsed == item_count for parsed, item_count in zip(parser_counts, item_counts)
        ),
    }

    return {
        "dataset": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_rows": source_rows,
            "selected_task": task,
            "rows": total,
            "exact_duplicate_queries": total - len(set(queries)),
            "unique_queries": len(set(queries)),
        },
        "feature_counts": {
            key: {"count": count, "percent": round(100 * count / total, 2)}
            for key, count in feature_counts.items()
        },
        "item_count_distribution": count_table(Counter(item_counts), total),
        "slot_style_distribution": count_table(styles, total),
        "opening_family_distribution": count_table(openings, total),
        "fixed_prefix_before_product_details_distribution": count_table(
            fixed_prefixes, total
        ),
        "length": {
            "characters": {
                "min": min(chars),
                "p25": round(percentile(chars, 0.25), 1),
                "median": round(statistics.median(chars), 1),
                "mean": round(statistics.mean(chars), 1),
                "p75": round(percentile(chars, 0.75), 1),
                "max": max(chars),
            },
            "whitespace_tokens": {
                "min": min(words),
                "p25": round(percentile(words, 0.25), 1),
                "median": round(statistics.median(words), 1),
                "mean": round(statistics.mean(words), 1),
                "p75": round(percentile(words, 0.75), 1),
                "max": max(words),
            },
        },
        "coarse_signatures": {
            "definition": "opening_family + slot_style + gold_item_count + explicit_same_merchant + purchase_terms",
            "unique": len(signatures),
            "top10_coverage": {
                "count": sum(count for _, count in signatures.most_common(10)),
                "percent": round(
                    100 * sum(count for _, count in signatures.most_common(10)) / total,
                    2,
                ),
            },
            "distribution": [
                {
                    "signature": list(signature),
                    "count": count,
                    "percent": round(100 * count / total, 2),
                }
                for signature, count in signatures.most_common()
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset, args.task), ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
