#!/usr/bin/env python3
"""Audit ShoppingBench SFT query leakage, diversity, and task coverage."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TASKS = ("product", "shop", "voucher")
REQUIRED_CARDINALITIES = {
    "product": {1},
    "shop": {2, 3, 4},
    "voucher": {1, 2, 3, 4},
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(value: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(value or "").lower()))


def reward_products(row: dict[str, Any]) -> list[dict[str, Any]]:
    reward = row.get("reward")
    if isinstance(reward, dict):
        return [reward]
    if isinstance(reward, list):
        return [item for item in reward if isinstance(item, dict)]
    return []


def product_titles(product: dict[str, Any]) -> list[str]:
    title = product.get("title")
    if isinstance(title, list):
        return [str(value) for value in title if value]
    return [str(title)] if title else []


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * quantile))]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty query set")
    query_tokens: list[int] = []
    exact_title_rows: list[bool] = []
    title_token_recalls: list[float] = []
    prefixes: Counter[str] = Counter()
    cardinalities: Counter[int] = Counter()
    normalized_queries: set[str] = set()
    for row in rows:
        query = normalize(row.get("query"))
        if not query:
            raise ValueError("query is empty")
        tokens = query.split()
        query_set = set(tokens)
        query_tokens.append(len(tokens))
        normalized_queries.add(query)
        prefixes[" ".join(tokens[:5])] += 1
        products = reward_products(row)
        if not products:
            raise ValueError("reward has no product")
        cardinalities[len(products)] += 1
        row_exact: list[bool] = []
        for product in products:
            titles = [normalize(value) for value in product_titles(product)]
            titles = [value for value in titles if value]
            exact = any(title in query for title in titles)
            row_exact.append(exact)
            best_recall = 0.0
            for title in titles:
                title_tokens = set(title.split())
                best_recall = max(best_recall, len(title_tokens & query_set) / len(title_tokens))
            title_token_recalls.append(best_recall)
        exact_title_rows.append(any(row_exact))
    top_prefix, top_prefix_count = prefixes.most_common(1)[0]
    return {
        "rows": len(rows),
        "unique_queries": len(normalized_queries),
        "unique_query_fraction": len(normalized_queries) / len(rows),
        "query_tokens": {
            "mean": statistics.mean(query_tokens),
            "median": statistics.median(query_tokens),
            "p90": percentile(query_tokens, 0.9),
        },
        "any_gold_title_exact_substring_rate": sum(exact_title_rows) / len(rows),
        "mean_gold_title_token_recall_in_query": statistics.mean(title_token_recalls),
        "top_five_token_prefix": top_prefix,
        "top_five_token_prefix_share": top_prefix_count / len(rows),
        "product_cardinality_histogram": {
            str(key): value for key, value in sorted(cardinalities.items())
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_specs(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {task: [] for task in TASKS}
    for row in rows:
        task = str(row.get("task") or "")
        if task not in result:
            raise ValueError(f"unknown task: {task!r}")
        result[task].append(row)
    return result


def audit(
    train_path: Path,
    val_path: Path,
    formal_test_dir: Path,
    *,
    max_exact_title_rate: float,
    max_title_token_recall: float,
    max_prefix_share: float,
) -> tuple[dict[str, Any], bool]:
    train = split_specs(read_jsonl(train_path))
    val = split_specs(read_jsonl(val_path))
    formal = {
        task: read_jsonl(formal_test_dir / f"synthesize_{task}_test.jsonl")
        for task in TASKS
    }
    metrics = {
        task: {
            "synthetic_train": summarize(train[task]),
            "synthetic_val": summarize(val[task]),
            "formal_test_posthoc_only": summarize(formal[task]),
        }
        for task in TASKS
    }
    train_queries = {normalize(row["query"]) for rows in train.values() for row in rows}
    val_queries = {normalize(row["query"]) for rows in val.values() for row in rows}
    formal_queries = {normalize(row["query"]) for rows in formal.values() for row in rows}
    gates = {
        "train_val_query_disjoint": not bool(train_queries & val_queries),
        "formal_query_disjoint": not bool((train_queries | val_queries) & formal_queries),
        "no_exact_title_leakage": all(
            metrics[task][split]["any_gold_title_exact_substring_rate"] <= max_exact_title_rate
            for task in TASKS
            for split in ("synthetic_train", "synthetic_val")
        ),
        "bounded_title_token_recall": all(
            metrics[task][split]["mean_gold_title_token_recall_in_query"] <= max_title_token_recall
            for task in TASKS
            for split in ("synthetic_train", "synthetic_val")
        ),
        "template_diversity": all(
            metrics[task][split]["top_five_token_prefix_share"] <= max_prefix_share
            for task in TASKS
            for split in ("synthetic_train", "synthetic_val")
        ),
        "task_cardinality_coverage": all(
            REQUIRED_CARDINALITIES[task].issubset(
                {
                    int(value)
                    for value in metrics[task]["synthetic_train"]["product_cardinality_histogram"]
                }
            )
            for task in TASKS
        ),
    }
    report = {
        "schema": "shoppingbench-sft-query-alignment-audit-v1",
        "evidence_level": "offline posthoc distribution audit",
        "formal_test_role": "comparison only; never used to generate synthetic queries or select thresholds",
        "thresholds": {
            "max_exact_title_rate": max_exact_title_rate,
            "max_title_token_recall": max_title_token_recall,
            "max_prefix_share": max_prefix_share,
            "required_cardinalities": {
                task: sorted(values) for task, values in REQUIRED_CARDINALITIES.items()
            },
        },
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "scientific_interpretation": (
            "passing is necessary but not sufficient for SFT transfer; retrieval and online grounding gates remain required"
        ),
    }
    return report, bool(report["passed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--formal-test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-exact-title-rate", type=float, default=0.10)
    parser.add_argument("--max-title-token-recall", type=float, default=0.70)
    parser.add_argument("--max-prefix-share", type=float, default=0.25)
    args = parser.parse_args()
    for name in ("max_exact_title_rate", "max_title_token_recall", "max_prefix_share"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    report, passed = audit(
        args.train,
        args.val,
        args.formal_test_dir,
        max_exact_title_rate=args.max_exact_title_rate,
        max_title_token_recall=args.max_title_token_recall,
        max_prefix_share=args.max_prefix_share,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
