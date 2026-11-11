"""Score ShoppingBench workflows with the official ASR/CAR contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable


TASKS = ("product", "shop", "voucher")


def read_jsonl(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_official_product_ids(trajectory: list[dict[str, Any]]) -> list[str]:
    product_ids = ""
    for step in trajectory:
        message = ((step.get("completion") or {}).get("message") or {})
        for call in message.get("tool_call") or []:
            if call.get("name") == "recommend_product":
                value = (call.get("parameters") or {}).get("product_ids", "")
                product_ids = value if isinstance(value, str) else ""
    # Keep upstream's exact comma split: whitespace after a comma makes that ID invalid.
    return product_ids.split(",") if product_ids else []


def official_tool_counts(trajectory: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for step in trajectory:
        message = ((step.get("completion") or {}).get("message") or {})
        for call in message.get("tool_call") or []:
            name = call.get("name")
            if isinstance(name, str) and name:
                counts[name] += 1
    return counts


def load_official_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for trajectory in read_jsonl(path):
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"Invalid official trajectory in {path}")
        query = str(((trajectory[0].get("extra_info") or {}).get("query")) or "")
        if not query or query in predictions:
            raise ValueError(f"Missing or duplicate query in {path}: {query!r}")
        predictions[query] = {
            "product_ids": extract_official_product_ids(trajectory),
            "steps": len(trajectory),
            "tool_counts": dict(official_tool_counts(trajectory)),
        }
    return predictions


def load_cp_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        query = str(row.get("query") or "")
        if not query or query in predictions:
            raise ValueError(f"Missing or duplicate query in {path}: {query!r}")
        predictions[query] = {
            "product_ids": [str(item) for item in row.get("predicted_product_ids") or []],
            "steps": len(((row.get("raw_trace") or {}).get("nodes_execution")) or []),
            "tool_counts": {
                "find_product": int(row.get("retrieval_calls") or 0),
            },
        }
    return predictions


class ProductScorer:
    def __init__(
        self,
        searcher: Any,
        embedding_model: Any | None = None,
        official_rule_score: Callable[[dict[str, Any], dict[str, Any]], tuple[Any, ...]] | None = None,
    ):
        self.searcher = searcher
        self.embedding_model = embedding_model
        self.official_rule_score = official_rule_score
        self.cache: dict[str, dict[str, Any] | None] = {}

    def product(self, product_id: str) -> dict[str, Any] | None:
        if product_id not in self.cache:
            document = self.searcher.doc(product_id)
            self.cache[product_id] = (
                json.loads(document.raw())["product"] if document else None
            )
        return self.cache[product_id]

    def title_matches(self, actual: str, expected: str) -> bool:
        if self.embedding_model is None:
            raise RuntimeError("Reference scorer requires an embedding model")
        embeddings = self.embedding_model.encode([actual, expected])
        similarity = self.embedding_model.similarity(embeddings, embeddings)[0][1]
        return float(similarity) >= 0.5

    def relevance(self, product: dict[str, Any], reward: dict[str, Any]) -> float:
        if self.official_rule_score is not None:
            return float(self.official_rule_score(product, reward)[0])
        if str(product["product_id"]) == str(reward["product_id"]):
            return 1.0

        total = 0
        hit = 0
        for title in reward.get("title") or []:
            total += 1
            hit += self.title_matches(str(product["title"]), str(title))

        for price_range in reward.get("price") or []:
            for mode, bounds in price_range.items():
                lower, upper = bounds
                price = product["price"]
                total += 1
                if mode == "less than":
                    hit += price <= upper
                elif mode == "greater than":
                    hit += price >= lower
                elif mode == "between":
                    hit += lower <= price <= upper
                else:
                    raise ValueError(f"Unknown price mode: {mode}")

        for service in reward.get("service") or []:
            total += 1
            hit += service in (product.get("service") or [])

        sku_variants = [set()]
        for option in (product.get("sku_options") or {}).values():
            sku_variants.append(set(option.items()))
        attributes = {
            (key, value)
            for key, values in (product.get("attributes") or {}).items()
            for value in values
        }
        required = []
        for option in reward.get("sku_options") or []:
            required.extend(option.items())
        for attribute in reward.get("attributes") or []:
            required.extend(
                (key, value) for key, values in attribute.items() for value in values
            )
        if required:
            total += len(required)
            hit += max(
                sum(pair in variant or pair in attributes for pair in required)
                for variant in sku_variants
            )

        if total == 0:
            raise ValueError(f"Reward has no scorable constraint: {reward}")
        return hit / total


def load_official_rule_score(official_root: Path) -> tuple[Callable[..., Any], Path]:
    orm_path = official_root / "src" / "agent" / "rewards" / "orm.py"
    if not orm_path.is_file():
        raise FileNotFoundError(f"Missing official evaluator: {orm_path}")
    spec = importlib.util.spec_from_file_location("shoppingbench_frozen_orm", orm_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official evaluator: {orm_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rule_score = getattr(module, "rule_score_reward", None)
    if not callable(rule_score):
        raise AttributeError(f"Missing rule_score_reward in {orm_path}")
    return rule_score, orm_path


def normalize_rewards(task: str, reward: Any) -> list[dict[str, Any]]:
    if task == "product" and isinstance(reward, dict):
        reward = [reward]
    if not isinstance(reward, list) or not reward:
        raise ValueError(f"Invalid {task} reward: {reward!r}")
    return reward


def budget_success(
    products: list[dict[str, Any] | None], voucher: dict[str, Any]
) -> bool:
    if any(product is None for product in products):
        return False
    concrete = [product for product in products if product is not None]
    total_price = sum(float(product["price"]) for product in concrete)
    if total_price <= voucher["budget"]:
        return True
    shop_ids = {str(product["shop_id"]) for product in concrete}
    applicable = voucher["voucher_type"] == "platform" or (
        voucher["voucher_type"] == "shop" and len(shop_ids) == 1
    )
    if not applicable or total_price < voucher["threshold"]:
        return False
    if voucher["discount_type"] == "fixed":
        discounted = total_price - voucher["face_value"]
    elif voucher["discount_type"] == "percentage":
        discounted = max(
            total_price * (1 - voucher["discount"]), total_price - voucher["cap"]
        )
    else:
        raise ValueError(f"Unknown discount type: {voucher['discount_type']}")
    return discounted <= voucher["budget"]


def score_row(
    task: str,
    dataset_row: dict[str, Any],
    prediction: dict[str, Any] | None,
    scorer: ProductScorer,
) -> dict[str, Any]:
    rewards = normalize_rewards(task, dataset_row["reward"])
    product_ids = list((prediction or {}).get("product_ids") or [])
    products = [
        scorer.product(product_ids[index]) if index < len(product_ids) else None
        for index in range(len(rewards))
    ]
    relevance = [
        scorer.relevance(product, reward) if product is not None else 0.0
        for product, reward in zip(products, rewards)
    ]
    car = mean(relevance)
    if task == "product":
        constraint = True
    elif task == "shop":
        shop_ids = {
            str(product["shop_id"]) for product in products if product is not None
        }
        constraint = len(products) == len(rewards) and None not in products and len(shop_ids) == 1
    else:
        constraint = budget_success(products, dataset_row["voucher"])
    return {
        "query": dataset_row["query"],
        "predicted_product_ids": product_ids,
        "car": car,
        "asr": float(math.isclose(car, 1.0) and constraint),
        "constraint_success": bool(constraint),
        "steps": int((prediction or {}).get("steps") or 0),
        "tool_counts": dict((prediction or {}).get("tool_counts") or {}),
        "missing_prediction": prediction is None,
    }


def bootstrap_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tools: Counter[str] = Counter()
    for row in rows:
        tools.update(row["tool_counts"])
    return {
        "rows": len(rows),
        "asr": mean(row["asr"] for row in rows),
        "car": mean(row["car"] for row in rows),
        "mean_steps": mean(row["steps"] for row in rows),
        "tool_calls": dict(sorted(tools.items())),
        "missing_predictions": sum(row["missing_prediction"] for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        choices=["Qwen/Qwen3-Embedding-0.6B"],
        help="Frozen upstream model name; loading is performed by official orm.py.",
    )
    parser.add_argument("--official-rollout-dir", type=Path, required=True)
    parser.add_argument("--cp-results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(args.index))
    official_rule_score, orm_path = load_official_rule_score(args.official_root)
    scorer = ProductScorer(searcher, official_rule_score=official_rule_score)
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        "official": {}, "cp_dagent": {}
    }
    inputs: dict[str, Any] = {
        "datasets": {},
        "predictions": {},
        "official_evaluator": {
            "path": str(orm_path),
            "sha256": file_sha256(orm_path),
            "function": "rule_score_reward",
            "embedding_model": args.embedding_model,
        },
    }

    for task in TASKS:
        dataset_path = args.official_root / "data" / f"synthesize_{task}_test.jsonl"
        official_path = args.official_rollout_dir / f"rollout_{task}.jsonl"
        cp_path = args.cp_results_dir / task / "results.jsonl"
        datasets = read_jsonl(dataset_path)
        if len(datasets) != 250:
            raise ValueError(f"Expected 250 {task} rows, got {len(datasets)}")
        official = load_official_predictions(official_path)
        cp = load_cp_predictions(cp_path)
        inputs["datasets"][task] = {
            "path": str(dataset_path), "sha256": file_sha256(dataset_path)
        }
        inputs["predictions"][task] = {
            "official": {"path": str(official_path), "sha256": file_sha256(official_path)},
            "cp_dagent": {"path": str(cp_path), "sha256": file_sha256(cp_path)},
        }
        for arm, predictions in (("official", official), ("cp_dagent", cp)):
            all_rows[arm][task] = [
                score_row(task, row, predictions.get(row["query"]), scorer)
                for row in datasets
            ]

    summaries = {
        arm: {
            "by_task": {task: summarize(tasks[task]) for task in TASKS},
            "overall": summarize([row for task in TASKS for row in tasks[task]]),
        }
        for arm, tasks in all_rows.items()
    }
    paired: dict[str, Any] = {}
    for scope in (*TASKS, "overall"):
        official_rows = (
            all_rows["official"][scope]
            if scope in TASKS else [row for task in TASKS for row in all_rows["official"][task]]
        )
        cp_rows = (
            all_rows["cp_dagent"][scope]
            if scope in TASKS else [row for task in TASKS for row in all_rows["cp_dagent"][task]]
        )
        paired[scope] = {}
        for metric in ("asr", "car"):
            differences = [cp[metric] - official[metric] for official, cp in zip(official_rows, cp_rows)]
            paired[scope][metric] = {
                "cp_minus_official": mean(differences),
                "paired_bootstrap_95_ci": bootstrap_ci(
                    differences, samples=args.bootstrap_samples, seed=args.seed
                ),
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            }

    report = {
        "schema": "shoppingbench-official-workflow-comparison-v1",
        "metric_contract": "official ShoppingBench ASR/CAR from upstream commit 2f4d132",
        "inputs": inputs,
        "summaries": summaries,
        "paired": paired,
        "rows": all_rows,
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "paired": paired}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
