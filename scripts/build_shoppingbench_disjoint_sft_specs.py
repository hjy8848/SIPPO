#!/usr/bin/env python3
"""Build deterministic ShoppingBench SFT task specs disjoint from the test set."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TASKS = ("product", "shop", "voucher")


def stable_int(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def reward_products(reward: Any) -> list[dict[str, Any]]:
    if isinstance(reward, dict):
        return [reward]
    if isinstance(reward, list):
        return [item for item in reward if isinstance(item, dict)]
    return []


def load_test_contract(test_dir: Path) -> tuple[set[str], set[str]]:
    product_ids: set[str] = set()
    normalized_queries: set[str] = set()
    for task in TASKS:
        path = test_dir / f"synthesize_{task}_test.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing ShoppingBench test file: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                query = " ".join(str(row.get("query") or "").lower().split())
                if query:
                    normalized_queries.add(query)
                for product in reward_products(row.get("reward")):
                    product_id = str(product.get("product_id") or "")
                    if product_id:
                        product_ids.add(product_id)
    return product_ids, normalized_queries


def normalize_product(row: dict[str, Any]) -> dict[str, Any] | None:
    product = row.get("product") if isinstance(row.get("product"), dict) else row
    product_id = str(product.get("product_id") or "")
    shop_id = str(product.get("shop_id") or "")
    title = " ".join(str(product.get("title") or "").split())
    try:
        price = float(product.get("price"))
    except (TypeError, ValueError):
        price = 0.0
    if not product_id.isdigit() or not shop_id.isdigit() or not title or price <= 0:
        return None
    return {
        "product_id": product_id,
        "shop_id": shop_id,
        "title": title,
        "price": price,
        "sku_options": product.get("sku_options") if isinstance(product.get("sku_options"), dict) else {},
        "attributes": product.get("attributes") if isinstance(product.get("attributes"), dict) else {},
        "service": [str(value) for value in product.get("service") or [] if value],
    }


def first_constraint(product: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    for option in product["sku_options"].values():
        if not isinstance(option, dict):
            continue
        for key, value in sorted(option.items()):
            if value not in (None, "") and key not in {"min_delivery_time", "max_delivery_time"}:
                raw_key = str(key)
                display_key = raw_key.replace("_", " ")
                return f"have {display_key} {value}", {raw_key: str(value)}
    for key, values in sorted(product["attributes"].items()):
        values = values if isinstance(values, list) else [values]
        values = [str(value) for value in values if value not in (None, "")]
        if values:
            raw_key = str(key)
            display_key = raw_key.replace("_", " ")
            return f"have {display_key} {values[0]}", {raw_key: [values[0]]}
    if product["service"]:
        return f"offer {product['service'][0]}", None
    return f"cost no more than {math.ceil(product['price'] * 1.1)} Philippine pesos", None


def visible_item(product: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    constraint_text, structured = first_constraint(product)
    reward = {"product_id": product["product_id"], "title": [product["title"]]}
    if structured:
        key = next(iter(structured))
        if key in product["attributes"]:
            reward["attributes"] = [structured]
        else:
            reward["sku_options"] = [structured]
    return f"{product['title']}; it should {constraint_text}", reward


def bounded_push(heap: list[tuple[int, str, dict[str, Any]]], limit: int, score: int, product: dict[str, Any]) -> None:
    item = (-score, product["product_id"], product)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def split_for_shop(seed: int, shop_id: str, val_percent: int) -> str:
    return "val" if stable_int(seed, "split", shop_id) % 100 < val_percent else "train"


def shop_candidate_per_mille(count_per_task: int) -> int:
    return min(1000, max(25, count_per_task // 6 + 25))


def assign_shop_cardinalities(
    groups: list[list[dict[str, Any]]],
    requested: tuple[int, ...],
) -> list[int]:
    """Preserve the requested histogram while assigning larger rows to capable shops."""

    assigned = [requested[index % len(requested)] for index in range(len(groups))]
    for index, (group, cardinality) in enumerate(zip(groups, assigned, strict=True)):
        if len(group) >= cardinality:
            continue
        candidate_indices = [
            *range(index + 1, len(groups)),
            *range(0, index),
        ]
        replacement = next(
            (
                other
                for other in candidate_indices
                if len(groups[other]) >= cardinality
                and len(group) >= assigned[other]
            ),
            None,
        )
        if replacement is None:
            raise ValueError("insufficient shop products for requested cardinality histogram")
        assigned[index], assigned[replacement] = assigned[replacement], assigned[index]
    if any(len(group) < cardinality for group, cardinality in zip(groups, assigned, strict=True)):
        raise ValueError("shop cardinality assignment violated a group capacity")
    return assigned


def collect_candidates(
    documents: Path,
    excluded_product_ids: set[str],
    *,
    count_per_task: int,
    val_percent: int,
    seed: int,
    shop_group_size: int,
    shop_cardinalities: tuple[int, ...] | None = None,
    voucher_cardinalities: tuple[int, ...] = (1,),
) -> dict[str, dict[str, list[Any]]]:
    individual_limit = max(count_per_task * 8, 64)
    shop_sampling_rate = shop_candidate_per_mille(count_per_task)
    individual: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        "train": [],
        "val": [],
    }
    shop_cardinalities = shop_cardinalities or (shop_group_size,)
    max_shop_cardinality = max(shop_group_size, max(shop_cardinalities))
    shop_products: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with documents.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = normalize_product(json.loads(line))
            if product is None or product["product_id"] in excluded_product_ids:
                continue
            split = split_for_shop(seed, product["shop_id"], val_percent)
            bounded_push(
                individual[split],
                individual_limit,
                stable_int(seed, "individual", product["product_id"]),
                product,
            )
            if stable_int(seed, "shop-candidate", product["shop_id"]) % 1000 < shop_sampling_rate:
                key = (split, product["shop_id"])
                if len(shop_products[key]) < max_shop_cardinality:
                    shop_products[key].append(product)

    result: dict[str, dict[str, list[Any]]] = {
        split: {task: [] for task in TASKS} for split in ("train", "val")
    }
    for split in ("train", "val"):
        groups = [
            products
            for (group_split, _), products in shop_products.items()
            if group_split == split and len(products) >= shop_group_size
        ]
        groups.sort(key=lambda products: stable_int(seed, "shop-order", products[0]["shop_id"]))
        selected_shop_groups = groups[:count_per_task]
        assigned_shop_cardinalities = assign_shop_cardinalities(
            selected_shop_groups,
            shop_cardinalities,
        )
        result[split]["shop"] = [
            group[:cardinality]
            for group, cardinality in zip(
                selected_shop_groups,
                assigned_shop_cardinalities,
                strict=True,
            )
        ]
        # Keep downstream product/voucher sampling identical to the frozen base cardinality.
        used = {
            product["product_id"]
            for group in selected_shop_groups
            for product in group[:shop_group_size]
        }
        products = [item[2] for item in sorted(individual[split], reverse=True) if item[2]["product_id"] not in used]
        result[split]["product"] = products[:count_per_task]
        voucher_groups = []
        cursor = count_per_task
        for index in range(count_per_task):
            cardinality = voucher_cardinalities[index % len(voucher_cardinalities)]
            group = products[cursor : cursor + cardinality]
            if len(group) != cardinality:
                raise ValueError(
                    f"insufficient {split}/voucher candidates for cardinality {cardinality}"
                )
            voucher_groups.append(group)
            cursor += cardinality
        result[split]["voucher"] = voucher_groups
        for task in TASKS:
            if len(result[split][task]) != count_per_task:
                raise ValueError(f"insufficient {split}/{task} candidates: {len(result[split][task])}")
    return result


def make_spec(task: str, products: list[dict[str, Any]], split: str, index: int) -> dict[str, Any]:
    items = [visible_item(product) for product in products]
    visible = [item[0] for item in items]
    rewards = [item[1] for item in items]
    if task == "product":
        query = f"Find this product in the catalog: {visible[0]}."
        reward: Any = rewards[0]
        voucher = None
    elif task == "shop":
        numbered = "; ".join(f"({i}) {text}" for i, text in enumerate(visible, 1))
        query = f"Find one shop that sells all of these items: {numbered}."
        reward = rewards
        voucher = None
    else:
        subtotal = sum(product["price"] for product in products)
        threshold = max(1, math.floor(subtotal * 0.4))
        face_value = max(1, math.floor(subtotal * 0.1))
        budget = math.ceil(subtotal - face_value + 5)
        product_request = (
            f"Find this product: {visible[0]}"
            if len(visible) == 1
            else "Find these products: "
            + "; ".join(f"({i}) {text}" for i, text in enumerate(visible, 1))
        )
        query = (
            f"{product_request}. My budget is {budget} Philippine pesos. "
            f"I have a platform voucher valid above {threshold} pesos with a fixed "
            f"discount of {face_value} pesos."
        )
        reward = rewards
        voucher = {
            "voucher_type": "platform",
            "threshold": threshold,
            "discount_type": "fixed",
            "face_value": face_value,
            "budget": budget,
        }
    row = {
        "spec_id": f"{split}-{task}-{index:05d}",
        "split": split,
        "task": task,
        "query": query,
        "reward": reward,
        "source_product_ids": [product["product_id"] for product in products],
        "source_shop_ids": sorted({product["shop_id"] for product in products}),
        "planner_input_fields": ["query"],
        "gold_runtime_access": False,
    }
    if voucher is not None:
        row["voucher"] = voucher
    return row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    excluded_ids, test_queries = load_test_contract(args.test_dir)
    voucher_cardinalities = tuple(getattr(args, "voucher_cardinalities", [1]))
    shop_cardinalities = tuple(
        getattr(args, "shop_cardinalities", None) or [args.shop_group_size]
    )
    candidates = collect_candidates(
        args.documents,
        excluded_ids,
        count_per_task=args.count_per_task,
        val_percent=args.val_percent,
        seed=args.seed,
        shop_group_size=args.shop_group_size,
        shop_cardinalities=shop_cardinalities,
        voucher_cardinalities=voucher_cardinalities,
    )
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val"):
        rows = []
        for task in TASKS:
            for index, candidate in enumerate(candidates[split][task]):
                products = candidate if task in {"shop", "voucher"} else [candidate]
                rows.append(make_spec(task, products, split, index))
        rows_by_split[split] = rows

    train_ids = {value for row in rows_by_split["train"] for value in row["source_product_ids"]}
    val_ids = {value for row in rows_by_split["val"] for value in row["source_product_ids"]}
    train_shops = {value for row in rows_by_split["train"] for value in row["source_shop_ids"]}
    val_shops = {value for row in rows_by_split["val"] for value in row["source_shop_ids"]}
    generated_queries = [" ".join(row["query"].lower().split()) for rows in rows_by_split.values() for row in rows]
    gates = {
        "test_product_ids_excluded": not bool((train_ids | val_ids) & excluded_ids),
        "train_val_product_disjoint": not bool(train_ids & val_ids),
        "train_val_shop_disjoint": not bool(train_shops & val_shops),
        "test_queries_excluded": not bool(set(generated_queries) & test_queries),
        "product_ids_absent_from_queries": all(
            product_id not in row["query"]
            for rows in rows_by_split.values()
            for row in rows
            for product_id in row["source_product_ids"]
        ),
        "planner_reads_query_only": all(
            row["planner_input_fields"] == ["query"] and row["gold_runtime_access"] is False
            for rows in rows_by_split.values()
            for row in rows
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"SFT spec leakage gate failed: {gates}")

    args.output_dir.mkdir(parents=True)
    for split, rows in rows_by_split.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    report = {
        "schema": "shoppingbench-cp-sft-spec-v1",
        "seed": args.seed,
        "count_per_task_per_split": args.count_per_task,
        "shop_group_size": args.shop_group_size,
        "shop_cardinalities": list(shop_cardinalities),
        "shop_cardinality_counts": {
            split: {
                str(cardinality): sum(
                    row["task"] == "shop"
                    and len(row["source_product_ids"]) == cardinality
                    for row in rows
                )
                for cardinality in sorted(set(shop_cardinalities))
            }
            for split, rows in rows_by_split.items()
        },
        "voucher_cardinalities": list(voucher_cardinalities),
        "voucher_cardinality_counts": {
            split: {
                str(cardinality): sum(
                    row["task"] == "voucher"
                    and len(row["source_product_ids"]) == cardinality
                    for row in rows
                )
                for cardinality in sorted(set(voucher_cardinalities))
            }
            for split, rows in rows_by_split.items()
        },
        "shop_candidate_per_mille": shop_candidate_per_mille(args.count_per_task),
        "test_product_ids": len(excluded_ids),
        "rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "unique_products": {"train": len(train_ids), "val": len(val_ids)},
        "unique_shops": {"train": len(train_shops), "val": len(val_shops)},
        "gates": gates,
        "external_api_calls": 0,
        "status": "task_specs_only_not_sft_messages",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-per-task", type=int, default=1000)
    parser.add_argument("--shop-group-size", type=int, default=3)
    parser.add_argument("--shop-cardinalities", type=int, nargs="+")
    parser.add_argument("--voucher-cardinalities", type=int, nargs="+", default=[1])
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.count_per_task <= 0 or args.shop_group_size < 2:
        parser.error("count-per-task must be positive and shop-group-size at least 2")
    if not args.voucher_cardinalities or any(
        value < 1 or value > 4 for value in args.voucher_cardinalities
    ):
        parser.error("voucher-cardinalities must contain values in [1, 4]")
    if args.shop_cardinalities and any(
        value < 2 or value > 4 for value in args.shop_cardinalities
    ):
        parser.error("shop-cardinalities must contain values in [2, 4]")
    if not 1 <= args.val_percent <= 50:
        parser.error("val-percent must be in [1, 50]")
    return args


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
