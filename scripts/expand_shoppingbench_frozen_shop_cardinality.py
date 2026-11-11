#!/usr/bin/env python3
"""Expand frozen ShoppingBench shop rows to 2/3/4 products without resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.build_shoppingbench_disjoint_sft_specs import make_spec, normalize_product, stable_int


SPLITS = ("train", "val")
TARGET_CARDINALITIES = (2, 3, 4)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def desired_histogram(count: int) -> Counter[int]:
    return Counter(TARGET_CARDINALITIES[index % len(TARGET_CARDINALITIES)] for index in range(count))


def collect_shop_products(
    documents: Path,
    shop_ids: set[str],
    required_product_ids: set[str],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_shop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    found_required: set[str] = set()
    with documents.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = normalize_product(json.loads(line))
            if product is None or product["shop_id"] not in shop_ids:
                continue
            by_shop[product["shop_id"]].append(product)
            if product["product_id"] in required_product_ids:
                found_required.add(product["product_id"])
    missing = required_product_ids - found_required
    if missing:
        raise ValueError(f"frozen source products missing from corpus: {sorted(missing)[:5]}")
    for shop_id, products in by_shop.items():
        products.sort(key=lambda product: (stable_int(seed, "extra-product", shop_id, product["product_id"]), product["product_id"]))
    return dict(by_shop)


def assign_cardinalities(rows: list[dict[str, Any]], capacities: list[int]) -> list[int]:
    assigned = [TARGET_CARDINALITIES[index % len(TARGET_CARDINALITIES)] for index in range(len(rows))]
    for index, cardinality in enumerate(assigned):
        if capacities[index] >= cardinality:
            continue
        replacement = next(
            (
                other
                for other in [*range(index + 1, len(rows)), *range(0, index)]
                if capacities[other] >= cardinality and capacities[index] >= assigned[other]
            ),
            None,
        )
        if replacement is None:
            raise ValueError("cannot preserve the 2/3/4 histogram with frozen shop identities")
        assigned[index], assigned[replacement] = assigned[replacement], assigned[index]
    if Counter(assigned) != desired_histogram(len(rows)):
        raise AssertionError("cardinality assignment changed the predeclared histogram")
    return assigned


def expand_split(
    rows: list[dict[str, Any]],
    products_by_shop: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shop_rows = [row for row in rows if row.get("task") == "shop"]
    capacities = [len(products_by_shop.get(str(row["source_shop_ids"][0]), [])) for row in shop_rows]
    assigned = assign_cardinalities(shop_rows, capacities)
    cardinality_by_spec = {
        str(row["spec_id"]): cardinality
        for row, cardinality in zip(shop_rows, assigned, strict=True)
    }
    expanded: list[dict[str, Any]] = []
    extras: list[str] = []
    for row in rows:
        if row.get("task") != "shop":
            expanded.append(row)
            continue
        shop_ids = [str(value) for value in row.get("source_shop_ids") or []]
        source_ids = [str(value) for value in row.get("source_product_ids") or []]
        if len(shop_ids) != 1 or len(source_ids) != 3:
            raise ValueError(f"{row.get('spec_id')}: expected one frozen shop and three products")
        shop_products = products_by_shop.get(shop_ids[0], [])
        by_id = {product["product_id"]: product for product in shop_products}
        ordered = [by_id[product_id] for product_id in source_ids]
        cardinality = cardinality_by_spec[str(row["spec_id"])]
        if cardinality == 3:
            expanded.append(row)
            continue
        if cardinality == 4:
            extra = next((product for product in shop_products if product["product_id"] not in source_ids), None)
            if extra is None:
                raise ValueError(f"{row.get('spec_id')}: no fourth product in frozen shop")
            ordered.append(extra)
            extras.append(extra["product_id"])
        index = int(str(row["spec_id"]).rsplit("-", 1)[1])
        updated = make_spec("shop", ordered[:cardinality], str(row["split"]), index)
        if updated["source_shop_ids"] != row["source_shop_ids"]:
            raise AssertionError(f"{row.get('spec_id')}: shop identity changed")
        if updated["source_product_ids"][: min(3, cardinality)] != source_ids[: min(3, cardinality)]:
            raise AssertionError(f"{row.get('spec_id')}: frozen product prefix changed")
        expanded.append(updated)
    return expanded, {
        "shop_cardinality_histogram": dict(sorted(Counter(assigned).items())),
        "added_product_ids": extras,
    }


def audit_identity(
    source: dict[str, list[dict[str, Any]]],
    expanded: dict[str, list[dict[str, Any]]],
) -> dict[str, bool]:
    gates = {
        "row_counts_unchanged": all(len(source[split]) == len(expanded[split]) for split in SPLITS),
        "row_order_and_spec_ids_unchanged": all(
            [row["spec_id"] for row in source[split]] == [row["spec_id"] for row in expanded[split]]
            for split in SPLITS
        ),
        "product_and_voucher_rows_identical": all(
            before == after
            for split in SPLITS
            for before, after in zip(source[split], expanded[split], strict=True)
            if before["task"] != "shop"
        ),
        "shop_ids_unchanged": all(
            before["source_shop_ids"] == after["source_shop_ids"]
            for split in SPLITS
            for before, after in zip(source[split], expanded[split], strict=True)
            if before["task"] == "shop"
        ),
        "shop_product_prefix_preserved": all(
            after["source_product_ids"][: min(3, len(after["source_product_ids"]))]
            == before["source_product_ids"][: min(3, len(after["source_product_ids"]))]
            for split in SPLITS
            for before, after in zip(source[split], expanded[split], strict=True)
            if before["task"] == "shop"
        ),
        "cardinality_three_rows_identical": all(
            before == after
            for split in SPLITS
            for before, after in zip(source[split], expanded[split], strict=True)
            if before["task"] == "shop" and len(after["source_product_ids"]) == 3
        ),
        "shop_cardinality_coverage_exact": all(
            Counter(len(row["source_product_ids"]) for row in expanded[split] if row["task"] == "shop")
            == desired_histogram(sum(row["task"] == "shop" for row in source[split]))
            for split in SPLITS
        ),
    }
    return gates


def build(source_dir: Path, documents: Path, output_dir: Path, *, seed: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    source = {split: read_jsonl(source_dir / f"{split}.jsonl") for split in SPLITS}
    shop_rows = [row for split in SPLITS for row in source[split] if row.get("task") == "shop"]
    shop_ids = {str(row["source_shop_ids"][0]) for row in shop_rows}
    required_ids = {str(value) for row in shop_rows for value in row["source_product_ids"]}
    products_by_shop = collect_shop_products(documents, shop_ids, required_ids, seed=seed)
    expanded: dict[str, list[dict[str, Any]]] = {}
    split_metrics: dict[str, Any] = {}
    for split in SPLITS:
        expanded[split], split_metrics[split] = expand_split(source[split], products_by_shop)
    gates = audit_identity(source, expanded)
    if not all(gates.values()):
        raise ValueError(f"frozen-row identity gate failed: {gates}")
    output_dir.mkdir(parents=True)
    for split in SPLITS:
        write_jsonl(output_dir / f"{split}.jsonl", expanded[split])
    manifest = {
        "schema": "shoppingbench-frozen-shop-cardinality-expansion-v1",
        "source_dir": str(source_dir),
        "documents": str(documents),
        "seed": seed,
        "target_cardinalities": list(TARGET_CARDINALITIES),
        "rows": {split: len(expanded[split]) for split in SPLITS},
        "source_sha256": {split: sha256(source_dir / f"{split}.jsonl") for split in SPLITS},
        "split_metrics": split_metrics,
        "gates": gates,
        "external_api_calls": 0,
        "status": "offline_specs_only_not_sft_messages",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.source_dir, args.documents, args.output_dir, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
