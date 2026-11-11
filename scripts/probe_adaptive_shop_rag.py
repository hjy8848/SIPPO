"""Run a small, gold-blind adaptive Shop retrieval probe on ShoppingBench."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from qa_manager.ROSA.shoppingbench_protocol import extract_visible_product_queries


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def canonical_query(slot: str) -> str:
    title, marker, condition = str(slot).partition("; it should have ")
    if not marker:
        price_title, price_marker, price_condition = str(slot).partition(
            "; it should cost "
        )
        if price_marker:
            title = price_title
            condition = price_condition
    query_terms = [title.strip()]
    if condition:
        condition = re.sub(
            r"^(?:color family|product color|color gradient|color|fit|is unisex|"
            r"size description|size int|size waist|grams|specification)\s*[:=]?\s*",
            "",
            condition,
            flags=re.I,
        )
        condition = re.sub(
            r"^(?:no more than|at least|no less than)\s+", "", condition, flags=re.I
        )
        condition = re.sub(r"\s+Philippine pesos\s*$", "", condition, flags=re.I)
        if condition.strip():
            query_terms.append(condition.strip())
    return " ".join(" ".join(query_terms).split())


def _json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def structured_fields(product: Mapping[str, Any]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}

    def visit(value: object, parent: str = "") -> None:
        value = _json_value(value)
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = normalize(key)
                if isinstance(_json_value(child), (Mapping, list)):
                    visit(child, normalized_key or parent)
                elif normalized_key:
                    fields.setdefault(normalized_key, []).append(str(child))
        elif isinstance(value, list):
            if parent and all(not isinstance(_json_value(item), (Mapping, list)) for item in value):
                fields.setdefault(parent, []).extend(str(item) for item in value)
            else:
                for child in value:
                    visit(child, parent)

    visit(product.get("sku_options"))
    visit(product.get("attributes"))
    return fields


def constraint_status(
    slot: str,
    product: Mapping[str, Any],
    field_universe: Sequence[str],
) -> str:
    price = product.get("price")
    upper = re.search(r"no more than\s+([\d.]+)", slot, re.I)
    lower = re.search(r"(?:at least|no less than)\s+([\d.]+)", slot, re.I)
    if upper or lower:
        try:
            numeric_price = float(price)
        except (TypeError, ValueError):
            return "unknown"
        if upper and numeric_price > float(upper.group(1)):
            return "contradicts"
        if lower and numeric_price < float(lower.group(1)):
            return "contradicts"
        return "satisfies"

    match = re.search(r"\bit should have\s+(.+?)\s*$", slot, re.I)
    suffix = normalize(match.group(1) if match else "")
    matching_fields = [
        field for field in field_universe if suffix == field or suffix.startswith(field + " ")
    ]
    field = max(matching_fields, key=len, default="")
    if not field:
        return "unknown"
    value = suffix[len(field) :].strip()
    values = [normalize(item) for item in structured_fields(product).get(field, [])]
    if not values:
        return "unknown"
    if any(value == item or value in item or item in value for item in values):
        return "satisfies"
    return "contradicts"


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(candidate.get("search_page", 99)),
        int(candidate.get("search_rank", 99)),
        str(candidate.get("product_id", "")),
    )


def add_candidates(
    store: list[dict[str, dict[str, Any]]],
    slot_index: int,
    products: Sequence[Mapping[str, Any]],
    *,
    query: str,
    page: int,
    shop_conditioned: bool,
) -> None:
    for rank, product in enumerate(products, start=1):
        product_id = str(product.get("product_id") or "")
        shop_id = str(product.get("shop_id") or "")
        if not product_id.isdigit() or not shop_id.isdigit():
            continue
        candidate = {
            **dict(product),
            "product_id": product_id,
            "shop_id": shop_id,
            "source_query": query,
            "search_page": page,
            "search_rank": rank,
            "shop_conditioned": shop_conditioned,
        }
        previous = store[slot_index].get(product_id)
        if previous is None or _candidate_key(candidate) < _candidate_key(previous):
            store[slot_index][product_id] = candidate


def coverage_table(store: Sequence[Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    shops = {
        str(product.get("shop_id"))
        for slot in store
        for product in slot.values()
        if str(product.get("shop_id") or "").isdigit()
    }
    rows = []
    for shop_id in shops:
        covered = []
        rank_sum = 0
        for slot_index, slot in enumerate(store):
            products = [p for p in slot.values() if p.get("shop_id") == shop_id]
            if products:
                covered.append(slot_index)
                rank_sum += min(int(p.get("search_rank", 99)) for p in products)
        rows.append(
            {
                "shop_id": shop_id,
                "covered_slots": covered,
                "coverage": len(covered),
                "rank_sum": rank_sum,
            }
        )
    return sorted(rows, key=lambda row: (-row["coverage"], row["rank_sum"], row["shop_id"]))


def solve_same_shop(
    slots: Sequence[str],
    store: Sequence[Mapping[str, Mapping[str, Any]]],
    shop_priority: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    field_universe = sorted(
        {
            field
            for slot in store
            for product in slot.values()
            for field in structured_fields(product)
        },
        key=len,
        reverse=True,
    )
    common_shops: set[str] | None = None
    for slot in store:
        slot_shops = {str(product.get("shop_id")) for product in slot.values()}
        common_shops = slot_shops if common_shops is None else common_shops & slot_shops
    best: tuple[tuple[Any, ...], list[dict[str, Any]], list[str]] | None = None
    for shop_id in sorted(common_shops or set()):
        slot_candidates = [
            [dict(product) for product in slot.values() if product.get("shop_id") == shop_id]
            for slot in store
        ]
        for combination in itertools.product(*slot_candidates):
            ids = [str(product.get("product_id")) for product in combination]
            if len(set(ids)) != len(ids):
                continue
            statuses = [
                constraint_status(slots[index], product, field_universe)
                for index, product in enumerate(combination)
            ]
            if "contradicts" in statuses:
                continue
            key = (
                statuses.count("unknown"),
                -statuses.count("satisfies"),
                shop_priority.get(shop_id, 9999),
                sum(int(product.get("search_page", 99)) for product in combination),
                sum(int(product.get("search_rank", 99)) for product in combination),
                tuple(ids),
            )
            if best is None or key < best[0]:
                best = (key, [dict(product) for product in combination], statuses)
    if best is None:
        return [], {"reason": "no_feasible_same_shop_assignment"}
    return best[1], {
        "reason": "feasible_same_shop_assignment",
        "constraint_statuses": best[2],
        "objective": list(best[0][:-1]),
    }


def run_probe(
    question: str,
    retriever: Any,
    *,
    topk: int = 5,
    max_pages: int = 5,
    shop_beam: int = 3,
    max_search_calls: int = 28,
) -> dict[str, Any]:
    slots = extract_visible_product_queries(question)
    if len(slots) < 2:
        raise ValueError(f"Shop probe needs at least two visible product slots, found {len(slots)}")
    queries = [canonical_query(slot) for slot in slots]
    store: list[dict[str, dict[str, Any]]] = [dict() for _ in slots]
    actions: list[dict[str, Any]] = []

    def search(slot_index: int, page: int, shop_id: str = "") -> list[dict[str, Any]]:
        if len(actions) >= max_search_calls:
            return []
        payload: dict[str, Any] = {"q": queries[slot_index], "page": page}
        if shop_id:
            payload["shop_id"] = shop_id
        products = retriever._search_local_index(payload, topk)
        actions.append(
            {
                "action": "search_in_shop" if shop_id else "search_global",
                "slot": slot_index + 1,
                "query": queries[slot_index],
                "shop_id": shop_id or None,
                "page": page,
                "returned": len(products),
                "product_ids": [str(product.get("product_id")) for product in products],
            }
        )
        add_candidates(
            store,
            slot_index,
            products,
            query=queries[slot_index],
            page=page,
            shop_conditioned=bool(shop_id),
        )
        return [dict(product) for product in products]

    for slot_index in range(len(slots)):
        search(slot_index, 1)
    initial_coverage = coverage_table(store)
    shop_priority = {row["shop_id"]: index for index, row in enumerate(initial_coverage)}
    selection, decision = solve_same_shop(slots, store, shop_priority)

    attempted_shop_slots: set[tuple[str, int]] = set()

    def expand_candidate_shops() -> None:
        current_coverage = coverage_table(store)
        for shop in current_coverage[:shop_beam]:
            shop_id = str(shop["shop_id"])
            missing = [index for index in range(len(slots)) if index not in shop["covered_slots"]]
            for slot_index in missing:
                attempt_key = (shop_id, slot_index)
                if attempt_key in attempted_shop_slots:
                    continue
                attempted_shop_slots.add(attempt_key)
                for page in range(1, max_pages + 1):
                    products = search(slot_index, page, shop_id)
                    if not products:
                        break
                    field_universe = sorted(
                        {
                            field
                            for candidate in store[slot_index].values()
                            for field in structured_fields(candidate)
                        },
                        key=len,
                        reverse=True,
                    )
                    viable = [
                        product
                        for product in products
                        if constraint_status(slots[slot_index], product, field_universe)
                        != "contradicts"
                    ]
                    if viable or len(actions) >= max_search_calls:
                        break
            if len(actions) >= max_search_calls:
                break

    if not selection:
        expand_candidate_shops()
        selection, decision = solve_same_shop(slots, store, shop_priority)

    # If the first top-k slices expose no viable shop, spend the remaining budget
    # on global discovery before conditioning on newly discovered shops.
    for page in range(2, max_pages + 1):
        if selection or len(actions) >= max_search_calls:
            break
        for slot_index in range(len(slots)):
            search(slot_index, page)
            if len(actions) >= max_search_calls:
                break
        current_coverage = coverage_table(store)
        for row in current_coverage:
            shop_priority.setdefault(row["shop_id"], len(shop_priority))
        selection, decision = solve_same_shop(slots, store, shop_priority)
        if selection or len(actions) >= max_search_calls:
            break
        expand_candidate_shops()
        selection, decision = solve_same_shop(slots, store, shop_priority)

    return {
        "slots": slots,
        "queries": queries,
        "initial_coverage": initial_coverage,
        "final_coverage": coverage_table(store),
        "actions": actions,
        "search_calls": len(actions),
        "predicted_product_ids": [product["product_id"] for product in selection],
        "selected_shop_id": selection[0]["shop_id"] if selection else None,
        "decision": decision,
        "selected_products": selection,
        "gold_runtime_access": False,
        "external_api_calls": 0,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(expected: Sequence[object], predicted: Sequence[object]) -> dict[str, Any]:
    expected_ids = [str(value) for value in expected]
    predicted_ids = [str(value) for value in predicted]
    matched = sum(
        left == right for left, right in itertools.zip_longest(expected_ids, predicted_ids)
    )
    return {
        "matched_product_ids": matched,
        "expected_product_count": len(expected_ids),
        "exact_id_fraction": matched / len(expected_ids) if expected_ids else 0.0,
        "exact_sequence_success": predicted_ids == expected_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--shop-beam", type=int, default=3)
    args = parser.parse_args()

    os.environ["SHOPPINGBENCH_INDEX_PATH"] = str(args.index)
    os.environ["SHOPPINGBENCH_LOCAL_ONLY"] = "1"
    from exp.shoppingbench_local_retriever import ShoppingBenchRetriever

    dataset_rows = {f"shop-{index:04d}": row for index, row in enumerate(_read_jsonl(args.dataset))}
    baseline_rows = {row["sample_id"]: row for row in _read_jsonl(args.baseline)}
    retriever = ShoppingBenchRetriever()
    reports = []
    for sample_id in args.sample_id:
        row = dataset_rows[sample_id]
        baseline = baseline_rows[sample_id]
        probe = run_probe(
            str(row["query"]),
            retriever,
            topk=args.topk,
            max_pages=args.max_pages,
            shop_beam=args.shop_beam,
        )
        expected = [str(value) for value in row.get("source_product_ids") or []]
        reports.append(
            {
                "sample_id": sample_id,
                "question": row["query"],
                "expected_product_ids": expected,
                "baseline_product_ids": baseline.get("predicted_product_ids") or [],
                "baseline_score": baseline.get("score") or {},
                "probe_score": _score(expected, probe["predicted_product_ids"]),
                "probe": probe,
            }
        )
    output = {
        "protocol": {
            "name": "asc-shop-zero-model-mechanism-probe-v1",
            "selection_rule": "first baseline exact success plus first baseline empty prediction",
            "sample_ids": args.sample_id,
            "topk": args.topk,
            "max_pages": args.max_pages,
            "shop_beam": args.shop_beam,
            "dataset_sha256": _sha256(args.dataset),
            "baseline_sha256": _sha256(args.baseline),
            "gold_runtime_access": False,
            "external_api_calls": 0,
            "model_calls": 0,
        },
        "rows": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
