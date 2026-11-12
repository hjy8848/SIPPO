"""Gold-blind conservative Product selection over retriever-owned attributes."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


def normalize_visible_value(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def structured_product_fields(product: Mapping[str, Any]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for source in (product.get("sku_options"), product.get("attributes")):
        if not source:
            continue
        try:
            payload = json.loads(str(source))
        except json.JSONDecodeError:
            continue

        def visit(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized_key = normalize_visible_value(key)
                    if isinstance(child, list) and all(
                        not isinstance(item, (dict, list)) for item in child
                    ):
                        fields.setdefault(normalized_key, []).extend(str(item) for item in child)
                    elif isinstance(child, (dict, list)):
                        visit(child)
                    else:
                        fields.setdefault(normalized_key, []).append(str(child))
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
    return fields


def visible_product_constraint(
    question: str,
    products: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    match = re.search(r"\bit should have\s+(.+?)\s*$", str(question), re.IGNORECASE)
    suffix = normalize_visible_value(match.group(1) if match else "")
    keys = {
        key
        for product in products
        for key in structured_product_fields(product)
        if suffix == key or suffix.startswith(key + " ")
    }
    field = max(keys, key=len, default="")
    value = suffix[len(field) :].strip() if field else suffix
    return field, value


def product_constraint_status(product: Mapping[str, Any], field: str, value: str) -> str:
    fields = structured_product_fields(product)
    if not field or field not in fields:
        return "unverified"
    normalized_values = [normalize_visible_value(item) for item in fields[field]]
    if any(value == item or value in item or item in value for item in normalized_values):
        return "satisfies"
    return "contradicts"


def apply_conservative_product_selection(
    question: str,
    annotated_products: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace a non-satisfying selection only when one candidate satisfies."""
    products = [dict(product) for product in annotated_products]
    field, value = visible_product_constraint(question, products)
    statuses = {
        str(product.get("product_id", "")): product_constraint_status(product, field, value)
        for product in products
    }
    satisfying = [product_id for product_id, status in statuses.items() if status == "satisfies"]
    model_selected = [
        str(product.get("product_id", "")) for product in products if product.get("model_selected")
    ]
    baseline_id = model_selected[0] if len(model_selected) == 1 else ""
    baseline_status = statuses.get(baseline_id, "missing")
    applied = bool(baseline_id) and baseline_status != "satisfies" and len(satisfying) == 1
    selected_id = satisfying[0] if applied else baseline_id
    if applied:
        products = [
            {**product, "model_selected": str(product.get("product_id", "")) == selected_id}
            for product in products
        ]
    return products, {
        "enabled": True,
        "applied": applied,
        "constraint_field": field,
        "constraint_value": value,
        "candidate_statuses": statuses,
        "satisfying_ids": satisfying,
        "baseline_id": baseline_id,
        "baseline_status": baseline_status,
        "selected_id": selected_id,
        "decision": "unique_satisfier_replaces_non_satisfier" if applied else "preserve",
        "gold_used": False,
    }
