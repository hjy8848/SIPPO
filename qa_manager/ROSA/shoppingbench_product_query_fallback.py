"""Gold-blind fallback for invalid single-product ShoppingBench queries."""

from __future__ import annotations

import re
from typing import Any, Sequence


_PRODUCT_DETAILS_MARKER = "Product details: "
_PLACEHOLDER_QUERY_RE = re.compile(
    r"(?:PLAIN|SEARCH|CONCRETE)_QUERY_OR_JSON_FILTER", re.IGNORECASE
)
_RESIDUAL_FIELD_RE = re.compile(
    r"&(?:amp;)?[a-z_][a-z0-9_]*(?:\[\])?\s*=", re.IGNORECASE
)


def visible_single_product_query(question: str) -> str:
    """Read one product description from the user-visible request only."""
    text = " ".join(str(question or "").split()).strip()
    if _PRODUCT_DETAILS_MARKER not in text:
        return ""
    details = text.split(_PRODUCT_DETAILS_MARKER, 1)[1]
    details = details.split(". Purchase terms: ", 1)[0].removesuffix(".")
    if re.search(r"\(\d+\)\s", details):
        return ""
    return details.strip()


def invalid_product_query(query: str) -> bool:
    """Reject copied placeholders and residual encoded query parameters."""
    normalized = " ".join(str(query or "").split()).strip()
    return bool(
        not normalized
        or _PLACEHOLDER_QUERY_RE.search(normalized)
        or _RESIDUAL_FIELD_RE.search(normalized)
    )


def select_product_query(
    question: str,
    model_queries: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """Preserve a valid model query, otherwise use visible Product details."""
    model_query = str(model_queries[0] if model_queries else "").strip()
    if not invalid_product_query(model_query):
        return model_query, {
            "applied": False,
            "reason": "model_query_valid",
            "gold_used": False,
        }

    fallback = visible_single_product_query(question)
    if not fallback:
        raise ValueError("invalid product query has no visible single-product fallback")
    return fallback, {
        "applied": True,
        "reason": "invalid_model_query",
        "rejected_model_query": model_query,
        "gold_used": False,
    }
