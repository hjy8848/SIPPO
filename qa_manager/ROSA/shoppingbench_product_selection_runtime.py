"""Request-scoped adapter for the isolated Product selection fallback."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Sequence

from qa_manager.ROSA import shoppingbench_protocol
from qa_manager.ROSA.shoppingbench_product_selection_fallback import (
    apply_conservative_product_selection,
)


_BASE_SELECT = shoppingbench_protocol.select_supported_products
_QUESTION: ContextVar[str] = ContextVar("product_selection_question", default="")
_DECISIONS: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "product_selection_decisions", default=()
)


@contextmanager
def product_selection_request(question: str) -> Iterator[None]:
    question_token = _QUESTION.set(str(question or ""))
    decisions_token = _DECISIONS.set(())
    try:
        yield
    finally:
        _DECISIONS.reset(decisions_token)
        _QUESTION.reset(question_token)


def _corrected_selection(
    products: Sequence[Mapping[str, Any]],
    model_conclusion: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_ids = {
        str(product.get("product_id", ""))
        for product in _BASE_SELECT(products, model_conclusion)
    }
    annotated = [
        {
            **dict(product),
            "model_selected": str(product.get("product_id", "")) in selected_ids,
        }
        for product in products
    ]
    return apply_conservative_product_selection(_QUESTION.get(), annotated)


def select_supported_products(
    products: Sequence[Mapping[str, Any]],
    model_conclusion: str,
) -> list[dict[str, Any]]:
    corrected, _ = _corrected_selection(products, model_conclusion)
    return [product for product in corrected if product.get("model_selected")]


def annotate_product_selection(
    products: Sequence[Mapping[str, Any]],
    model_conclusion: str,
) -> list[dict[str, Any]]:
    corrected, decision = _corrected_selection(products, model_conclusion)
    _DECISIONS.set((*_DECISIONS.get(), decision))
    return corrected


def product_selection_decisions() -> list[dict[str, Any]]:
    return [dict(decision) for decision in _DECISIONS.get()]
