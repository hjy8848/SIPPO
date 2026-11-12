"""Deterministic routing for the causal Reason-to-Search Planner specialist."""

from __future__ import annotations

import re
from typing import Any, Sequence


PLANNER_ROUTE_MODES = frozenset({"all", "causal_exact", "anchor_exact"})
SPECIALIST_ROUTE_SCOPES = frozenset({"planner", "workflow"})
CAUSAL_TOPOLOGY = ("search", "search", "reason", "search")

_COMPARISON_RE = re.compile(
    r"\b(?:older|younger|earlier|later|first|last|closer\s+to|"
    r"larger|smaller|greater|less|fewer|more|longer|shorter|"
    r"comparison\s+value)\b",
    re.IGNORECASE,
)
_TARGET_PATTERNS = (
    re.compile(r"^where\s+was\s+the\b.*\bcandidate\b.*\bborn\b", re.IGNORECASE),
    re.compile(
        r"^what\s+(?:was|is)\s+the\s+(?:primary\s+)?"
        r"(?:occupation|profession|nationality)\s+of\s+the\b.*\bcandidate\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^what\s+(?:was|is)\s+the\s+genre\s+of\s+the\b.*\bwork\b",
        re.IGNORECASE,
    ),
    re.compile(r"^who\s+directed\s+the\b.*\bwork\b", re.IGNORECASE),
)


def normalize_route_mode(value: str) -> str:
    mode = str(value or "all").strip().lower()
    if mode not in PLANNER_ROUTE_MODES:
        raise ValueError(
            f"Unsupported PLANNER_ROUTE_MODE={value!r}; "
            f"expected one of {sorted(PLANNER_ROUTE_MODES)}"
        )
    return mode


def normalize_specialist_route_scope(value: str) -> str:
    scope = str(value or "planner").strip().lower()
    if scope not in SPECIALIST_ROUTE_SCOPES:
        raise ValueError(
            f"Unsupported SPECIALIST_ROUTE_SCOPE={value!r}; "
            f"expected one of {sorted(SPECIALIST_ROUTE_SCOPES)}"
        )
    return scope


def classify_causal_comparison_question(question: str) -> dict[str, Any]:
    """Recognize the frozen question grammar used to build the causal dataset."""

    text = " ".join(str(question or "").strip().split())
    if not text:
        return {
            "selected": False,
            "reason": "empty_question",
            "reasoning_mode": "unknown",
        }
    if " or " not in text.casefold():
        return {
            "selected": False,
            "reason": "missing_candidate_disjunction",
            "reasoning_mode": "unknown",
        }
    if not _COMPARISON_RE.search(text):
        return {
            "selected": False,
            "reason": "missing_comparison_operation",
            "reasoning_mode": "unknown",
        }
    if not any(pattern.search(text) for pattern in _TARGET_PATTERNS):
        return {
            "selected": False,
            "reason": "missing_supported_downstream_target",
            "reasoning_mode": "unknown",
        }
    reasoning_mode = (
        "anchor_distance"
        if re.search(r"\bcloser\s+to\b", text, re.IGNORECASE)
        else "direct"
    )
    return {
        "selected": True,
        "reason": "causal_comparison_question",
        "reasoning_mode": reasoning_mode,
    }


def plan_topology(nodes: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    return tuple(str(node[0]).strip().lower() for node in nodes if node)


def is_exact_causal_plan(nodes: Sequence[Sequence[Any]]) -> bool:
    return plan_topology(nodes) == CAUSAL_TOPOLOGY
