"""Protocol helpers for the single-policy linear ROSA trajectory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from qa_manager.normalize_answers import normalize_answer


LINEAR_SYSTEM_PROMPT = """You are a retrieval-and-reasoning policy for multi-hop question answering.

Work in a causal sequence. At each turn output exactly one action and nothing else:

<search>one concrete retrieval query</search>

<reason_search>
<reason>reasoning over the question and previous observations</reason>
<query>the next concrete retrieval query enabled by that reasoning</query>
</reason_search>

<answer>the concise final answer</answer>

Rules:
- SEARCH retrieves evidence and is followed by an observation.
- REASON_SEARCH is one atomic action: both reason and query are required.
- There is no standalone REASON action. Reasoning must either enable the next search or be omitted.
- Use only information from the question and observations.
- Do not invent observations, tool results, node IDs, plans, or agent roles.
- Keep the final answer short and do not include an explanation inside <answer>.
"""


class LinearActionKind(str, Enum):
    SEARCH = "search"
    REASON_SEARCH = "reason_search"
    ANSWER = "answer"


@dataclass(frozen=True)
class LinearAction:
    kind: LinearActionKind
    query: str = ""
    reason: str = ""
    answer: str = ""


@dataclass
class LinearTrajectoryState:
    """Track causal action order and enforce the retrieval budget."""

    max_tool_calls: int
    action_count: int = 0
    tool_calls: int = 0
    reason_search_count: int = 0
    answer: str = ""
    invalid_reason: str = ""

    @property
    def terminated(self) -> bool:
        return bool(self.answer or self.invalid_reason)

    @property
    def valid(self) -> bool:
        return bool(self.answer) and not self.invalid_reason

    def apply(self, action: LinearAction) -> bool:
        """Apply one action and return whether a retrieval should execute."""

        if self.terminated:
            raise ValueError("cannot append an action to a terminated trajectory")
        self.action_count += 1
        if action.kind == LinearActionKind.ANSWER:
            self.answer = action.answer
            return False
        if self.tool_calls >= self.max_tool_calls:
            self.invalid_reason = "tool_budget_exhausted"
            return False
        self.tool_calls += 1
        if action.kind == LinearActionKind.REASON_SEARCH:
            self.reason_search_count += 1
        return True

    def invalidate(self, reason: str) -> None:
        if not self.terminated:
            self.invalid_reason = str(reason or "invalid_action")


_SEARCH_RE = re.compile(
    r"\s*<search>\s*(.*?)\s*</search>\s*", re.DOTALL | re.IGNORECASE
)
_ANSWER_RE = re.compile(
    r"\s*<answer>\s*(.*?)\s*</answer>\s*", re.DOTALL | re.IGNORECASE
)
_REASON_SEARCH_RE = re.compile(
    r"\s*<reason_search>\s*<reason>\s*(.*?)\s*</reason>\s*"
    r"<query>\s*(.*?)\s*</query>\s*</reason_search>\s*",
    re.DOTALL | re.IGNORECASE,
)


def _clean_value(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def parse_linear_action(text: str) -> LinearAction:
    """Parse one complete action, rejecting free text and standalone reasoning."""

    raw = str(text or "").strip()
    match = _SEARCH_RE.fullmatch(raw)
    if match:
        query = _clean_value(match.group(1))
        if not query:
            raise ValueError("empty_search_query")
        return LinearAction(LinearActionKind.SEARCH, query=query)

    match = _REASON_SEARCH_RE.fullmatch(raw)
    if match:
        reason = _clean_value(match.group(1))
        query = _clean_value(match.group(2))
        if not reason:
            raise ValueError("empty_reason")
        if not query:
            raise ValueError("empty_reason_search_query")
        return LinearAction(
            LinearActionKind.REASON_SEARCH,
            query=query,
            reason=reason,
        )

    match = _ANSWER_RE.fullmatch(raw)
    if match:
        answer = _clean_value(match.group(1))
        if not answer:
            raise ValueError("empty_answer")
        return LinearAction(LinearActionKind.ANSWER, answer=answer)

    if re.search(r"<reason(?:\s|>)", raw, re.IGNORECASE):
        raise ValueError("standalone_reason_is_not_an_action")
    raise ValueError("malformed_linear_action")


def build_linear_messages(question: str) -> list[dict[str, str]]:
    question = str(question or "").strip()
    return [
        {"role": "system", "content": LINEAR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]


def format_search_observation(documents: Iterable[str], step: int) -> str:
    docs = [str(document).strip() for document in documents if str(document).strip()]
    body = "\n".join(docs) if docs else "[No retrieval result]"
    return f'<observation search_step="{int(step)}">\n{body}\n</observation>'


def exact_match_answer(prediction: str, references) -> float:
    if isinstance(references, str):
        candidates = [references]
    elif isinstance(references, (list, tuple, set)):
        candidates = list(references)
    else:
        candidates = [references]
    normalized_prediction = normalize_answer(str(prediction or ""))
    return float(
        bool(normalized_prediction)
        and any(
            normalized_prediction == normalize_answer(str(item or ""))
            for item in candidates
        )
    )
