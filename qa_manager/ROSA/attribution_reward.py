"""Shared, dependency-light definitions for frozen-model attribution rewards."""

import math
import re
from typing import Iterable

from qa_manager.normalize_answers import normalize_answer


NO_EVIDENCE_DOCS = ["No relevant external evidence was provided."]
SEARCH_MATCHED_ATTRIBUTION_THRESHOLD = 0.5653623938560486
SEARCH_EMPTY_ATTRIBUTION_THRESHOLD = 0.6487774848937988
SUMMARY_ATTRIBUTION_THRESHOLD = 0.8756318092346191


def build_search_attribution_messages(
    system_prompt: str,
    node_query: str,
    docs: Iterable[str],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": str(system_prompt)},
        {
            "role": "user",
            "content": (
                f"**Current Sub-question**: {node_query}\n\n"
                f"**Search Context**:\n" + "\n".join(str(doc) for doc in docs or [])
            ),
        },
    ]


def build_summary_attribution_messages(
    system_prompt: str,
    question: str,
    history: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": str(system_prompt)},
        {
            "role": "user",
            "content": (
                f"**Original Question**: {question}\n\n"
                f"**Resolved Sub-questions**:\n{str(history).strip()}"
            ),
        },
    ]


def build_query_binding_messages(
    system_prompt: str,
    question: str,
    draft_query: str,
    step_results: dict[str, str],
) -> list[dict[str, str]]:
    """Mirror the runtime QueryRewriteAgent prompt for a dependent DAG edge."""
    references = re.findall(r"#(\d+)#", str(draft_query))
    if references:
        content = f"Original Goal: {question}\nDraft: {draft_query}\n"
        for reference in references:
            content += f"# {reference} = {step_results.get(reference, 'UNKNOWN')}\n"
        content += "Rewritten query:"
    else:
        content = str(draft_query)
    return [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user", "content": content},
    ]


def history_as_search_context(history: str) -> list[str]:
    """Expose previously resolved facts through the normal search-context slot."""
    history = str(history).strip()
    if not history:
        return list(NO_EVIDENCE_DOCS)
    return [f"Previously resolved workflow facts:\n{history}"]


def node_dependency_ids(node: dict) -> list[str]:
    task = node.get("task", node.get("query", ""))
    return re.findall(r"#(\d+)#", str(task))


def descendant_closure(nodes: Iterable[dict], root_id: str) -> set[str]:
    """Return a node and every downstream node that transitively depends on it."""
    children: dict[str, set[str]] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        for dependency in node_dependency_ids(node):
            children.setdefault(dependency, set()).add(node_id)
    closure = {str(root_id)}
    frontier = [str(root_id)]
    while frontier:
        current = frontier.pop()
        for child in children.get(current, set()):
            if child not in closure:
                closure.add(child)
                frontier.append(child)
    return closure


def render_workflow_history(
    nodes: Iterable[dict],
    conclusion_overrides: dict[str, str] | None = None,
) -> str:
    """Render node records exactly as the final summarizer sees them."""
    overrides = conclusion_overrides or {}
    blocks = []
    for node in nodes:
        node_id = str(node.get("id", ""))
        node_type = str(node.get("type", "")).upper()
        task = str(node.get("task", node.get("query", ""))).strip()
        conclusion = str(overrides.get(node_id, node.get("conclusion", ""))).strip()
        blocks.append(f"Task {node_id} [{node_type}]: {task}\nConclusion: {conclusion}")
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


def tagged_attribution_target(tag: str, output: str) -> tuple[str, str]:
    if tag not in {"conclusion", "answer"}:
        raise ValueError(f"unsupported attribution target tag: {tag}")
    return f"<{tag}>", f"{str(output).strip()}</{tag}>"


def thresholded_attribution_reward(
    evidence_logprob: float,
    counterfactual_logprob: float,
    threshold: float,
    temperature: float = 1.0,
) -> tuple[float, float]:
    """Return a conservative positive-only reward and its raw attribution margin."""
    if temperature <= 0:
        raise ValueError("attribution reward temperature must be positive")
    margin = float(evidence_logprob) - float(counterfactual_logprob)
    if margin <= float(threshold):
        return 0.0, margin
    return math.tanh((margin - float(threshold)) / temperature), margin


def select_matched_counterfactual_docs(
    search_steps: list[dict],
    target_index: int,
) -> tuple[list, str]:
    """Choose non-leaking, size-matched documents from the same dataset."""
    target = search_steps[target_index]
    target_source = str(target.get("data_source", "")).strip().casefold()
    if not target_source or target_source == "unknown":
        return list(NO_EVIDENCE_DOCS), "no_evidence"

    target_docs = list(target.get("docs") or [])
    target_output = normalize_answer(str(target.get("output", "")))
    target_count = len(target_docs)
    target_chars = sum(len(str(doc)) for doc in target_docs)
    candidates = []
    for candidate_index, candidate in enumerate(search_steps):
        if candidate_index == target_index or candidate.get("question") == target.get("question"):
            continue
        if str(candidate.get("data_source", "")).strip().casefold() != target_source:
            continue
        candidate_docs = list(candidate.get("docs") or [])
        if not candidate_docs or candidate_docs == target_docs:
            continue
        if target_output and any(
            target_output in normalize_answer(str(document)) for document in candidate_docs
        ):
            continue
        candidate_chars = sum(len(str(doc)) for doc in candidate_docs)
        distance = (
            abs(len(candidate_docs) - target_count),
            abs(candidate_chars - target_chars),
            candidate_index,
        )
        candidates.append((distance, candidate_docs))

    if not candidates:
        return list(NO_EVIDENCE_DOCS), "no_evidence"
    _, matched_docs = min(candidates, key=lambda item: item[0])
    return matched_docs, "same_dataset"
