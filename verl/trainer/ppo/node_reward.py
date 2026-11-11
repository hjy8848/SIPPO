"""Pure helpers for ROSA information-gain and grounding rewards."""

import math
import re
from typing import Iterable

from qa_manager.normalize_answers import normalize_answer
from qa_manager.ROSA.attribution_reward import (
    NO_EVIDENCE_DOCS,
    SEARCH_EMPTY_ATTRIBUTION_THRESHOLD,
    SEARCH_MATCHED_ATTRIBUTION_THRESHOLD,
    SUMMARY_ATTRIBUTION_THRESHOLD,
    build_search_attribution_messages,
    build_summary_attribution_messages,
    descendant_closure,
    render_workflow_history,
    select_matched_counterfactual_docs,
    tagged_attribution_target,
    thresholded_attribution_reward,
)


TITLE_PATTERN = re.compile(r"\(Title:\s*(.*?)\)\s", flags=re.IGNORECASE)
def extract_document_titles(docs: Iterable[str]) -> list[str]:
    titles = []
    for doc in docs or []:
        match = TITLE_PATTERN.search(str(doc))
        if match:
            title = match.group(1).strip().strip('"')
            if title:
                titles.append(title)
    return titles


def usable_gold_titles(gold_titles: Iterable[str]) -> list[str]:
    """Return title-like labels and reject numeric corpus/node identifiers."""
    usable = []
    for title in gold_titles or []:
        text = str(title).strip().strip('"')
        if text and not text.isdigit():
            usable.append(text)
    return usable


def gold_title_coverage_delta(
    docs: Iterable[str],
    gold_titles: Iterable[str],
    covered_titles: Iterable[str] = (),
) -> tuple[float, set[str], bool]:
    gold = usable_gold_titles(gold_titles)
    if not gold:
        return 0.0, set(covered_titles), False

    normalized_gold = {normalize_answer(title): title for title in gold if normalize_answer(title)}
    normalized_docs = {normalize_answer(title) for title in extract_document_titles(docs)}
    covered = {normalize_answer(title) for title in covered_titles if normalize_answer(title)}
    hits = normalized_docs.intersection(normalized_gold)
    new_hits = hits - covered
    updated = covered.union(hits)
    return len(new_hits) / max(1, len(normalized_gold)), updated, True


def exact_match_any(candidate: str, answers: Iterable[str]) -> bool:
    normalized_candidate = normalize_answer(str(candidate or ""))
    return bool(normalized_candidate) and any(
        normalized_candidate == normalize_answer(str(answer)) for answer in answers or []
    )


def conclusion_is_document_grounded(conclusion: str, docs: Iterable[str]) -> bool:
    normalized_conclusion = normalize_answer(str(conclusion or ""))
    if not normalized_conclusion:
        return False
    return any(normalized_conclusion in normalize_answer(str(doc)) for doc in docs or [])


def verified_search_reward(gold_title_delta: float, conclusion: str, docs: Iterable[str]) -> float:
    """Reward newly retrieved gold evidence only when the conclusion is grounded in it."""
    delta = max(0.0, min(1.0, float(gold_title_delta)))
    return delta if delta and conclusion_is_document_grounded(conclusion, docs) else 0.0


def verified_reason_reward(
    conclusion: str,
    answers: Iterable[str],
    dependency_ids: Iterable[str],
    reason_required: bool,
) -> float:
    """Score a required merge reason with two dependencies and a gold-exact conclusion."""
    dependencies = {str(value) for value in dependency_ids if str(value)}
    if not reason_required or len(dependencies) < 2:
        return 0.0
    return float(exact_match_any(conclusion, answers))


def merge_reason_required(question_type: str) -> bool:
    """Return whether the annotated question type requires merging two facts."""
    normalized_type = str(question_type or "").strip().casefold().replace("-", "_")
    return normalized_type in {"comparison", "bridge_comparison"}


def verified_planner_hindsight(
    search_deltas: Iterable[float],
    reason_rewards: Iterable[float],
    reason_required: bool,
    reason_reward_enabled: bool = True,
) -> float:
    """Combine gold-evidence coverage with verified downstream merge success."""
    coverage = min(1.0, sum(max(0.0, float(value)) for value in search_deltas))
    if not reason_reward_enabled or not reason_required:
        return coverage
    reason_score = max([float(value) for value in reason_rewards] or [0.0])
    return 0.5 * coverage + 0.5 * reason_score


def outcome_gate(score: float, outcome_exact: float, enabled: bool = True) -> float:
    if enabled and float(outcome_exact) < 0.5:
        return 0.0
    return float(score)


def information_gain_score(evidence_logprob: float, counterfactual_logprob: float, temperature: float = 1.0) -> float:
    if temperature <= 0:
        raise ValueError("information gain temperature must be positive")
    return math.tanh((float(evidence_logprob) - float(counterfactual_logprob)) / temperature)


def calibrated_information_gain_score(
    evidence_logprob: float,
    counterfactual_logprob: float,
    threshold: float,
    temperature: float = 1.0,
) -> tuple[float, float]:
    """Center a signed IG reward at a threshold fixed on calibration data."""
    if temperature <= 0:
        raise ValueError("information gain temperature must be positive")
    margin = float(evidence_logprob) - float(counterfactual_logprob)
    return math.tanh((margin - float(threshold)) / temperature), margin


def binary_judge_score(yes_logprob: float, no_logprob: float) -> tuple[float, float]:
    """Return a [-1, 1] score and max-class confidence from two log scores."""
    max_logprob = max(float(yes_logprob), float(no_logprob))
    yes = math.exp(float(yes_logprob) - max_logprob)
    no = math.exp(float(no_logprob) - max_logprob)
    probability_yes = yes / (yes + no)
    score = 2.0 * probability_yes - 1.0
    confidence = max(probability_yes, 1.0 - probability_yes)
    return score, confidence
