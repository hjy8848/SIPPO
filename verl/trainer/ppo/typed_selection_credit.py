"""Bounded EM-advantage weights for causal typed selections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from qa_manager.compute_metrics import compute_scores
from qa_manager.normalize_answers import normalize_answer


REASON_ROLE = "ReasonerAgent"
TYPED_SELECTION_METADATA_KEYS = (
    "typed_selection_credit_eligible",
    "typed_selection_credit_decisive",
)


@dataclass(frozen=True)
class TypedSelectionSignal:
    eligible: bool
    decisive: bool
    factual_em: int
    alternative_em: int
    factual_wins: bool
    alternative_wins: bool


def ensure_typed_selection_metadata(
    non_tensor_batch: dict[str, np.ndarray],
    row_count: int,
) -> None:
    """Represent an absent typed selection explicitly on every action row."""

    for key in TYPED_SELECTION_METADATA_KEYS:
        non_tensor_batch.setdefault(key, np.zeros(row_count, dtype=np.bool_))


def evaluate_terminal_typed_selection(
    *,
    selected_candidate: str,
    alternative_candidates: Sequence[str],
    gold_answers: Sequence[str],
    final_answer: str,
    final_em: float,
    terminal_sink: bool,
    legal_selection: bool,
) -> TypedSelectionSignal:
    """Evaluate the exact, replay-free terminal intervention used by training."""

    selected = str(selected_candidate or "").strip()
    alternatives = [
        str(candidate).strip()
        for candidate in alternative_candidates
        if str(candidate).strip()
    ]
    answers = [str(answer) for answer in gold_answers] or [""]
    factual_em = int(compute_scores([selected], [answers])["em"]) if selected else 0
    alternative_em = max(
        (int(compute_scores([candidate], [answers])["em"]) for candidate in alternatives),
        default=0,
    )
    identity_exact = bool(
        legal_selection
        and terminal_sink
        and selected
        and alternatives
        and normalize_answer(final_answer) == normalize_answer(selected)
        and int(round(float(final_em))) == factual_em
    )
    decisive = bool(identity_exact and factual_em != alternative_em)
    return TypedSelectionSignal(
        eligible=identity_exact,
        decisive=decisive,
        factual_em=factual_em,
        alternative_em=alternative_em,
        factual_wins=bool(decisive and factual_em > alternative_em),
        alternative_wins=bool(decisive and alternative_em > factual_em),
    )


def apply_typed_selection_advantage_weight(
    *,
    advantages: torch.Tensor,
    roles: np.ndarray,
    eligible: np.ndarray,
    decisive: np.ndarray,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Increase only decisive Reason EM advantages without changing their sign."""

    if advantages.ndim != 2:
        raise ValueError("typed-selection credit expects rank-2 advantages")
    if not np.isfinite(alpha) or not 0.0 < alpha <= 0.2:
        raise ValueError("typed-selection credit alpha must be in (0, 0.2]")

    row_count = advantages.shape[0]
    roles = np.asarray(roles, dtype=object).reshape(-1)
    eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
    decisive = np.asarray(decisive, dtype=np.bool_).reshape(-1)
    if any(values.shape[0] != row_count for values in (roles, eligible, decisive)):
        raise ValueError("typed-selection metadata must contain one value per row")
    if np.any(decisive & ~eligible):
        raise ValueError("decisive typed selections must be eligible")
    non_reason_signal = (eligible | decisive) & (roles != REASON_ROLE)
    if np.any(non_reason_signal):
        raise ValueError("typed-selection credit is restricted to ReasonerAgent rows")

    target_rows = torch.as_tensor(
        decisive,
        dtype=torch.bool,
        device=advantages.device,
    )
    scales = torch.ones(row_count, dtype=advantages.dtype, device=advantages.device)
    scales[target_rows] += float(alpha)
    weighted = advantages * scales.unsqueeze(-1)
    sign_changed = ((weighted > 0) != (advantages > 0)) & (advantages != 0)
    if torch.any(sign_changed):
        raise ValueError("typed-selection credit changed an advantage sign")

    targeted = advantages[target_rows]
    return weighted, {
        "aux/typed_selection_credit/enabled": 1.0,
        "aux/typed_selection_credit/alpha": float(alpha),
        "aux/typed_selection_credit/eligible_rows": float(eligible.sum()),
        "aux/typed_selection_credit/decisive_rows": float(decisive.sum()),
        "aux/typed_selection_credit/decisive_rate": float(
            decisive.sum() / max(1, eligible.sum())
        ),
        "aux/typed_selection_credit/positive_target_rows": float(
            (targeted.sum(dim=-1) > 0).sum().item()
        ),
        "aux/typed_selection_credit/negative_target_rows": float(
            (targeted.sum(dim=-1) < 0).sum().item()
        ),
        "aux/typed_selection_credit/zero_target_rows": float(
            (targeted.sum(dim=-1) == 0).sum().item()
        ),
        "aux/typed_selection_credit/sign_changes": 0.0,
    }
