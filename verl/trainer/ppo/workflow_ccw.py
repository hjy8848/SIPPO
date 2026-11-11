"""Positive DAG counterfactual weights for workflow-level EM advantages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence
import unicodedata

import torch

from qa_manager.normalize_answers import normalize_answer


@dataclass(frozen=True)
class WorkflowCCWOutput:
    advantages: torch.Tensor
    row_weights: torch.Tensor
    token_weights: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class StochasticCreditEstimate:
    raw_delta: float
    gated_delta: float
    noise_floor: float
    factual_mean: float
    factual_std: float
    alternative_mean: float
    alternative_std: float
    paired_difference_std: float
    paired_standard_error: float
    identity_em_disagreement_rate: float
    identity_answer_disagreement_rate: float
    informative: bool


PLANNER_ROLE = "PlanningAgent"


def normalize_ccw_identity_answer(answer: object) -> str:
    """Canonicalize harmless spelling accents before replay disagreement checks."""

    decomposed = unicodedata.normalize("NFKD", str(answer))
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return normalize_answer(without_marks)


def compare_identity_replay(
    factual_em: float,
    repeated_em: float,
    factual_answer: object,
    repeated_answer: object,
) -> tuple[bool, bool]:
    """Compare byte-equivalent Plan/suffix executions for the hard identity gate."""

    em_matches = float(factual_em) == float(repeated_em)
    answer_matches = normalize_ccw_identity_answer(
        factual_answer
    ) == normalize_ccw_identity_answer(repeated_answer)
    return em_matches, answer_matches


def _sample_mean_std(values: Sequence[float]) -> tuple[float, float]:
    numeric = [float(value) for value in values]
    if len(numeric) < 2:
        raise ValueError("stochastic credit estimation requires at least two replays")
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("replay outcomes must be finite")
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1)
    return mean, math.sqrt(max(variance, 0.0))


def _pairwise_disagreement_rate(values: Sequence[object]) -> float:
    pair_count = len(values) * (len(values) - 1) // 2
    if pair_count == 0:
        return 0.0
    disagreements = sum(
        values[left] != values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return disagreements / pair_count


def estimate_stochastic_counterfactual_credit(
    factual_em: Sequence[float],
    alternative_em: Sequence[float],
    *,
    factual_answers: Sequence[object] | None = None,
    noise_z: float = 1.0,
    minimum_effect: float = 0.0,
) -> StochasticCreditEstimate:
    """Estimate a paired replay effect with continuous uncertainty shrinkage."""

    if len(factual_em) != len(alternative_em):
        raise ValueError("factual and alternative plans require matched replay counts")
    if noise_z < 0.0 or not math.isfinite(noise_z):
        raise ValueError("noise_z must be finite and non-negative")
    if minimum_effect < 0.0 or not math.isfinite(minimum_effect):
        raise ValueError("minimum_effect must be finite and non-negative")

    factual_mean, factual_std = _sample_mean_std(factual_em)
    alternative_mean, alternative_std = _sample_mean_std(alternative_em)
    replay_count = len(factual_em)
    paired_differences = [
        float(factual) - float(alternative)
        for factual, alternative in zip(factual_em, alternative_em, strict=True)
    ]
    raw_delta, paired_difference_std = _sample_mean_std(paired_differences)
    paired_standard_error = paired_difference_std / math.sqrt(replay_count)
    noise_floor = max(minimum_effect, noise_z * paired_standard_error)
    retained_magnitude = max(abs(raw_delta) - noise_floor, 0.0)
    gated_delta = (
        math.copysign(retained_magnitude, raw_delta)
        if retained_magnitude > 0.0
        else 0.0
    )
    informative = retained_magnitude > 0.0

    if factual_answers is not None:
        if len(factual_answers) != replay_count:
            raise ValueError("factual_answers must match the factual replay count")
        normalized_answers = [
            normalize_ccw_identity_answer(answer) for answer in factual_answers
        ]
        answer_disagreement = _pairwise_disagreement_rate(normalized_answers)
    else:
        answer_disagreement = 0.0

    return StochasticCreditEstimate(
        raw_delta=raw_delta,
        gated_delta=gated_delta,
        noise_floor=noise_floor,
        factual_mean=factual_mean,
        factual_std=factual_std,
        alternative_mean=alternative_mean,
        alternative_std=alternative_std,
        paired_difference_std=paired_difference_std,
        paired_standard_error=paired_standard_error,
        identity_em_disagreement_rate=_pairwise_disagreement_rate(
            [float(value) for value in factual_em]
        ),
        identity_answer_disagreement_rate=answer_disagreement,
        informative=informative,
    )


def apply_workflow_ccw_weights(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    signed_credits: torch.Tensor,
    eligible_rows: torch.Tensor | None = None,
    *,
    alpha: float = math.log(1.5),
    minimum_weight: float = 0.5,
    maximum_weight: float = 1.5,
) -> WorkflowCCWOutput:
    """Scale EM advantages without changing their sign or token support."""

    if advantages.ndim != 2 or response_mask.shape != advantages.shape:
        raise ValueError("advantages and response_mask must be matching rank-2 tensors")
    if signed_credits.ndim != 1 or signed_credits.shape[0] != advantages.shape[0]:
        raise ValueError("signed_credits must contain exactly one value per response row")
    if not 0.0 < minimum_weight <= 1.0 <= maximum_weight:
        raise ValueError("CCW bounds must be positive and contain one")
    if alpha < 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")

    mask = response_mask.bool()
    if eligible_rows is None:
        eligible = torch.ones(advantages.shape[0], dtype=torch.bool, device=advantages.device)
    else:
        if eligible_rows.ndim != 1 or eligible_rows.shape[0] != advantages.shape[0]:
            raise ValueError("eligible_rows must contain exactly one flag per response row")
        eligible = eligible_rows.to(device=advantages.device, dtype=torch.bool)

    detached_credits = signed_credits.detach().to(
        device=advantages.device,
        dtype=advantages.dtype,
    )
    if not torch.isfinite(detached_credits).all():
        raise ValueError("signed_credits contains a non-finite value")

    proposed = torch.exp(detached_credits * alpha).clamp(
        min=minimum_weight,
        max=maximum_weight,
    )
    row_weights = torch.where(eligible, proposed, torch.ones_like(proposed)).detach()
    token_weights = torch.where(
        mask,
        row_weights.unsqueeze(-1),
        torch.ones_like(advantages),
    ).detach()
    weighted_advantages = advantages * token_weights

    active = mask & advantages.ne(0)
    sign_flip_count = int(((advantages * weighted_advantages) < 0)[active].sum().item())
    if sign_flip_count:
        raise RuntimeError("positive CCW weights unexpectedly flipped an EM advantage")
    if not torch.isfinite(weighted_advantages).all():
        raise RuntimeError("CCW produced a non-finite weighted advantage")

    response_tokens = int(mask.sum().item())
    weighted_response_tokens = int(mask.sum().item())
    metrics = {
        "aux/workflow_ccw/enabled": 1.0,
        "aux/workflow_ccw/eligible_rows": float(eligible.sum().item()),
        "aux/workflow_ccw/response_token_count_before": float(response_tokens),
        "aux/workflow_ccw/response_token_count_after": float(weighted_response_tokens),
        "aux/workflow_ccw/token_mass_unchanged": float(response_tokens == weighted_response_tokens),
        "aux/workflow_ccw/sign_flip_count": float(sign_flip_count),
        "aux/workflow_ccw/weight_min": float(row_weights.min().item()),
        "aux/workflow_ccw/weight_max": float(row_weights.max().item()),
        "aux/workflow_ccw/weight_mean": float(row_weights.mean().item()),
        "aux/workflow_ccw/weight_below_one": float(row_weights.lt(1).sum().item()),
        "aux/workflow_ccw/weight_above_one": float(row_weights.gt(1).sum().item()),
        "aux/workflow_ccw/weights_detached": float(not row_weights.requires_grad),
    }
    return WorkflowCCWOutput(
        advantages=weighted_advantages,
        row_weights=row_weights,
        token_weights=token_weights,
        metrics=metrics,
    )


def apply_planner_only_ccw_weights(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    roles: Sequence[object],
    causal_deltas: torch.Tensor,
    probe_valid: torch.Tensor,
    *,
    alpha: float = math.log(1.5),
    minimum_weight: float = 0.5,
    maximum_weight: float = 1.5,
) -> WorkflowCCWOutput:
    """Apply detached counterfactual weights only to Planner response rows."""

    row_count = advantages.shape[0]
    if len(roles) != row_count:
        raise ValueError("roles must contain exactly one value per response row")
    if causal_deltas.ndim != 1 or causal_deltas.shape[0] != row_count:
        raise ValueError("causal_deltas must contain exactly one value per response row")
    if probe_valid.ndim != 1 or probe_valid.shape[0] != row_count:
        raise ValueError("probe_valid must contain exactly one flag per response row")

    mask = response_mask.to(device=advantages.device, dtype=torch.bool)
    active_counts = mask.sum(dim=-1)
    row_advantages = torch.where(
        active_counts.gt(0),
        (advantages * mask).sum(dim=-1) / active_counts.clamp_min(1),
        torch.zeros(row_count, device=advantages.device, dtype=advantages.dtype),
    )
    advantage_sign = torch.sign(row_advantages).detach()
    planner_rows = torch.tensor(
        [str(role) == PLANNER_ROLE for role in roles],
        device=advantages.device,
        dtype=torch.bool,
    )
    eligible = planner_rows & probe_valid.to(
        device=advantages.device,
        dtype=torch.bool,
    ) & advantage_sign.ne(0)
    signed_credits = advantage_sign * causal_deltas.detach().to(
        device=advantages.device,
        dtype=advantages.dtype,
    )

    output = apply_workflow_ccw_weights(
        advantages,
        response_mask,
        signed_credits,
        eligible_rows=eligible,
        alpha=alpha,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )
    output.metrics.update(
        {
            "aux/workflow_ccw/planner_rows": float(planner_rows.sum().item()),
            "aux/workflow_ccw/planner_probe_valid_rows": float(
                (planner_rows & probe_valid.to(device=advantages.device, dtype=torch.bool))
                .sum()
                .item()
            ),
            "aux/workflow_ccw/nonplanner_nonunit_weights": float(
                output.row_weights[~planner_rows].ne(1).sum().item()
            ),
        }
    )
    if output.metrics["aux/workflow_ccw/nonplanner_nonunit_weights"]:
        raise RuntimeError("Planner-only CCW changed a non-Planner row weight")
    return output
