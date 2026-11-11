"""Online Planner preference signal from sibling workflow outcomes.

The pairwise logistic loss is linearized at the rollout policy. For the
single on-policy AdamW update used by DAGENT, the resulting token coefficients
produce exactly the same first-order gradient while allowing pair members to
live in different dynamic microbatches.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


PLANNER_ROLE = "PlanningAgent"


@dataclass(frozen=True)
class PlannerOutcomeContrastResult:
    coefficients: torch.Tensor
    assigned_outcomes: dict[object, float]
    metrics: dict[str, float]


def _stable_question_seed(seed: int, question_id: object) -> int:
    digest = hashlib.sha256(str(question_id).encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], byteorder="little", signed=False)


def _shuffle_binary_outcomes(
    outcomes: list[float], *, seed: int, question_id: object
) -> list[float]:
    """Permute labels within a question while guaranteeing a changed assignment."""

    if len(outcomes) < 2 or len(set(outcomes)) < 2:
        return list(outcomes)
    rng = random.Random(_stable_question_seed(seed, question_id))
    indices = list(range(len(outcomes)))
    for _ in range(32):
        rng.shuffle(indices)
        shuffled = [outcomes[index] for index in indices]
        if shuffled != outcomes:
            return shuffled
    # A one-position rotation always changes a mixed binary assignment at some row.
    return outcomes[1:] + outcomes[:1]


def build_planner_outcome_contrast_coefficients(
    *,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    question_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    roles: np.ndarray,
    outcomes: np.ndarray,
    workflow_steps: np.ndarray,
    chosen_eligibility: np.ndarray | None = None,
    contract_validity: np.ndarray | None = None,
    runtime_lineage_validity: np.ndarray | None = None,
    answer_sink_support: np.ndarray | None = None,
    valid_rows: torch.Tensor | np.ndarray | None = None,
    beta: float = 1.0,
    data_parallel_size: int = 1,
    shuffled: bool = False,
    shuffle_seed: int = 0,
    pairing_mode: str = "same_question",
    question_weighting: str = "question",
    length_normalization: bool = True,
    pair_selection: str = "all",
    remove_shared_prefix: bool = False,
    response_token_ids: torch.Tensor | None = None,
    prompt_token_ids: torch.Tensor | None = None,
    prompt_attention_mask: torch.Tensor | None = None,
) -> PlannerOutcomeContrastResult:
    """Build token coefficients for an equal-question Planner contrast loss.

    For each question, every successful Planner sequence is paired with every
    unsuccessful sibling. Sequence scores are mean response-token log
    probabilities. Coefficients are the exact derivative of the pairwise
    logistic loss at ``old_log_probs`` and are normalized first over pairs in a
    question and then over informative questions.

    Duplicate rows introduced by complete-workflow DP padding share one logical
    action's coefficient. ``data_parallel_size`` compensates for the gradient
    average across actor data-parallel ranks.
    """

    if response_mask.ndim != 2:
        raise ValueError("POC expects a rank-2 response_mask")
    if old_log_probs.shape != response_mask.shape:
        raise ValueError("old_log_probs must match response_mask")
    if remove_shared_prefix:
        if response_token_ids is None or response_token_ids.shape != response_mask.shape:
            raise ValueError("shared-prefix masking requires response token IDs")
        if prompt_token_ids is None or prompt_attention_mask is None:
            raise ValueError("shared-prefix masking requires prompt IDs and mask")
        if (prompt_token_ids.shape != prompt_attention_mask.shape
                or prompt_token_ids.shape[0] != response_mask.shape[0]):
            raise ValueError("prompt token IDs and masks must align with batch rows")
    if beta <= 0:
        raise ValueError("POC beta must be positive")
    if data_parallel_size < 1:
        raise ValueError("POC data_parallel_size must be positive")
    if pairing_mode not in {"same_question", "cross_question"}:
        raise ValueError("POC pairing_mode must be same_question or cross_question")
    if question_weighting not in {"question", "all_questions", "pair"}:
        raise ValueError("POC question_weighting must be question, all_questions or pair")
    if pair_selection not in {"all", "one_random"}:
        raise ValueError("POC pair_selection must be all or one_random")
    if pairing_mode == "cross_question" and pair_selection == "one_random":
        raise ValueError("one_random pair_selection requires same_question pairing")
    if not torch.isfinite(old_log_probs).all():
        raise ValueError("POC old_log_probs contain non-finite values")

    row_count = response_mask.shape[0]
    metadata = {
        "question_ids": np.asarray(question_ids, dtype=object),
        "trajectory_ids": np.asarray(trajectory_ids, dtype=object),
        "roles": np.asarray(roles, dtype=object),
        "outcomes": np.asarray(outcomes, dtype=np.float64),
        "workflow_steps": np.asarray(workflow_steps, dtype=np.int64),
    }
    if chosen_eligibility is None:
        metadata["chosen_eligibility"] = np.ones(row_count, dtype=np.bool_)
    else:
        metadata["chosen_eligibility"] = np.asarray(
            chosen_eligibility, dtype=np.bool_
        )
    for name, values in (
        ("contract_validity", contract_validity),
        ("runtime_lineage_validity", runtime_lineage_validity),
        ("answer_sink_support", answer_sink_support),
    ):
        if values is not None:
            metadata[name] = np.asarray(values, dtype=np.bool_)
    for name, values in metadata.items():
        if values.ndim != 1 or values.shape[0] != row_count:
            raise ValueError(f"POC {name} must contain one value per batch row")
    if not np.all(np.isfinite(metadata["outcomes"])):
        raise ValueError("POC outcomes contain non-finite values")
    if not np.all(
        np.isclose(metadata["outcomes"], 0.0, atol=1e-6)
        | np.isclose(metadata["outcomes"], 1.0, atol=1e-6)
    ):
        raise ValueError("POC requires binary exact-match outcomes")

    if valid_rows is None:
        valid_row_mask = np.ones(row_count, dtype=np.bool_)
    elif isinstance(valid_rows, torch.Tensor):
        valid_row_mask = valid_rows.detach().cpu().numpy().astype(np.bool_, copy=False)
    else:
        valid_row_mask = np.asarray(valid_rows, dtype=np.bool_)
    valid_row_mask = valid_row_mask.reshape(-1)
    if valid_row_mask.shape[0] != row_count:
        raise ValueError("POC valid_rows must contain one value per batch row")
    if not np.any(valid_row_mask):
        raise ValueError("POC valid_rows excludes every batch row")
    all_question_count = len(set(metadata["question_ids"][valid_row_mask].tolist()))

    workflow_rows: dict[object, list[int]] = defaultdict(list)
    workflow_question: dict[object, object] = {}
    workflow_outcome: dict[object, float] = {}
    workflow_chosen_eligible: dict[object, bool] = {}
    for row, (question_id, trajectory_id, outcome, chosen_eligible) in enumerate(
        zip(
            metadata["question_ids"],
            metadata["trajectory_ids"],
            metadata["outcomes"],
            metadata["chosen_eligibility"],
            strict=True,
        )
    ):
        if not valid_row_mask[row]:
            continue
        previous_question = workflow_question.setdefault(trajectory_id, question_id)
        if previous_question != question_id:
            raise ValueError(f"POC trajectory {trajectory_id!r} belongs to multiple questions")
        previous_outcome = workflow_outcome.setdefault(trajectory_id, float(outcome))
        if not np.isclose(previous_outcome, outcome, atol=1e-6, rtol=0.0):
            raise ValueError(f"POC trajectory {trajectory_id!r} has inconsistent outcomes")
        previous_eligibility = workflow_chosen_eligible.setdefault(
            trajectory_id, bool(chosen_eligible)
        )
        if previous_eligibility != bool(chosen_eligible):
            raise ValueError(
                f"POC trajectory {trajectory_id!r} has inconsistent chosen eligibility"
            )
        workflow_rows[trajectory_id].append(row)

    workflow_audit_flags: dict[str, dict[object, bool]] = {}
    for name in (
        "contract_validity",
        "runtime_lineage_validity",
        "answer_sink_support",
    ):
        if name not in metadata:
            continue
        workflow_values: dict[object, bool] = {}
        for row, (trajectory_id, value) in enumerate(
            zip(metadata["trajectory_ids"], metadata[name], strict=True)
        ):
            if not valid_row_mask[row]:
                continue
            previous = workflow_values.setdefault(trajectory_id, bool(value))
            if previous != bool(value):
                raise ValueError(
                    f"POC trajectory {trajectory_id!r} has inconsistent {name}"
                )
        workflow_audit_flags[name] = workflow_values

    planner_rows: dict[object, list[int]] = {}
    duplicate_row_count = 0
    mask = response_mask.bool()
    for trajectory_id, rows in workflow_rows.items():
        by_step: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            if str(metadata["roles"][row]) == PLANNER_ROLE:
                by_step[int(metadata["workflow_steps"][row])].append(row)
        if not by_step:
            continue
        if len(by_step) != 1:
            raise ValueError(
                f"POC trajectory {trajectory_id!r} has {len(by_step)} logical Planner actions"
            )
        duplicates = next(iter(by_step.values()))
        token_counts = [int(mask[row].sum().item()) for row in duplicates]
        if not token_counts or token_counts[0] < 1:
            raise ValueError(f"POC trajectory {trajectory_id!r} has an empty Planner response")
        if any(count != token_counts[0] for count in token_counts[1:]):
            raise ValueError(
                f"POC trajectory {trajectory_id!r} duplicate Planner rows disagree on length"
            )
        representative = duplicates[0]
        representative_values = old_log_probs[representative][mask[representative]]
        for duplicate in duplicates[1:]:
            duplicate_values = old_log_probs[duplicate][mask[duplicate]]
            if not torch.allclose(
                representative_values, duplicate_values, atol=1e-6, rtol=0.0
            ):
                raise ValueError(
                    f"POC trajectory {trajectory_id!r} duplicate Planner rows disagree on log-prob"
                )
            if remove_shared_prefix:
                if not torch.equal(response_token_ids[representative][mask[representative]].cpu(),
                                   response_token_ids[duplicate][mask[duplicate]].cpu()):
                    raise ValueError("duplicate Planner response token IDs disagree")
                if not torch.equal(
                    prompt_token_ids[representative][prompt_attention_mask[representative].bool()].cpu(),
                    prompt_token_ids[duplicate][prompt_attention_mask[duplicate].bool()].cpu(),
                ):
                    raise ValueError("duplicate Planner prompt token IDs disagree")
        planner_rows[trajectory_id] = duplicates
        duplicate_row_count += len(duplicates) - 1

    raw_question_workflows: dict[object, list[object]] = defaultdict(list)
    for trajectory_id in planner_rows:
        raw_question_workflows[workflow_question[trajectory_id]].append(trajectory_id)
    raw_mixed_question_count = 0
    for workflows in raw_question_workflows.values():
        raw_outcomes = [workflow_outcome[item] for item in workflows]
        raw_mixed_question_count += int(
            any(value > 0.5 for value in raw_outcomes)
            and any(value < 0.5 for value in raw_outcomes)
        )
    raw_positive_workflow_count = sum(
        workflow_outcome[item] > 0.5 for item in planner_rows
    )
    raw_eligible_workflow_count = sum(
        workflow_chosen_eligible[item] for item in planner_rows
    )
    raw_eligible_positive_workflow_count = sum(
        workflow_outcome[item] > 0.5 and workflow_chosen_eligible[item]
        for item in planner_rows
    )

    ineligible_positive_workflows = {
        trajectory_id
        for trajectory_id in planner_rows
        if workflow_outcome[trajectory_id] > 0.5
        and not workflow_chosen_eligible[trajectory_id]
    }
    for trajectory_id in ineligible_positive_workflows:
        planner_rows.pop(trajectory_id)

    question_workflows: dict[object, list[object]] = defaultdict(list)
    for trajectory_id in planner_rows:
        question_workflows[workflow_question[trajectory_id]].append(trajectory_id)
    for workflows in question_workflows.values():
        workflows.sort(key=str)

    assigned_outcomes = dict(workflow_outcome)
    shuffled_label_count = 0
    if shuffled:
        for question_id, workflows in question_workflows.items():
            original = [workflow_outcome[workflow_id] for workflow_id in workflows]
            permuted = _shuffle_binary_outcomes(
                original, seed=shuffle_seed, question_id=question_id
            )
            for workflow_id, old_value, new_value in zip(
                workflows, original, permuted, strict=True
            ):
                assigned_outcomes[workflow_id] = new_value
                shuffled_label_count += int(old_value != new_value)

    informative_groups: list[tuple[object, list[tuple[object, object]]]] = []
    if pairing_mode == "same_question":
        for question_id, workflows in question_workflows.items():
            chosen = [item for item in workflows if assigned_outcomes[item] > 0.5]
            rejected = [item for item in workflows if assigned_outcomes[item] < 0.5]
            if not chosen or not rejected:
                continue
            if pair_selection == "one_random":
                rng = random.Random(_stable_question_seed(shuffle_seed, question_id))
                pairs = [(rng.choice(chosen), rng.choice(rejected))]
            else:
                pairs = [
                    (chosen_id, rejected_id)
                    for chosen_id in chosen
                    for rejected_id in rejected
                ]
            informative_groups.append((question_id, pairs))
    else:
        chosen = [
            trajectory_id
            for trajectory_id in planner_rows
            if assigned_outcomes[trajectory_id] > 0.5
        ]
        rejected = [
            trajectory_id
            for trajectory_id in planner_rows
            if assigned_outcomes[trajectory_id] < 0.5
        ]
        pairs = [
            (chosen_id, rejected_id)
            for chosen_id in chosen
            for rejected_id in rejected
            if workflow_question[chosen_id] != workflow_question[rejected_id]
        ]
        if pairs:
            informative_groups.append(("cross_question", pairs))

    pair_masks = {}
    skipped_empty_prefix_pairs = 0
    if remove_shared_prefix:
        filtered_groups = []
        for question_id, pairs in informative_groups:
            filtered_pairs = []
            for chosen_id, rejected_id in pairs:
                chosen_row, rejected_row = planner_rows[chosen_id][0], planner_rows[rejected_id][0]
                left_prompt = prompt_token_ids[chosen_row][prompt_attention_mask[chosen_row].bool()]
                right_prompt = prompt_token_ids[rejected_row][prompt_attention_mask[rejected_row].bool()]
                if not torch.equal(left_prompt.cpu(), right_prompt.cpu()):
                    raise ValueError("shared-prefix comparison requires identical Planner prompts")
                left_mask, right_mask = mask[chosen_row].clone(), mask[rejected_row].clone()
                left_positions = torch.nonzero(left_mask).flatten()
                right_positions = torch.nonzero(right_mask).flatten()
                left_ids = response_token_ids[chosen_row][left_positions]
                right_ids = response_token_ids[rejected_row][right_positions]
                prefix = 0
                for left_id, right_id in zip(left_ids.tolist(), right_ids.tolist()):
                    if left_id != right_id:
                        break
                    prefix += 1
                left_mask[left_positions[:prefix]] = False
                right_mask[right_positions[:prefix]] = False
                if not left_mask.any() or not right_mask.any():
                    skipped_empty_prefix_pairs += 1
                    continue
                pair_masks[(chosen_id, rejected_id)] = (left_mask, right_mask)
                filtered_pairs.append((chosen_id, rejected_id))
            if filtered_pairs:
                filtered_groups.append((question_id, filtered_pairs))
        informative_groups = filtered_groups

    pair_count = sum(len(pairs) for _, pairs in informative_groups)

    coefficients = torch.zeros_like(old_log_probs, dtype=torch.float32)
    losses: list[torch.Tensor] = []
    if informative_groups:
        total_pairs = max(pair_count, 1)
        for _, pairs in informative_groups:
            if question_weighting == "question":
                pair_scale = 1.0 / len(informative_groups) / len(pairs)
            elif question_weighting == "all_questions":
                pair_scale = 1.0 / all_question_count / len(pairs)
            else:
                pair_scale = 1.0 / total_pairs
            for chosen_id, rejected_id in pairs:
                chosen_row = planner_rows[chosen_id][0]
                chosen_mask = (pair_masks[(chosen_id, rejected_id)][0]
                               if remove_shared_prefix else mask[chosen_row])
                chosen_values = old_log_probs[chosen_row][chosen_mask].float()
                chosen_score = (
                    chosen_values.mean() if length_normalization else chosen_values.sum()
                )
                rejected_row = planner_rows[rejected_id][0]
                rejected_mask = (pair_masks[(chosen_id, rejected_id)][1]
                                 if remove_shared_prefix else mask[rejected_row])
                rejected_values = old_log_probs[rejected_row][rejected_mask].float()
                rejected_score = (
                    rejected_values.mean()
                    if length_normalization
                    else rejected_values.sum()
                )
                margin = beta * (chosen_score - rejected_score)
                losses.append(F.softplus(-margin) * pair_scale)
                slope = beta * torch.sigmoid(-margin).item() * pair_scale

                for workflow_id, sign in ((chosen_id, -1.0), (rejected_id, 1.0)):
                    duplicates = planner_rows[workflow_id]
                    action_mask = chosen_mask if sign < 0 else rejected_mask
                    token_count = int(action_mask.sum().item())
                    length_scale = token_count if length_normalization else 1
                    value = (
                        sign
                        * slope
                        * data_parallel_size
                        / (length_scale * len(duplicates))
                    )
                    for row in duplicates:
                        if remove_shared_prefix and not torch.equal(mask[row], mask[duplicates[0]]):
                            raise ValueError("duplicate Planner rows disagree on token mask")
                        coefficients[row][action_mask] += value

    non_planner = torch.as_tensor(
        [str(role) != PLANNER_ROLE for role in metadata["roles"]],
        device=coefficients.device,
        dtype=torch.bool,
    )
    old_loss = float(torch.stack(losses).sum().item()) if losses else 0.0
    metrics = {
        "aux/planner_outcome_contrast/raw_planner_workflow_count": float(
            len(planner_rows) + len(ineligible_positive_workflows)
        ),
        "aux/planner_outcome_contrast/raw_question_count": float(
            len(raw_question_workflows)
        ),
        "aux/planner_outcome_contrast/all_question_count": float(all_question_count),
        "aux/planner_outcome_contrast/raw_positive_workflow_count": float(
            raw_positive_workflow_count
        ),
        "aux/planner_outcome_contrast/raw_eligible_workflow_count": float(
            raw_eligible_workflow_count
        ),
        "aux/planner_outcome_contrast/raw_eligible_positive_workflow_count": float(
            raw_eligible_positive_workflow_count
        ),
        "aux/planner_outcome_contrast/raw_mixed_question_count": float(
            raw_mixed_question_count
        ),
        "aux/planner_outcome_contrast/planner_workflow_count": float(len(planner_rows)),
        "aux/planner_outcome_contrast/question_count": float(len(question_workflows)),
        "aux/planner_outcome_contrast/informative_question_count": float(
            len(informative_groups)
        ),
        "aux/planner_outcome_contrast/pair_count": float(pair_count),
        "aux/planner_outcome_contrast/shared_prefix_empty_pairs": float(skipped_empty_prefix_pairs),
        "aux/planner_outcome_contrast/duplicate_planner_row_count": float(
            duplicate_row_count
        ),
        "aux/planner_outcome_contrast/ineligible_positive_workflow_count": float(
            len(ineligible_positive_workflows)
        ),
        "aux/planner_outcome_contrast/shuffled": float(shuffled),
        "aux/planner_outcome_contrast/shuffled_label_count": float(
            shuffled_label_count
        ),
        "aux/planner_outcome_contrast/loss_at_old_policy": old_loss,
        "aux/planner_outcome_contrast/coefficient_l1": float(
            coefficients.abs().sum().item()
        ),
        "aux/planner_outcome_contrast/coefficient_sum": float(coefficients.sum().item()),
        "aux/planner_outcome_contrast/non_planner_nonzero_tokens": float(
            coefficients[non_planner].ne(0).sum().item()
        ),
        "aux/planner_outcome_contrast/ignored_padding_rows": float(
            row_count - int(valid_row_mask.sum())
        ),
    }
    all_raw_planner_workflows = set(planner_rows) | ineligible_positive_workflows
    for name, output_name in (
        ("contract_validity", "contract_valid_workflow_count"),
        ("runtime_lineage_validity", "runtime_lineage_valid_workflow_count"),
        ("answer_sink_support", "answer_sink_supported_workflow_count"),
    ):
        if name in workflow_audit_flags:
            metrics[f"aux/planner_outcome_contrast/{output_name}"] = float(
                sum(
                    workflow_audit_flags[name][trajectory_id]
                    for trajectory_id in all_raw_planner_workflows
                )
            )
            metrics[
                f"aux/planner_outcome_contrast/positive_{output_name}"
            ] = float(
                sum(
                    workflow_outcome[trajectory_id] > 0.5
                    and workflow_audit_flags[name][trajectory_id]
                    for trajectory_id in all_raw_planner_workflows
                )
            )
    component_names = (
        "contract_validity",
        "runtime_lineage_validity",
        "answer_sink_support",
    )
    if all(name in workflow_audit_flags for name in component_names):
        metrics[
            "aux/planner_outcome_contrast/eligibility_component_mismatch_workflow_count"
        ] = float(
            sum(
                workflow_chosen_eligible[trajectory_id]
                != all(
                    workflow_audit_flags[name][trajectory_id]
                    for name in component_names
                )
                for trajectory_id in all_raw_planner_workflows
            )
        )
    return PlannerOutcomeContrastResult(
        coefficients=coefficients,
        assigned_outcomes=assigned_outcomes,
        metrics=metrics,
    )


def planner_outcome_contrast_loss(
    log_probs: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """Return the linearized POC loss for one full batch or microbatch."""

    if log_probs.shape != coefficients.shape:
        raise ValueError("POC log_probs and coefficients must have the same shape")
    return torch.sum(log_probs.float() * coefficients.to(log_probs.device).float())


def summarize_parameter_gradient_audit(
    *,
    em_squared_norm: float,
    real_squared_norm: float,
    shuffled_squared_norm: float,
    real_em_dot: float,
    shuffled_em_dot: float,
    real_shuffled_dot: float,
    loss_weight: float,
) -> dict[str, float]:
    """Convert global gradient inner products into stable audit metrics."""

    values = {
        "em_squared_norm": em_squared_norm,
        "real_squared_norm": real_squared_norm,
        "shuffled_squared_norm": shuffled_squared_norm,
        "real_em_dot": real_em_dot,
        "shuffled_em_dot": shuffled_em_dot,
        "real_shuffled_dot": real_shuffled_dot,
        "loss_weight": loss_weight,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("POC parameter-gradient audit received non-finite statistics")
    if min(em_squared_norm, real_squared_norm, shuffled_squared_norm) < 0:
        raise ValueError("POC parameter-gradient squared norms must be non-negative")
    if loss_weight <= 0:
        raise ValueError("POC parameter-gradient audit loss_weight must be positive")

    em_norm = math.sqrt(em_squared_norm)
    real_norm = math.sqrt(real_squared_norm)
    shuffled_norm = math.sqrt(shuffled_squared_norm)

    def cosine(dot: float, left_norm: float, right_norm: float) -> float:
        denominator = left_norm * right_norm
        if denominator <= 0:
            return 0.0
        return max(-1.0, min(1.0, dot / denominator))

    real_em_cosine = cosine(real_em_dot, real_norm, em_norm)
    shuffled_em_cosine = cosine(shuffled_em_dot, shuffled_norm, em_norm)
    real_shuffled_cosine = cosine(
        real_shuffled_dot, real_norm, shuffled_norm
    )
    em_denominator = max(em_norm, 1e-30)
    shuffled_denominator = max(shuffled_norm, 1e-30)
    return {
        "aux/planner_outcome_contrast/parameter_em_gradient_norm": em_norm,
        "aux/planner_outcome_contrast/parameter_real_poc_gradient_norm": real_norm,
        "aux/planner_outcome_contrast/parameter_weighted_real_poc_gradient_norm": (
            real_norm * loss_weight
        ),
        "aux/planner_outcome_contrast/parameter_shuffled_poc_gradient_norm": shuffled_norm,
        "aux/planner_outcome_contrast/parameter_real_poc_to_em_gradient_ratio": (
            real_norm / em_denominator
        ),
        "aux/planner_outcome_contrast/parameter_weighted_real_poc_to_em_gradient_ratio": (
            real_norm * loss_weight / em_denominator
        ),
        "aux/planner_outcome_contrast/parameter_shuffled_poc_to_em_gradient_ratio": (
            shuffled_norm / em_denominator
        ),
        "aux/planner_outcome_contrast/parameter_real_to_shuffled_gradient_norm_ratio": (
            real_norm / shuffled_denominator
        ),
        "aux/planner_outcome_contrast/parameter_real_poc_em_cosine": real_em_cosine,
        "aux/planner_outcome_contrast/parameter_shuffled_poc_em_cosine": shuffled_em_cosine,
        "aux/planner_outcome_contrast/parameter_real_shuffled_cosine": real_shuffled_cosine,
        "aux/planner_outcome_contrast/parameter_real_minus_shuffled_em_cosine": (
            real_em_cosine - shuffled_em_cosine
        ),
    }
