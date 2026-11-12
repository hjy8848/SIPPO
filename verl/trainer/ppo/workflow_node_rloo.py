"""Hierarchical workflow and node-level advantage estimation for ROSA."""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch


UNGATED_POSITIVE_TYPES = {
    "search_ig",
    "query_rewrite_delta",
    "planner_hindsight",
    "invalid_action",
}


def _as_numpy(values, name: str, size: int, dtype=None) -> np.ndarray:
    if values is None:
        raise ValueError(f"workflow_node_rloo requires {name}")
    array = np.asarray(values, dtype=dtype)
    if array.shape[0] != size:
        raise ValueError(f"{name} count does not match reward batch size")
    return array


def compute_workflow_node_rloo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_ids: np.ndarray,
    outcome_rewards: np.ndarray,
    outcome_exact: np.ndarray,
    node_rewards: np.ndarray,
    node_reward_types: np.ndarray,
    node_reward_weights: np.ndarray,
    node_reward_max_rms_ratios: np.ndarray,
    node_reward_failed_positive_scales: np.ndarray,
    node_reward_all_equal_caps: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[Any] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine workflow RLOO with bounded, type-normalized node advantages.

    Outcome and process scores intentionally arrive in separate fields. This
    prevents legacy scalar reward addition from obscuring which signal drives
    an update and keeps the strict outcome-only estimator available unchanged.
    """
    reward_dtype = (
        token_level_rewards.dtype if token_level_rewards.is_floating_point() else torch.float32
    )
    del config, kwargs

    row_count = response_mask.shape[0]
    question_ids = _as_numpy(index, "question ids", row_count, dtype=object)
    workflow_ids = _as_numpy(trajectory_ids, "trajectory ids", row_count, dtype=object)
    outcomes = _as_numpy(outcome_rewards, "outcome_rewards", row_count, dtype=np.float64)
    exact = _as_numpy(outcome_exact, "outcome_exact", row_count, dtype=np.float64)
    local_scores = _as_numpy(node_rewards, "node_rewards", row_count, dtype=np.float64).copy()
    local_types = _as_numpy(node_reward_types, "node_reward_types", row_count, dtype=object)
    local_weights = _as_numpy(node_reward_weights, "node_reward_weights", row_count, dtype=np.float64)
    max_rms_ratios = _as_numpy(
        node_reward_max_rms_ratios,
        "node_reward_max_rms_ratios",
        row_count,
        dtype=np.float64,
    )
    failed_positive_scales = _as_numpy(
        node_reward_failed_positive_scales,
        "node_reward_failed_positive_scales",
        row_count,
        dtype=np.float64,
    )
    all_equal_caps = _as_numpy(
        node_reward_all_equal_caps,
        "node_reward_all_equal_caps",
        row_count,
        dtype=np.float64,
    )
    numeric_fields = {
        "outcome_rewards": outcomes,
        "outcome_exact": exact,
        "node_rewards": local_scores,
        "node_reward_weights": local_weights,
        "node_reward_max_rms_ratios": max_rms_ratios,
        "node_reward_failed_positive_scales": failed_positive_scales,
        "node_reward_all_equal_caps": all_equal_caps,
    }
    for field_name, values in numeric_fields.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{field_name} contains a non-finite value")
    if np.any(local_weights < 0.0) or np.any(max_rms_ratios < 0.0) or np.any(all_equal_caps < 0.0):
        raise ValueError("node reward weights and caps must be non-negative")
    if np.any((failed_positive_scales < 0.0) | (failed_positive_scales > 1.0)):
        raise ValueError("node reward failed positive scales must be in [0, 1]")

    workflow_rows: dict[object, list[int]] = defaultdict(list)
    workflow_question: dict[object, object] = {}
    for row_idx, (question_id, workflow_id) in enumerate(zip(question_ids, workflow_ids, strict=True)):
        previous_question = workflow_question.setdefault(workflow_id, question_id)
        if previous_question != question_id:
            raise ValueError(f"trajectory {workflow_id!r} is assigned to multiple questions")
        workflow_rows[workflow_id].append(row_idx)

    workflow_outcomes: dict[object, float] = {}
    question_workflows: dict[object, list[object]] = defaultdict(list)
    for workflow_id, rows in workflow_rows.items():
        values = outcomes[rows]
        if not np.allclose(values, values[0], atol=1e-6, rtol=0.0):
            raise ValueError(f"trajectory {workflow_id!r} has inconsistent outcome rewards across roles")
        workflow_outcomes[workflow_id] = float(values[0])
        question_workflows[workflow_question[workflow_id]].append(workflow_id)

    global_advantage = np.zeros(row_count, dtype=np.float64)
    for workflows in question_workflows.values():
        if len(workflows) < 2:
            continue
        values = np.asarray([workflow_outcomes[workflow_id] for workflow_id in workflows], dtype=np.float64)
        total = float(values.sum())
        for position, workflow_id in enumerate(workflows):
            advantage = values[position] - (total - values[position]) / (len(workflows) - 1)
            global_advantage[workflow_rows[workflow_id]] = advantage

    # Semantic positive rewards on failed trajectories are deliberately weak.
    # Verifiable information gain and deterministic invalid penalties are exempt.
    for row_idx, reward_type in enumerate(local_types):
        if (
            local_scores[row_idx] > 0
            and exact[row_idx] < 0.5
            and str(reward_type) not in UNGATED_POSITIVE_TYPES
        ):
            local_scores[row_idx] *= failed_positive_scales[row_idx]

    normalized_local = np.zeros(row_count, dtype=np.float64)
    type_rows: dict[str, list[int]] = defaultdict(list)
    for row_idx, reward_type in enumerate(local_types):
        reward_type = str(reward_type)
        if reward_type and reward_type != "none":
            type_rows[reward_type].append(row_idx)

    for reward_type, rows in type_rows.items():
        values = local_scores[rows]
        if reward_type == "invalid_action":
            # Formatting failures are deterministic and already share a
            # calibrated scale. Keep their penalty active even when every
            # sampled workflow fails in the same way; centering would erase
            # the only useful cold-start signal.
            normalized_local[rows] = np.clip(values, -1.0, 0.0)
            continue
        if len(rows) < 2:
            continue
        std = float(values.std())
        if std <= epsilon:
            continue
        normalized_local[rows] = np.clip((values - float(values.mean())) / (std + epsilon), -2.0, 2.0)

    weighted_local = normalized_local * local_weights
    global_rms = float(np.sqrt(np.mean(np.square(global_advantage))))
    local_rms = float(np.sqrt(np.mean(np.square(weighted_local))))
    max_rms_ratio = float(np.min(max_rms_ratios)) if row_count else 0.0

    if local_rms > epsilon:
        if global_rms > epsilon:
            allowed_rms = max_rms_ratio * global_rms
            if local_rms > allowed_rms:
                weighted_local *= allowed_rms / local_rms
        else:
            cap = float(np.min(all_equal_caps)) if row_count else 0.0
            weighted_local = np.clip(weighted_local, -cap, cap)

    row_advantage = torch.as_tensor(
        global_advantage + weighted_local,
        dtype=reward_dtype,
        device=response_mask.device,
    )
    advantages = row_advantage.unsqueeze(-1) * response_mask.to(dtype=reward_dtype)
    return advantages, advantages
