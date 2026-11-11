"""Workflow-level GRPO with bounded node-level advantages."""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.ppo.workflow_node_rloo import (
    compute_workflow_node_rloo_advantage,
)


def _config_bool(config: Optional[Any], key: str, default: bool) -> bool:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return bool(getter(key, default))
    return bool(getattr(config, key, default))


def compute_workflow_node_grpo_advantage(
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
    """Compute GRPO over complete workflows, then add bounded local advantages.

    A DAGENT workflow contains heterogeneous response rows from several agents.
    Standard response-level GRPO would normalize Planner, Search, Reasoner, and
    Summary rows against one another. This estimator first computes one outcome
    per complete workflow, applies GRPO across workflows sampled for the same
    question, and then reuses the audited node-advantage combination.
    """

    del kwargs
    question_ids = np.asarray(index, dtype=object)
    workflow_ids = np.asarray(trajectory_ids, dtype=object)
    outcomes = np.asarray(outcome_rewards, dtype=np.float64)
    row_count = response_mask.shape[0]
    for name, values in {
        "question ids": question_ids,
        "trajectory ids": workflow_ids,
        "outcome rewards": outcomes,
    }.items():
        if values.shape[0] != row_count:
            raise ValueError(f"{name} count does not match reward batch size")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("outcome rewards contain a non-finite value")

    workflow_rows: dict[object, list[int]] = defaultdict(list)
    workflow_question: dict[object, object] = {}
    for row_idx, (question_id, workflow_id) in enumerate(
        zip(question_ids, workflow_ids, strict=True)
    ):
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

    normalize_by_std = _config_bool(config, "norm_adv_by_std_in_grpo", True)
    synthetic_rloo_outcomes = np.zeros(row_count, dtype=np.float64)
    for workflows in question_workflows.values():
        workflow_count = len(workflows)
        if workflow_count < 2:
            continue
        values = np.asarray(
            [workflow_outcomes[workflow_id] for workflow_id in workflows],
            dtype=np.float64,
        )
        centered = values - float(values.mean())
        if normalize_by_std:
            std = float(values.std(ddof=1))
            grpo_advantages = (
                centered / (std + epsilon) if std > epsilon else np.zeros_like(centered)
            )
        else:
            grpo_advantages = centered

        # RLOO(z) = n/(n-1) * (z - mean(z)). Scaling z by (n-1)/n
        # makes the existing workflow/node combiner reproduce GRPO exactly.
        synthetic_values = grpo_advantages * (workflow_count - 1) / workflow_count
        for position, workflow_id in enumerate(workflows):
            synthetic_rloo_outcomes[workflow_rows[workflow_id]] = synthetic_values[position]

    return compute_workflow_node_rloo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=question_ids,
        trajectory_ids=workflow_ids,
        outcome_rewards=synthetic_rloo_outcomes,
        outcome_exact=outcome_exact,
        node_rewards=node_rewards,
        node_reward_types=node_reward_types,
        node_reward_weights=node_reward_weights,
        node_reward_max_rms_ratios=node_reward_max_rms_ratios,
        node_reward_failed_positive_scales=node_reward_failed_positive_scales,
        node_reward_all_equal_caps=node_reward_all_equal_caps,
        epsilon=epsilon,
        config=config,
    )
