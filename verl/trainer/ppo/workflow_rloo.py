"""Workflow-level advantage estimation for heterogeneous agent trajectories."""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch


def compute_workflow_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_ids: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[Any] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute leave-one-workflow-out advantages for heterogeneous traces.

    A workflow produces several response rows (classification, planning,
    execution, and synthesis). Standard response-level RLOO would treat those
    rows as independent samples and compare roles from the same workflow
    against one another. This estimator first collapses rows by trajectory,
    computes the leave-one-out baseline across complete workflows sampled for
    the same question, and broadcasts the workflow advantage to its tokens.
    """
    del epsilon, config, kwargs

    if index is None or trajectory_ids is None:
        raise ValueError("workflow_rloo requires both question ids and trajectory ids")

    question_ids = np.asarray(index, dtype=object)
    workflow_ids = np.asarray(trajectory_ids, dtype=object)
    if question_ids.shape[0] != token_level_rewards.shape[0]:
        raise ValueError("question id count does not match reward batch size")
    if workflow_ids.shape[0] != token_level_rewards.shape[0]:
        raise ValueError("trajectory id count does not match reward batch size")

    row_scores = token_level_rewards.sum(dim=-1)
    workflow_rows: dict[object, list[int]] = defaultdict(list)
    workflow_question: dict[object, object] = {}

    for row_idx, (question_id, workflow_id) in enumerate(zip(question_ids, workflow_ids, strict=True)):
        previous_question = workflow_question.setdefault(workflow_id, question_id)
        if previous_question != question_id:
            raise ValueError(f"trajectory {workflow_id!r} is assigned to multiple questions")
        workflow_rows[workflow_id].append(row_idx)

    question_workflows: dict[object, list[object]] = defaultdict(list)
    workflow_rewards: dict[object, torch.Tensor] = {}
    for workflow_id, rows in workflow_rows.items():
        rewards = row_scores[rows]
        if not torch.allclose(rewards, rewards[0].expand_as(rewards), atol=1e-6, rtol=0.0):
            raise ValueError(f"trajectory {workflow_id!r} has inconsistent outcome rewards across roles")
        workflow_rewards[workflow_id] = rewards[0]
        question_workflows[workflow_question[workflow_id]].append(workflow_id)

    row_advantages = torch.zeros_like(row_scores)
    with torch.no_grad():
        for workflow_ids_for_question in question_workflows.values():
            workflow_reward_tensor = torch.stack(
                [workflow_rewards[workflow_id] for workflow_id in workflow_ids_for_question]
            )
            workflow_count = workflow_reward_tensor.numel()
            if workflow_count < 2:
                continue

            reward_sum = workflow_reward_tensor.sum()
            for workflow_position, workflow_id in enumerate(workflow_ids_for_question):
                loo_baseline = (reward_sum - workflow_reward_tensor[workflow_position]) / (workflow_count - 1)
                workflow_advantage = workflow_reward_tensor[workflow_position] - loo_baseline
                row_advantages[workflow_rows[workflow_id]] = workflow_advantage

        advantages = row_advantages.unsqueeze(-1) * response_mask

    return advantages, advantages
