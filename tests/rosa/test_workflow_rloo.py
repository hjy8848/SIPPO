import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = Path(__file__).parents[2] / "verl" / "trainer" / "ppo" / "workflow_rloo.py"
SPEC = importlib.util.spec_from_file_location("workflow_rloo_impl", MODULE_PATH)
WORKFLOW_RLOO = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WORKFLOW_RLOO)
compute_workflow_rloo_outcome_advantage = WORKFLOW_RLOO.compute_workflow_rloo_outcome_advantage


def _terminal_rewards(values, response_length=3):
    rewards = torch.zeros((len(values), response_length), dtype=torch.float32)
    rewards[:, -1] = torch.tensor(values, dtype=torch.float32)
    return rewards


def test_workflow_rloo_compares_complete_workflows_not_role_rows():
    rewards = _terminal_rewards([1.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.75, 0.75])
    response_mask = torch.ones_like(rewards)
    question_ids = np.array(["q1"] * 5 + ["q2"] * 3, dtype=object)
    trajectory_ids = np.array(
        ["q1-a", "q1-a", "q1-b", "q1-b", "q1-b", "q2-a", "q2-b", "q2-b"],
        dtype=object,
    )

    advantages, returns = compute_workflow_rloo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=question_ids,
        trajectory_ids=trajectory_ids,
    )

    expected = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, -0.5, 0.5, 0.5])
    assert torch.allclose(advantages[:, 0], expected)
    assert torch.allclose(advantages, returns)
    assert torch.allclose(advantages, expected.unsqueeze(-1).expand_as(advantages))


def test_workflow_rloo_masks_padding_tokens():
    rewards = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    advantages, _ = compute_workflow_rloo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["q", "q"], dtype=object),
        trajectory_ids=np.array(["a", "b"], dtype=object),
    )

    assert torch.equal(advantages[0], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(advantages[1], torch.tensor([-1.0, 0.0, 0.0]))


def test_workflow_rloo_rejects_role_rewards_that_disagree_within_workflow():
    with pytest.raises(ValueError, match="inconsistent outcome rewards"):
        compute_workflow_rloo_outcome_advantage(
            token_level_rewards=_terminal_rewards([1.0, 0.5, 0.0]),
            response_mask=torch.ones((3, 3)),
            index=np.array(["q", "q", "q"], dtype=object),
            trajectory_ids=np.array(["a", "a", "b"], dtype=object),
        )


def test_workflow_rloo_zeros_single_workflow_groups():
    advantages, _ = compute_workflow_rloo_outcome_advantage(
        token_level_rewards=_terminal_rewards([0.8, 0.8]),
        response_mask=torch.ones((2, 3)),
        index=np.array(["q", "q"], dtype=object),
        trajectory_ids=np.array(["only", "only"], dtype=object),
    )

    assert torch.count_nonzero(advantages) == 0
