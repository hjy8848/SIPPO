import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "verl" / "trainer" / "ppo" / "workflow_ccw.py"
SPEC = importlib.util.spec_from_file_location("workflow_ccw_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
apply_workflow_ccw_weights = MODULE.apply_workflow_ccw_weights
apply_planner_only_ccw_weights = MODULE.apply_planner_only_ccw_weights
estimate_stochastic_counterfactual_credit = (
    MODULE.estimate_stochastic_counterfactual_credit
)
normalize_ccw_identity_answer = MODULE.normalize_ccw_identity_answer
compare_identity_replay = MODULE.compare_identity_replay


def test_ccw_preserves_advantage_direction_and_token_support():
    advantages = torch.tensor(
        [[1.0, 1.0, 0.0], [-2.0, -2.0, 0.0], [0.0, 0.0, 0.0]]
    )
    response_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=torch.bool
    )
    credits = torch.tensor([1.0, -1.0, 0.5], requires_grad=True)

    output = apply_workflow_ccw_weights(
        advantages,
        response_mask,
        credits,
    )

    assert torch.allclose(output.row_weights, torch.tensor([1.5, 2 / 3, math.sqrt(1.5)]))
    assert torch.equal(output.advantages.ne(0), advantages.ne(0))
    assert output.metrics["aux/workflow_ccw/sign_flip_count"] == 0
    assert output.metrics["aux/workflow_ccw/token_mass_unchanged"] == 1
    assert not output.row_weights.requires_grad
    assert not output.token_weights.requires_grad


def test_ccw_leaves_ineligible_rows_at_one():
    advantages = torch.ones((2, 2))
    response_mask = torch.ones((2, 2), dtype=torch.bool)
    output = apply_workflow_ccw_weights(
        advantages,
        response_mask,
        torch.tensor([1.0, -1.0]),
        eligible_rows=torch.tensor([True, False]),
    )
    assert torch.allclose(output.row_weights, torch.tensor([1.5, 1.0]))


@pytest.mark.parametrize(
    "credits",
    [torch.tensor([float("nan")]), torch.tensor([float("inf")])],
)
def test_ccw_rejects_non_finite_credit(credits):
    with pytest.raises(ValueError, match="non-finite"):
        apply_workflow_ccw_weights(
            torch.ones((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
            credits,
        )


def test_planner_only_ccw_never_changes_other_roles():
    advantages = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]
    )
    output = apply_planner_only_ccw_weights(
        advantages,
        torch.ones_like(advantages, dtype=torch.bool),
        ["PlanningAgent", "ReasonerAgent", "PlanningAgent", "AnswerSummarizationAgent"],
        torch.tensor([1.0, 1.0, 1.0, -1.0], requires_grad=True),
        torch.tensor([True, True, True, True]),
    )

    assert torch.allclose(output.row_weights, torch.tensor([1.5, 1.0, 2 / 3, 1.0]))
    assert output.metrics["aux/workflow_ccw/nonplanner_nonunit_weights"] == 0
    assert not output.row_weights.requires_grad


def test_planner_only_ccw_requires_valid_probe_and_nonzero_em_advantage():
    advantages = torch.tensor([[1.0], [0.0], [-1.0]])
    output = apply_planner_only_ccw_weights(
        advantages,
        torch.ones_like(advantages, dtype=torch.bool),
        ["PlanningAgent"] * 3,
        torch.ones(3),
        torch.tensor([False, True, True]),
    )

    assert torch.allclose(output.row_weights, torch.tensor([1.0, 1.0, 2 / 3]))


def test_stochastic_credit_softly_shrinks_effect_by_paired_standard_error():
    estimate = estimate_stochastic_counterfactual_credit(
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        noise_z=0.5,
    )

    assert estimate.raw_delta == pytest.approx(0.25)
    assert estimate.paired_standard_error == pytest.approx(0.25)
    assert estimate.noise_floor == pytest.approx(0.125)
    assert estimate.gated_delta == pytest.approx(0.125)
    assert estimate.informative


def test_stochastic_credit_gates_effect_inside_noise_band_to_zero():
    estimate = estimate_stochastic_counterfactual_credit(
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        noise_z=1.96,
    )

    assert estimate.raw_delta == pytest.approx(0.25)
    assert estimate.noise_floor > abs(estimate.raw_delta)
    assert estimate.gated_delta == 0.0
    assert not estimate.informative


def test_identity_answer_normalization_ignores_diacritic_only_difference():
    estimate = estimate_stochastic_counterfactual_credit(
        [0, 0],
        [0, 0],
        factual_answers=["LuLiang", "Lüliang"],
    )

    assert normalize_ccw_identity_answer("LuLiang") == normalize_ccw_identity_answer(
        "Lüliang"
    )
    assert estimate.identity_answer_disagreement_rate == 0.0


def test_paired_estimator_uses_within_seed_differences():
    estimate = estimate_stochastic_counterfactual_credit(
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        noise_z=1.0,
    )

    assert estimate.raw_delta == pytest.approx(0.25)
    assert estimate.paired_difference_std == pytest.approx(0.5)
    assert estimate.paired_standard_error == pytest.approx(0.25)
    assert estimate.gated_delta == 0.0


def test_identity_replay_requires_matching_em_and_normalized_answer():
    assert compare_identity_replay(1, 1, "Luliang", "L\u00fcliang") == (True, True)
    assert compare_identity_replay(1, 0, "same", "same") == (False, True)
    assert compare_identity_replay(1, 1, "first", "second") == (True, False)
