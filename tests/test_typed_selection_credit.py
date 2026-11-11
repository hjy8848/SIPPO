import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "trainer"
    / "ppo"
    / "typed_selection_credit.py"
)
SPEC = importlib.util.spec_from_file_location("typed_selection_credit_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
apply_typed_selection_advantage_weight = MODULE.apply_typed_selection_advantage_weight
ensure_typed_selection_metadata = MODULE.ensure_typed_selection_metadata
evaluate_terminal_typed_selection = MODULE.evaluate_terminal_typed_selection


def test_terminal_selection_is_decisive_in_both_em_directions():
    factual = evaluate_terminal_typed_selection(
        selected_candidate="Alice",
        alternative_candidates=["Bob"],
        gold_answers=["Alice"],
        final_answer="Alice",
        final_em=1.0,
        terminal_sink=True,
        legal_selection=True,
    )
    counterfactual = evaluate_terminal_typed_selection(
        selected_candidate="Bob",
        alternative_candidates=["Alice"],
        gold_answers=["Alice"],
        final_answer="Bob",
        final_em=0.0,
        terminal_sink=True,
        legal_selection=True,
    )

    assert factual.eligible and factual.decisive and factual.factual_wins
    assert counterfactual.eligible and counterfactual.decisive
    assert counterfactual.alternative_wins


@pytest.mark.parametrize(
    "terminal_sink,legal_selection,final_answer,final_em",
    [
        (False, True, "Alice", 1.0),
        (True, False, "Alice", 1.0),
        (True, True, "Bob", 1.0),
        (True, True, "Alice", 0.0),
    ],
)
def test_selection_requires_legal_exact_terminal_identity(
    terminal_sink, legal_selection, final_answer, final_em
):
    signal = evaluate_terminal_typed_selection(
        selected_candidate="Alice",
        alternative_candidates=["Bob"],
        gold_answers=["Alice"],
        final_answer=final_answer,
        final_em=final_em,
        terminal_sink=terminal_sink,
        legal_selection=legal_selection,
    )
    assert signal.eligible is False
    assert signal.decisive is False


def test_both_wrong_selection_is_eligible_but_not_decisive():
    signal = evaluate_terminal_typed_selection(
        selected_candidate="Alice",
        alternative_candidates=["Bob"],
        gold_answers=["Carol"],
        final_answer="Alice",
        final_em=0.0,
        terminal_sink=True,
        legal_selection=True,
    )
    assert signal.eligible is True
    assert signal.decisive is False


def test_advantage_weight_is_reason_only_bounded_and_sign_preserving():
    advantages = torch.tensor(
        [[2.0, 0.0], [-3.0, -1.0], [0.0, 0.0], [4.0, 1.0]]
    )
    weighted, metrics = apply_typed_selection_advantage_weight(
        advantages=advantages,
        roles=np.asarray(
            ["ReasonerAgent", "ReasonerAgent", "ReasonerAgent", "PlanningAgent"],
            dtype=object,
        ),
        eligible=np.asarray([True, True, True, False]),
        decisive=np.asarray([True, True, True, False]),
        alpha=0.2,
    )

    assert torch.allclose(weighted[0], advantages[0] * 1.2)
    assert torch.allclose(weighted[1], advantages[1] * 1.2)
    assert torch.equal(weighted[2], advantages[2])
    assert torch.equal(weighted[3], advantages[3])
    assert metrics["aux/typed_selection_credit/decisive_rows"] == 3.0
    assert metrics["aux/typed_selection_credit/sign_changes"] == 0.0


def test_advantage_weight_rejects_non_reason_signal_and_excess_alpha():
    advantages = torch.ones(1, 2)
    with pytest.raises(ValueError, match="ReasonerAgent"):
        apply_typed_selection_advantage_weight(
            advantages=advantages,
            roles=np.asarray(["PlanningAgent"], dtype=object),
            eligible=np.asarray([True]),
            decisive=np.asarray([True]),
            alpha=0.2,
        )
    with pytest.raises(ValueError, match="alpha"):
        apply_typed_selection_advantage_weight(
            advantages=advantages,
            roles=np.asarray(["ReasonerAgent"], dtype=object),
            eligible=np.asarray([True]),
            decisive=np.asarray([True]),
            alpha=0.21,
        )


def test_missing_selection_metadata_gets_one_false_value_per_action_row():
    metadata = {"role": np.asarray(["PlanningAgent", "ReasonerAgent"], dtype=object)}

    ensure_typed_selection_metadata(metadata, row_count=2)

    assert metadata["typed_selection_credit_eligible"].tolist() == [False, False]
    assert metadata["typed_selection_credit_decisive"].tolist() == [False, False]


def test_existing_selection_metadata_is_not_overwritten_or_resized():
    existing = np.asarray([True], dtype=np.bool_)
    metadata = {"typed_selection_credit_eligible": existing}

    ensure_typed_selection_metadata(metadata, row_count=2)

    assert metadata["typed_selection_credit_eligible"] is existing
    assert metadata["typed_selection_credit_eligible"].shape == (1,)
