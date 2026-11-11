from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_dagent_reasoner_pcgrad_dpo.py"
SPEC = importlib.util.spec_from_file_location("reasoner_pcgrad_dpo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def pair(group: str, operation: str, variant: str) -> dict:
    return {
        "group_id": group,
        "operation": operation,
        "variant": variant,
        "split": "train",
        "agent_name": "ReasonerAgent",
    }


def test_group_batcher_keeps_complete_twins():
    rows = []
    for index in range(3):
        for variant in MODULE.VARIANTS:
            rows.append(pair(f"g{index}", "earlier", variant))
    twins = MODULE.index_twins(rows)
    batcher = MODULE.GroupBatcher(twins, "earlier", groups_per_batch=2, seed=4)
    selected = batcher.next_rows()
    assert len(selected) == 4
    assert {row["variant"] for row in selected} == set(MODULE.VARIANTS)
    assert len({row["group_id"] for row in selected}) == 2


def test_mean_gradient_is_plain_average():
    gradients = {
        "closer_to_anchor": [torch.tensor([1.0, 0.0])],
        "earlier": [torch.tensor([0.0, 1.0])],
        "later": [torch.tensor([1.0, 1.0])],
    }
    combined, audit = MODULE.combine_operation_gradients(gradients, "mean", seed=1)
    assert torch.allclose(combined[0], torch.tensor([2 / 3, 2 / 3]))
    assert audit["projection_count"] == 0


def test_pcgrad_projects_a_conflicting_component():
    gradients = {
        "closer_to_anchor": [torch.tensor([1.0, 0.0])],
        "earlier": [torch.tensor([-1.0, 1.0])],
        "later": [torch.tensor([0.0, 1.0])],
    }
    combined, audit = MODULE.combine_operation_gradients(gradients, "pcgrad", seed=1)
    assert audit["raw_cosines"]["closer_to_anchor__earlier"] < 0
    assert audit["projection_count"] >= 1
    assert torch.isfinite(combined[0]).all()
    assert combined[0].norm() > 0


def test_chosen_response_nll_scores_even_rows_only():
    logp_sums = torch.tensor([-4.0, -100.0, -9.0, -200.0])
    token_counts = torch.tensor([2, 5, 3, 7])
    loss = MODULE.chosen_response_nll(logp_sums, token_counts)
    assert torch.allclose(loss, torch.tensor(2.5))


def test_chosen_response_nll_rejects_incomplete_pairs():
    with torch.no_grad():
        try:
            MODULE.chosen_response_nll(torch.tensor([-1.0]), torch.tensor([1]))
        except ValueError as error:
            assert "complete pairs" in str(error)
        else:
            raise AssertionError("expected incomplete pair validation")
