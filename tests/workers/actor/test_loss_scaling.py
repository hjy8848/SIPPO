import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[3]
    / "verl"
    / "workers"
    / "actor"
    / "loss_scaling.py"
)
SPEC = importlib.util.spec_from_file_location("loss_scaling_impl", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
compute_microbatch_loss_scale = MODULE.compute_microbatch_loss_scale


def test_token_mean_scaling_reconstructs_heterogeneous_minibatch_loss():
    mask = torch.zeros((4, 100), dtype=torch.long)
    mask[0] = 1
    mask[1:, 0] = 1
    token_losses = torch.zeros_like(mask, dtype=torch.float32)
    token_losses[0] = 1.0
    token_losses[1:, 0] = -1.0

    partitions = ([0], [1, 2, 3])
    accumulated = 0.0
    for rows in partitions:
        micro_mask = mask[rows]
        micro_loss = token_losses[rows][micro_mask.bool()].mean()
        scale = compute_microbatch_loss_scale(micro_mask, mask, "token-mean")
        accumulated += float(micro_loss.item()) * scale

    expected = token_losses[mask.bool()].mean().item()
    assert accumulated == pytest.approx(expected)
    assert accumulated > 0.0


@pytest.mark.parametrize(
    "mode",
    ["seq-mean-token-sum", "seq-mean-token-mean"],
)
def test_sequence_mean_modes_scale_by_active_rows(mode):
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]])
    scale = compute_microbatch_loss_scale(mask[[0]], mask, mode)
    assert scale == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    "mode",
    ["seq-mean-token-sum-norm", "normalized-weighted-sum"],
)
def test_additive_modes_do_not_rescale_microbatches(mode):
    mask = torch.tensor([[1, 1], [1, 0]])
    assert compute_microbatch_loss_scale(mask[[0]], mask, mode) == 1.0


def test_empty_minibatch_mask_is_rejected():
    mask = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="no active mass"):
        compute_microbatch_loss_scale(mask[[0]], mask, "token-mean")
