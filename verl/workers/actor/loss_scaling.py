"""Loss-preserving gradient accumulation across heterogeneous microbatches."""

from __future__ import annotations

import torch


_SEQUENCE_MEAN_MODES = {
    "seq-mean-token-sum",
    "seq-mean-token-mean",
}
_ADDITIVE_MODES = {
    "seq-mean-token-sum-norm",
    "normalized-weighted-sum",
}


def compute_microbatch_loss_scale(
    microbatch_loss_mask: torch.Tensor,
    minibatch_loss_mask: torch.Tensor,
    loss_agg_mode: str,
) -> float:
    """Return the factor that exactly reconstructs the full-minibatch loss."""

    if microbatch_loss_mask.ndim != 2 or minibatch_loss_mask.ndim != 2:
        raise ValueError("loss masks must be rank-2 tensors")

    if loss_agg_mode == "token-mean":
        microbatch_mass = microbatch_loss_mask.sum()
        minibatch_mass = minibatch_loss_mask.sum()
    elif loss_agg_mode in _SEQUENCE_MEAN_MODES:
        microbatch_mass = microbatch_loss_mask.sum(dim=-1).gt(0).sum()
        minibatch_mass = minibatch_loss_mask.sum(dim=-1).gt(0).sum()
    elif loss_agg_mode in _ADDITIVE_MODES:
        return 1.0
    else:
        raise ValueError(f"Unsupported loss aggregation mode: {loss_agg_mode}")

    denominator = float(minibatch_mass.detach().item())
    if denominator <= 0.0:
        raise ValueError("minibatch loss mask has no active mass")
    numerator = float(microbatch_mass.detach().item())
    if numerator < 0.0:
        raise ValueError("microbatch loss mask has negative mass")
    return numerator / denominator
