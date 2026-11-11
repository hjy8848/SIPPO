"""Pure CPU helpers for the all-zero-EM F1 tie-break contract."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def all_zero_em_f1_rewards(
    em: Sequence[float],
    f1: Sequence[float],
    valid: Sequence[bool],
    *,
    weight: float,
) -> np.ndarray:
    """Return bounded rewards without changing any non-zero-EM decision.

    F1 is used only when every workflow in the question has EM=0 and the
    workflow is valid.  Mixed and all-success groups remain exact EM groups.
    """

    em_values = np.asarray(em, dtype=np.float64)
    f1_values = np.asarray(f1, dtype=np.float64)
    valid_values = np.asarray(valid, dtype=np.bool_)
    if not (em_values.shape == f1_values.shape == valid_values.shape):
        raise ValueError("em, f1, and valid must have the same shape")
    if not np.isfinite(em_values).all() or not np.isfinite(f1_values).all():
        raise ValueError("EM and F1 values must be finite")
    if not np.isfinite(weight) or not 0.0 < weight <= 0.5:
        raise ValueError("weight must be finite and in (0, 0.5]")
    if not np.all(np.isin(em_values, [0.0, 1.0])):
        raise ValueError("EM values must be binary")
    if np.any((f1_values < 0.0) | (f1_values > 1.0)):
        raise ValueError("F1 values must be in [0, 1]")

    rewards = em_values.copy()
    valid_f1 = f1_values[valid_values]
    if (
        em_values.size
        and float(em_values.max()) == 0.0
        and valid_f1.size >= 2
        and float(np.ptp(valid_f1)) > 1e-12
    ):
        rewards = np.where(valid_values, weight * f1_values, 0.0)
    return rewards.astype(np.float32)
