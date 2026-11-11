import numpy as np
import pytest

from verl.utils.dataset.multiturn_sft_dataset import _normalize_binary_loss_mask


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (1, 1),
        (False, 0),
        (True, 1),
        (0.0, 0),
        (1.0, 1),
        (np.int64(1), 1),
        (np.float64(0.0), 0),
        (np.array(1.0), 1),
    ],
)
def test_normalize_binary_loss_mask(value, expected):
    assert _normalize_binary_loss_mask(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        -1,
        2,
        0.5,
        float("nan"),
        float("inf"),
        "1",
        np.array([0, 1]),
    ],
)
def test_normalize_binary_loss_mask_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _normalize_binary_loss_mask(value)
