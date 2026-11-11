import numpy as np
import pytest

from verl.trainer.ppo.lexicographic_f1 import all_zero_em_f1_rewards


def test_mixed_em_uses_exact_em_and_ignores_f1():
    result = all_zero_em_f1_rewards([1, 0, 0], [0.0, 1.0, 0.9], [True] * 3, weight=0.1)
    np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])


def test_all_success_has_no_extra_f1_signal():
    result = all_zero_em_f1_rewards([1, 1], [0.2, 0.9], [True, True], weight=0.1)
    np.testing.assert_array_equal(result, [1.0, 1.0])


def test_all_zero_uses_f1_only_for_valid_workflows():
    result = all_zero_em_f1_rewards([0, 0, 0], [0.2, 0.9, 0.8], [True, True, False], weight=0.1)
    np.testing.assert_allclose(result, [0.02, 0.09, 0.0])


def test_all_zero_equal_f1_has_no_preference():
    result = all_zero_em_f1_rewards([0, 0], [0.4, 0.4], [True, True], weight=0.1)
    np.testing.assert_allclose(result, [0.0, 0.0])


def test_rejects_nonfinite_or_out_of_range_inputs():
    with pytest.raises(ValueError):
        all_zero_em_f1_rewards([0], [np.nan], [True], weight=0.1)
    with pytest.raises(ValueError):
        all_zero_em_f1_rewards([0], [1.1], [True], weight=0.1)
