import pytest

from scripts.audit_dagent_informative_sampling import (
    evaluate_selector,
    fit_dataset_signal_rates,
)


def _rows(signal_scale):
    rows = []
    for step in range(1, 41):
        hotpot = 8 + step % 9
        musique = 8 + (step * 2) % 7
        wiki = 32 - hotpot - musique
        mixed = round(signal_scale * (0.08 * wiki + 0.24 * hotpot + 0.04 * musique))
        rows.append(
            {
                "step": step,
                "dataset_counts": [wiki, hotpot, musique],
                "questions": 32,
                "workflows": 128,
                "mixed_count": mixed,
                "estimated_tokens": 100_000 + step * 10,
            }
        )
    return rows


def test_frozen_dataset_mix_predictor_recovers_signal_ordering():
    coefficients = fit_dataset_signal_rates(_rows(1.0))
    assert coefficients[1] > coefficients[0] > coefficients[2]


def test_selector_uses_only_pre_rollout_features_and_keeps_budget_matched():
    report = evaluate_selector(
        _rows(1.0), _rows(1.1), bootstrap_samples=1000, seed=7
    )
    assert report["post_rollout_features_used"] is False
    assert report["workflows_per_step_values"] == [128]
    assert report["selected_to_overall_mean_token_ratio"] == pytest.approx(1.0, abs=0.01)
    assert report["selected_mixed_signal_rate"] > report["overall_mixed_signal_rate"]
    assert report["oracle_is_non_deployable"] is True


def test_report_never_skips_directly_from_offline_audit_to_short_training():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_dagent_informative_sampling.py"
    ).read_text(encoding="utf-8")
    assert '"advance_to_one_batch_sampling_check"' in source
    assert '"advance_to_matched_short_sampling_run"' not in source
    assert '"one_batch_gradient_check": False' in source
    assert '"short_training": False' in source
