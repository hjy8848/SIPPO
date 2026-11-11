import math

import pytest

from scripts.analyze_dagent_ccw_weights import analyze


def trajectory(dataset, factual_counterfactual_pairs):
    return {
        "dataset": dataset,
        "hcapo_cases": [
            {
                "case_id": f"{dataset}:{index}",
                "node_type": "reason" if index % 2 else "search",
                "factual_em": factual,
                "counterfactual_em": counterfactual,
            }
            for index, (factual, counterfactual) in enumerate(
                factual_counterfactual_pairs
            )
        ],
    }


def test_weights_preserve_direction_and_upweight_changed_actions():
    report = analyze(
        [trajectory("2wiki", [(1, 0), (0, 1), (1, 1), (0, 0)])]
    )

    assert report["passed_safety_invariants"] is True
    assert report["changed_cases"] == 2
    assert report["changed_weight_mean"] == pytest.approx(1.5)
    assert report["unchanged_weight_mean"] == pytest.approx(1.0)
    assert report["categories"] == {
        "necessary_success_action": 1,
        "harmful_failed_action": 1,
        "outcome_unchanged": 2,
    }
    assert report["diagnostic_limitations"][
        "deletion_only_binary_em_can_produce_negative_credit"
    ] is False


def test_alpha_and_bounds_are_validated():
    rows = [trajectory("hotpotqa", [(1, 0)])]
    with pytest.raises(ValueError, match="alpha"):
        analyze(rows, alpha=0)
    with pytest.raises(ValueError, match="bounds"):
        analyze(rows, minimum_weight=1.1, maximum_weight=1.5)


def test_default_alpha_is_log_1_5():
    report = analyze([trajectory("musique", [(1, 0)])])
    assert report["formula"]["alpha"] == pytest.approx(math.log(1.5))
