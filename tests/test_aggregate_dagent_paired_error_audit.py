from __future__ import annotations

import pytest

from scripts.aggregate_dagent_paired_error_audit import aggregate, validate


def test_validate_and_aggregate_weighted_errors() -> None:
    cases = [
        {
            "case_id": "Q001",
            "dataset": "hotpotqa",
            "stratum": "psc_only",
            "selection_weight": 3.0,
            "pure": {"em": 0.0},
            "psc": {"em": 1.0},
        },
        {
            "case_id": "Q002",
            "dataset": "hotpotqa",
            "stratum": "both_wrong",
            "selection_weight": 7.0,
            "pure": {"em": 0.0},
            "psc": {"em": 0.0},
        },
    ]
    labels = {
        "Q001": {
            "pure_primary": "retrieval_miss",
            "pure_secondary": "",
            "psc_primary": "correct",
            "psc_secondary": "",
            "pair_difference": "psc_correct",
        },
        "Q002": {
            "pure_primary": "retrieval_miss",
            "pure_secondary": "",
            "psc_primary": "answer_normalization_or_alias",
            "psc_secondary": "",
            "pair_difference": "both_wrong_different_category",
        },
    }
    validate(cases, labels)
    report = aggregate(cases, labels)
    assert report["eligible_population_questions"] == 10.0
    assert report["arm_error_totals"]["pure"] == {
        "sample": 2,
        "population_estimate": 10.0,
    }
    assert report["arm_error_totals"]["psc"] == {
        "sample": 1,
        "population_estimate": 7.0,
    }
    assert report["design_weighted_stage_distribution"]["pure"] == [
        {"category": "planning_or_grounding", "count": 10.0, "percent": 100.0}
    ]


def test_validate_rejects_correct_label_for_wrong_arm() -> None:
    cases = [
        {
            "case_id": "Q001",
            "dataset": "hotpotqa",
            "stratum": "both_wrong",
            "selection_weight": 1.0,
            "pure": {"em": 0.0},
            "psc": {"em": 0.0},
        }
    ]
    labels = {
        "Q001": {
            "pure_primary": "correct",
            "pure_secondary": "",
            "psc_primary": "retrieval_miss",
            "psc_secondary": "",
            "pair_difference": "both_wrong_different_category",
        }
    }
    with pytest.raises(ValueError, match="EM/label mismatch"):
        validate(cases, labels)
