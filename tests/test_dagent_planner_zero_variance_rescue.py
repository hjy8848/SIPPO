from scripts.audit_dagent_planner_zero_variance_rescue import summarize


def arm(key, logged, estimate, validation, identity=True):
    return {
        "plan_key": key,
        "multiplicity": 2,
        "logged_ems": [logged, logged],
        "estimate_q": estimate,
        "validation_q": validation,
        "samples": [{"identity_match": identity}],
    }


def report(results, expected_groups):
    return summarize(
        results,
        expected_groups=expected_groups,
        minimum_eligible_groups=8,
        minimum_informative_groups=4,
        minimum_gain=0.05,
        minimum_sign_accuracy=0.60,
    )


def test_rescue_passes_when_conditional_value_recovers_zero_variance_groups():
    results = []
    for index in range(12):
        results.append(
            {
                "group_id": f"run:2wiki_{index}",
                "arms": [
                    arm("good", 0, 1.0, 1.0),
                    arm("bad", 0, 0.0, 0.0),
                ],
            }
        )
    result = report(results, expected_groups=12)
    assert result["selector"]["rescue_gain"] == 0.5
    assert result["passed"] is True


def test_mixed_groups_are_excluded_from_rescue():
    results = [
        {
            "group_id": "run:hotpot_1",
            "arms": [
                {**arm("good", 1, 1.0, 1.0), "logged_ems": [1]},
                {**arm("bad", 0, 0.0, 0.0), "logged_ems": [0]},
            ],
        }
    ]
    result = report(results, expected_groups=1)
    assert result["eligible_zero_variance_diverse_groups"] == 0
    assert result["hard_gates"]["enough_zero_variance_diverse_groups"] is False


def test_identity_mismatch_is_a_hard_failure():
    results = []
    for index in range(12):
        results.append(
            {
                "group_id": f"run:musique_{index}",
                "arms": [
                    arm("good", 0, 1.0, 1.0, identity=index != 0),
                    arm("bad", 0, 0.0, 0.0),
                ],
            }
        )
    result = report(results, expected_groups=12)
    assert result["hard_gates"]["identity_controls_exact"] is False
    assert result["passed"] is False
