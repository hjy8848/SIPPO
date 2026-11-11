from scripts.audit_dagent_ccw_paired_suffixes import (
    LoggedTrajectory,
    attach_workflow_rloo_advantages,
    replication_metrics,
    summarize,
)


def make_case(trajectory, dataset, role, estimate, validation, credit):
    return {
        "trajectory_id": trajectory,
        "dataset": dataset,
        "node_type": role,
        "estimate_delta": estimate,
        "validation_delta": validation,
        "signed_credit": credit,
        "weight": 0.8 if credit < 0 else 1.2 if credit > 0 else 1.0,
    }


def test_replication_metrics_reward_reproducible_direction():
    cases = [
        make_case(f"t{i}", "2wiki", "search", 0.5, 0.5, 0.5)
        for i in range(20)
    ]
    metrics = replication_metrics(cases, seed=1)
    assert metrics["validation_direction_accuracy_ties_half"] == 1.0
    assert metrics["validation_selection_gain_over_random"] == 0.25
    assert metrics["direction_cluster_bootstrap_95ci"] == [1.0, 1.0]


def test_summary_checks_bounded_sign_preserving_weights():
    cases = []
    for index in range(40):
        credit = -0.5 if index == 0 else 0.5
        cases.append(
            make_case(
                f"t{index}",
                "2wiki" if index % 2 else "hotpotqa",
                "reason" if index % 3 == 0 else "search",
                0.5,
                0.5,
                credit,
            )
        )
    results = [
        {"trajectory_id": case["trajectory_id"], "dataset": case["dataset"], "cases": [case]}
        for case in cases
    ]
    report = summarize(results, seed=2)
    assert report["safety_invariants"] == {
        "all_weights_finite": True,
        "all_weights_in_0_5_1_5": True,
        "em_direction_preserved": True,
    }
    assert report["credit_distribution"]["negative"] == 1
    assert report["credit_distribution"]["positive"] == 39


def test_attach_workflow_rloo_advantages_matches_leave_one_out():
    trajectories = [
        LoggedTrajectory(
            trajectory_id=f"2wiki:q1:rollout:{index}",
            question_group_id="2wiki:q1",
            rollout_index=index,
            dataset="2wiki",
            question="q1",
            answers=["answer"],
            logged_answer="answer" if reward else "wrong",
            logged_em=reward,
            nodes=[],
        )
        for index, reward in enumerate([1, 0, 0, 0])
    ]
    annotated = attach_workflow_rloo_advantages(trajectories, 4)
    by_rollout = {row.rollout_index: row.rloo_advantage for row in annotated}
    assert by_rollout == {0: 1.0, 1: -1 / 3, 2: -1 / 3, 3: -1 / 3}


def test_attach_workflow_rloo_advantages_rejects_incomplete_group():
    trajectory = LoggedTrajectory(
        trajectory_id="2wiki:q1:rollout:0",
        question_group_id="2wiki:q1",
        rollout_index=0,
        dataset="2wiki",
        question="q1",
        answers=["answer"],
        logged_answer="answer",
        logged_em=1,
        nodes=[],
    )
    try:
        attach_workflow_rloo_advantages([trajectory], 4)
    except ValueError as exc:
        assert "expected 4" in str(exc)
    else:
        raise AssertionError("incomplete RLOO groups must be rejected")
