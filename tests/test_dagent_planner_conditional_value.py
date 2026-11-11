from scripts.audit_dagent_ccw_agent_roles import RoleTrajectory
from scripts.audit_dagent_ccw_paired_suffixes import LoggedTrajectory
from pathlib import Path

from scripts.audit_dagent_planner_conditional_value import (
    build_plan_groups,
    plan_key,
    resolve_datasets,
    rloo_advantages,
    summarize,
)


def test_all_dataset_selector_accepts_custom_trajectory_filename():
    path = Path("full_logs_custom_replication.json")
    assert resolve_datasets([path], "all") == {"full_logs_custom_replication"}


def role_row(rollout_index, task, reward):
    trajectory = LoggedTrajectory(
        trajectory_id=f"2wiki:q1:rollout:{rollout_index}",
        question_group_id="2wiki:q1",
        rollout_index=rollout_index,
        dataset="2wiki",
        question="question",
        answers=["answer"],
        logged_answer="answer" if reward else "wrong",
        logged_em=reward,
        nodes=[{"id": "1", "type": "search", "task": task}],
        rloo_advantage=0.0,
    )
    return RoleTrajectory(trajectory, None, (), (), "")


def test_plan_groups_deduplicate_semantically_identical_plans():
    rows = [
        role_row(0, "find answer", 1),
        role_row(1, "find   answer", 0),
        role_row(2, "different", 0),
        role_row(3, "other", 0),
    ]
    groups = build_plan_groups(rows)
    assert len(groups) == 1
    assert len(groups[0].arms) == 3
    assert sorted(len(arm.members) for arm in groups[0].arms) == [1, 1, 2]
    assert plan_key(rows[0]) == plan_key(rows[1])


def test_rloo_advantages_match_leave_one_out_definition():
    assert rloo_advantages([1.0, 0.0, 0.0, 0.0]) == [1.0, -1 / 3, -1 / 3, -1 / 3]


def test_pcov_report_passes_when_estimate_predicts_validation_better_than_logged():
    results = []
    for index in range(20):
        results.append(
            {
                "group_id": f"g{index}",
                "unique_plan_count": 2,
                "arms": [
                    {
                        "plan_key": "good",
                        "multiplicity": 2,
                        "logged_ems": [0, 0],
                        "estimate_q": 1.0,
                        "validation_q": 1.0,
                        "samples": [{"identity_match": True}],
                    },
                    {
                        "plan_key": "bad",
                        "multiplicity": 2,
                        "logged_ems": [1, 1],
                        "estimate_q": 0.0,
                        "validation_q": 0.0,
                        "samples": [{"identity_match": True}],
                    },
                ],
            }
        )
    report = summarize(
        results,
        expected_groups=20,
        minimum_diverse_groups=16,
        minimum_gain=0.03,
    )
    assert report["selector"]["pcov_vs_logged"] == 1.0
    assert report["estimator"]["pcov_mse_to_validation_q"] == 0.0
    assert report["passed"] is True


def test_identity_mismatch_is_a_hard_failure():
    results = [
        {
            "group_id": f"g{index}",
            "unique_plan_count": 2,
            "arms": [
                {
                    "plan_key": "good",
                    "multiplicity": 2,
                    "logged_ems": [0, 0],
                    "estimate_q": 1.0,
                    "validation_q": 1.0,
                    "samples": [{"identity_match": index != 0}],
                },
                {
                    "plan_key": "bad",
                    "multiplicity": 2,
                    "logged_ems": [1, 1],
                    "estimate_q": 0.0,
                    "validation_q": 0.0,
                    "samples": [{"identity_match": True}],
                },
            ],
        }
        for index in range(20)
    ]
    report = summarize(
        results,
        expected_groups=20,
        minimum_diverse_groups=16,
        minimum_gain=0.03,
    )
    assert report["hard_gates"]["identity_controls_exact"] is False
    assert report["passed"] is False


def test_missing_groups_cannot_pass_completion_gate():
    results = []
    for index in range(19):
        results.append(
            {
                "group_id": f"g{index}",
                "unique_plan_count": 2,
                "arms": [
                    {
                        "plan_key": "good",
                        "multiplicity": 2,
                        "logged_ems": [0, 0],
                        "estimate_q": 1.0,
                        "validation_q": 1.0,
                        "samples": [{"identity_match": True}],
                    },
                    {
                        "plan_key": "bad",
                        "multiplicity": 2,
                        "logged_ems": [1, 1],
                        "estimate_q": 0.0,
                        "validation_q": 0.0,
                        "samples": [{"identity_match": True}],
                    },
                ],
            }
        )
    report = summarize(
        results,
        expected_groups=20,
        minimum_diverse_groups=16,
        minimum_gain=0.03,
    )
    assert report["hard_gates"]["all_groups_completed"] is False
    assert report["passed"] is False
