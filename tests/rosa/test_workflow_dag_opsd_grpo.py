import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[2]
    / "verl"
    / "trainer"
    / "ppo"
    / "workflow_dag_opsd_grpo.py"
)
SPEC = importlib.util.spec_from_file_location("workflow_dag_opsd_grpo_impl", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
compute_advantage = MODULE.compute_workflow_dag_opsd_grpo_advantage
compute_workflow_grpo = MODULE.compute_workflow_grpo_outcome_advantage
compute_workflow_loss_weights = MODULE.compute_workflow_token_loss_weights
compute_actor_data_parallel_size = MODULE.compute_actor_data_parallel_size
build_complete_workflow_dp_partitions = MODULE.build_complete_workflow_dp_partitions
build_complete_workflow_minibatch_indices = MODULE.build_complete_workflow_minibatch_indices
build_complete_question_dp_partitions = MODULE.build_complete_question_dp_partitions
build_question_preserving_minibatch_indices = (
    MODULE.build_question_preserving_minibatch_indices
)
uses_complete_workflow_actor_batches = MODULE.uses_complete_workflow_actor_batches


def test_rloo_and_grpo_use_complete_workflow_actor_batches():
    assert uses_complete_workflow_actor_batches("workflow_rloo")
    assert uses_complete_workflow_actor_batches("workflow_grpo")
    assert not uses_complete_workflow_actor_batches("grpo")


def _inputs(outcomes=(1.0, 0.0), response_length=3):
    row_count = len(outcomes)
    return {
        "token_level_rewards": torch.zeros((row_count, response_length)),
        "response_mask": torch.ones((row_count, response_length), dtype=torch.long),
        "index": np.array(["q"] * row_count, dtype=object),
        "trajectory_ids": np.array([f"w-{idx}" for idx in range(row_count)], dtype=object),
        "outcome_rewards": np.asarray(outcomes, dtype=np.float32),
        "outcome_exact": np.asarray(outcomes, dtype=np.float32),
        "student_log_probs": torch.zeros((row_count, response_length)),
        "teacher_log_probs": torch.zeros((row_count, response_length)),
        "opsd_token_mask": torch.zeros((row_count, response_length)),
        "roles": np.array(["PlanningAgent"] * row_count, dtype=object),
        "workflow_steps": np.arange(row_count, dtype=np.int64),
        "peer_available": np.zeros(row_count, dtype=np.bool_),
        "config": {
            "norm_adv_by_std_in_grpo": False,
            "dag_opsd": {
                "mix_strength": 1.0,
                "alpha_clip": 0.2,
                "gap_temperature": 1.0,
                "role_normalize": False,
            },
        },
    }


def test_no_peer_or_mask_is_exactly_workflow_grpo():
    kwargs = _inputs()
    advantages, returns = compute_advantage(**kwargs)

    expected = torch.tensor([[0.5] * 3, [-0.5] * 3])
    assert torch.allclose(advantages, expected)
    assert torch.equal(advantages, returns)

    baseline, _ = compute_workflow_grpo(
        token_level_rewards=kwargs["token_level_rewards"],
        response_mask=kwargs["response_mask"],
        index=kwargs["index"],
        trajectory_ids=kwargs["trajectory_ids"],
        outcome_rewards=kwargs["outcome_rewards"],
        config=kwargs["config"],
    )
    assert torch.equal(advantages, baseline)


def test_teacher_endorsement_softens_failed_action_penalty():
    kwargs = _inputs()
    kwargs["peer_available"][1] = True
    kwargs["opsd_token_mask"][1, :2] = 1
    kwargs["teacher_log_probs"][1, 0] = 2.0
    kwargs["teacher_log_probs"][1, 1] = -2.0

    advantages, _ = compute_advantage(**kwargs)

    assert -0.5 < advantages[1, 0] < 0.0
    assert advantages[1, 1] < -0.5
    assert advantages[1, 2] == pytest.approx(-0.5)


def test_shaping_never_flips_sign_and_stays_in_trust_region():
    kwargs = _inputs()
    kwargs["peer_available"][:] = True
    kwargs["opsd_token_mask"][:] = 1
    kwargs["teacher_log_probs"][0] = torch.tensor([-100.0, 100.0, -100.0])
    kwargs["teacher_log_probs"][1] = torch.tensor([100.0, -100.0, 100.0])

    advantages, _ = compute_advantage(**kwargs)
    multiplier = advantages / torch.tensor([[0.5] * 3, [-0.5] * 3])

    assert torch.all(advantages[0] > 0)
    assert torch.all(advantages[1] < 0)
    assert torch.all(multiplier >= 0.8)
    assert torch.all(multiplier <= 1.2)


def test_all_equal_outcomes_still_have_zero_advantage():
    kwargs = _inputs(outcomes=(1.0, 1.0))
    kwargs["peer_available"][:] = True
    kwargs["opsd_token_mask"][:] = 1
    kwargs["teacher_log_probs"][:] = 10.0

    advantages, _ = compute_advantage(**kwargs)

    assert torch.count_nonzero(advantages) == 0


def test_padding_and_unselected_tokens_are_untouched():
    kwargs = _inputs()
    kwargs["response_mask"][1, 2] = 0
    kwargs["peer_available"][1] = True
    kwargs["opsd_token_mask"][1, 0] = 1
    kwargs["teacher_log_probs"][1] = 5.0

    advantages, _ = compute_advantage(**kwargs)

    assert advantages[1, 1] == pytest.approx(-0.5)
    assert advantages[1, 2] == 0.0


def test_inconsistent_workflow_outcomes_are_rejected():
    kwargs = _inputs(outcomes=(1.0, 0.5, 0.0))
    kwargs["trajectory_ids"] = np.array(["same", "same", "other"], dtype=object)

    with pytest.raises(ValueError, match="inconsistent outcome rewards"):
        compute_advantage(**kwargs)


def test_role_normalization_remains_sign_safe_and_bounded():
    kwargs = _inputs(outcomes=(1.0, 1.0, 0.0, 0.0), response_length=4)
    kwargs["trajectory_ids"] = np.array(["good", "good", "bad", "bad"], dtype=object)
    kwargs["roles"] = np.array(
        ["PlanningAgent", "ReasonerAgent", "PlanningAgent", "ReasonerAgent"],
        dtype=object,
    )
    kwargs["peer_available"][2:] = True
    kwargs["opsd_token_mask"][2, :] = 1
    kwargs["opsd_token_mask"][3, 0] = 1
    kwargs["teacher_log_probs"][2:] = -100.0
    kwargs["config"]["dag_opsd"]["role_normalize"] = True
    kwargs["config"]["dag_opsd"]["max_role_scale"] = 4.0

    advantages, _ = compute_advantage(**kwargs)

    assert torch.all(advantages[2:] < 0)
    assert torch.all(advantages[2:] >= -0.6)
    assert torch.all(advantages[2:] <= -0.4)


def test_workflow_loss_weights_give_each_trajectory_equal_total_mass():
    mask = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.long
    )
    ids = np.array(["long", "long", "short"], dtype=object)

    weights, metrics = compute_workflow_loss_weights(
        mask, ids, np.array([0, 1, 0]), data_parallel_size=2
    )

    assert weights[:2].sum().item() == pytest.approx(1.0)
    assert weights[2].sum().item() == pytest.approx(1.0)
    assert weights.sum().item() == pytest.approx(2.0)
    assert metrics["actor/workflow_loss_workflow_count"] == 2
    assert metrics["actor/workflow_loss_data_parallel_size"] == 2
    assert metrics["actor/workflow_loss_mass_per_workflow"] == pytest.approx(1.0)


def test_two_rank_fsdp_average_matches_global_workflow_mean_when_split():
    mask = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.long
    )
    ids = np.array(["cross-rank", "rank-zero", "cross-rank", "rank-one"], dtype=object)
    token_losses = torch.tensor(
        [[2.0, 4.0, 0.0], [10.0, 0.0, 0.0], [6.0, 8.0, 10.0], [12.0, 14.0, 0.0]]
    )

    weights, _ = compute_workflow_loss_weights(
        mask, ids, np.array([0, 0, 1, 0]), data_parallel_size=2
    )
    rank_zero_loss = (token_losses[:2] * weights[:2]).sum()
    rank_one_loss = (token_losses[2:] * weights[2:]).sum()
    fsdp_averaged_loss = (rank_zero_loss + rank_one_loss) / 2

    expected = torch.stack(
        [
            token_losses[[0, 2]][mask[[0, 2]].bool()].mean(),
            token_losses[1][mask[1].bool()].mean(),
            token_losses[3][mask[3].bool()].mean(),
        ]
    ).mean()
    assert fsdp_averaged_loss.item() == pytest.approx(expected.item())


def test_actor_data_parallel_size_accounts_for_sequence_parallelism():
    assert compute_actor_data_parallel_size(2, 1) == 2
    assert compute_actor_data_parallel_size(8, 2) == 4
    with pytest.raises(ValueError, match="divisible"):
        compute_actor_data_parallel_size(3, 2)


def test_dp_partitions_keep_workflows_whole_and_pad_to_equal_rows():
    ids = np.array(
        ["w0", "w0", "w0", "w1", "w2", "w2", "w3", "w3", "w3", "w3"],
        dtype=object,
    )
    partitions = build_complete_workflow_dp_partitions(ids, data_parallel_size=2)

    assert [partition.size for partition in partitions] == [6, 6]
    first_ids = set(ids[partitions[0]].tolist())
    second_ids = set(ids[partitions[1]].tolist())
    assert first_ids == {"w0", "w1"}
    assert second_ids == {"w2", "w3"}
    assert first_ids.isdisjoint(second_ids)
    assert np.count_nonzero(partitions[0] == 0) == 3


def test_workflow_minibatches_never_split_actions_of_one_workflow():
    ids = np.array(
        ["w0", "w0", "w1", "w2", "w2", "w2", "w3"], dtype=object
    )
    minibatches = build_complete_workflow_minibatch_indices(
        ids, workflows_per_minibatch=2
    )

    assert len(minibatches) == 2
    assert set(ids[minibatches[0]].tolist()) == {"w0", "w1"}
    assert set(ids[minibatches[1]].tolist()) == {"w2", "w3"}


def test_question_minibatches_keep_all_siblings_and_actions_together():
    questions = np.asarray(
        [question for question, rows in (("q0", 7), ("q1", 5), ("q2", 6), ("q3", 4)) for _ in range(rows)],
        dtype=object,
    )
    trajectories = np.asarray(
        [f"{question}-w{row // 2}" for question, rows in (("q0", 7), ("q1", 5), ("q2", 6), ("q3", 4)) for row in range(rows)],
        dtype=object,
    )

    minibatches = build_question_preserving_minibatch_indices(
        questions, trajectories, target_action_rows=6
    )

    assert len(minibatches) == 4
    owners = {}
    for minibatch_index, indices in enumerate(minibatches):
        for question in set(questions[indices].tolist()):
            assert question not in owners
            owners[question] = minibatch_index
            assert np.count_nonzero(questions[indices] == question) == np.count_nonzero(
                questions == question
            )
    assert set(owners) == {"q0", "q1", "q2", "q3"}


def test_question_dp_partitions_keep_questions_on_one_rank_and_mark_padding():
    questions = np.asarray(
        ["q0"] * 7 + ["q1"] * 5 + ["q2"] * 4 + ["q3"] * 3,
        dtype=object,
    )
    trajectories = np.asarray(
        [f"{question}-w{index // 2}" for index, question in enumerate(questions)],
        dtype=object,
    )

    partitions, real_rows = build_complete_question_dp_partitions(
        questions, trajectories, data_parallel_size=2
    )

    assert len(partitions[0]) == len(partitions[1])
    rank_questions = [
        set(questions[partition[:count]].tolist())
        for partition, count in zip(partitions, real_rows, strict=True)
    ]
    assert rank_questions[0].isdisjoint(rank_questions[1])
    assert rank_questions[0] | rank_questions[1] == {"q0", "q1", "q2", "q3"}
    assert sum(real_rows) == len(questions)
    assert sum(len(partition) - count for partition, count in zip(partitions, real_rows, strict=True)) > 0


def test_question_minibatches_reproduce_five_to_six_action_row_updates():
    row_counts = [18 + index % 7 for index in range(32)]
    questions = np.asarray(
        [f"q{index}" for index, count in enumerate(row_counts) for _ in range(count)],
        dtype=object,
    )
    trajectories = np.asarray(
        [f"q{index}-w{row % 4}" for index, count in enumerate(row_counts) for row in range(count)],
        dtype=object,
    )

    minibatches = build_question_preserving_minibatch_indices(
        questions, trajectories, target_action_rows=128
    )

    assert len(minibatches) in {5, 6}


def test_formal_rloo_contract_is_one_optimizer_step_per_global_step():
    ids = np.asarray(
        [
            workflow_id
            for workflow in range(96)
            for workflow_id in [f"w{workflow}"] * (3 + workflow % 4)
        ],
        dtype=object,
    )
    partitions = build_complete_workflow_dp_partitions(ids, data_parallel_size=2)

    for partition in partitions:
        local_ids = ids[partition]
        assert len(set(local_ids.tolist())) == 48
        minibatches = build_complete_workflow_minibatch_indices(
            local_ids,
            workflows_per_minibatch=48,
        )
        assert len(minibatches) == 1
        assert set(local_ids[minibatches[0]].tolist()) == set(local_ids.tolist())


def test_workflow_loss_weights_do_not_double_count_padding_duplicates():
    mask = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.long)
    ids = np.array(["duplicated", "duplicated", "other"], dtype=object)
    steps = np.array([0, 0, 0], dtype=np.int64)

    weights, metrics = compute_workflow_loss_weights(
        mask, ids, steps, data_parallel_size=1
    )

    assert weights[0].sum().item() == pytest.approx(0.25)
    assert weights[1].sum().item() == pytest.approx(0.25)
    assert weights[2].sum().item() == pytest.approx(0.5)
    assert metrics["actor/workflow_loss_duplicate_row_count"] == 1
    assert metrics["actor/workflow_loss_logical_action_count"] == 2


def test_workflow_loss_weights_reject_empty_action():
    mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="empty action"):
        compute_workflow_loss_weights(
            mask,
            np.array(["valid", "empty"], dtype=object),
            np.array([0, 0], dtype=np.int64),
        )
