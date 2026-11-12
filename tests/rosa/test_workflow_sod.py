import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[2] / "verl" / "trainer" / "ppo" / "workflow_sod.py"
)
SPEC = importlib.util.spec_from_file_location("workflow_sod_impl", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
apply_workflow_sod_advantage = MODULE.apply_workflow_sod_advantage
build_opening_tag_excluded_mask = MODULE.build_opening_tag_excluded_mask
build_prefix_aligned_references = MODULE.build_prefix_aligned_references
build_reference_action_records = MODULE.build_reference_action_records
choose_fixed_aligned_sibling = MODULE.choose_fixed_aligned_sibling
choose_distinct_successful_sibling = MODULE.choose_distinct_successful_sibling
count_target_future_actions_in_reference_prefix = (
    MODULE.count_target_future_actions_in_reference_prefix
)
render_missing_action_reference = MODULE.render_missing_action_reference
render_causally_aligned_reference = MODULE.render_causally_aligned_reference
render_full_privileged_reference = MODULE.render_full_privileged_reference
required_invalid_workflow_reward_cap = MODULE.required_invalid_workflow_reward_cap
should_score_workflow_sod_target = MODULE.should_score_workflow_sod_target
load_privileged_reference_bank = MODULE.load_privileged_reference_bank
workflow_action_match_key = MODULE.workflow_action_match_key


class _CharTokenizer:
    def decode(self, ids, **_kwargs):
        return "".join(chr(token_id) for token_id in ids)

    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _char_response(text):
    return torch.tensor([ord(char) for char in text]), torch.ones(
        len(text), dtype=torch.long
    )


@pytest.mark.parametrize(
    ("role", "text", "expected_removed"),
    [
        ("PlanningAgent", "<plan>\n<search id=\"1\">x</search></plan>", 6),
        ("ReasonerAgent", "<conclusion>Ada</conclusion>", 12),
        ("AnswerSummarizationAgent", "<answer>Ada</answer>", 8),
        ("AnswerGenerationAgent", "<rewrite>Ada spouse</rewrite>", 9),
        ("QueryRewriteAgent", "Who was Ada Lovelace's spouse?", 0),
    ],
)
def test_opening_tag_mask_removes_only_role_protocol_prefix(
    role, text, expected_removed
):
    response_ids, response_mask = _char_response(text)

    selected, fallback, removed = build_opening_tag_excluded_mask(
        _CharTokenizer(), response_ids, response_mask, role
    )

    assert not fallback
    assert removed == expected_removed
    assert selected.sum().item() == len(text) - expected_removed


def test_opening_tag_mask_keeps_malformed_response_as_safe_fallback():
    response_ids, response_mask = _char_response("Ada Lovelace")

    selected, fallback, removed = build_opening_tag_excluded_mask(
        _CharTokenizer(), response_ids, response_mask, "AnswerSummarizationAgent"
    )

    assert fallback
    assert removed == 0
    torch.testing.assert_close(selected, response_mask.bool())


def test_full_privileged_reference_contains_every_accepted_action():
    records = [
        {"step": 0, "role": "PlanningAgent", "node_id": "planning", "response": "plan"},
        {"step": 1, "role": "QueryRewriteAgent", "node_id": "n1", "response": "query"},
        {"step": 2, "role": "ReasonerAgent", "node_id": "n1", "response": "reason"},
    ]

    rendered = render_full_privileged_reference(records)

    assert rendered.startswith("<privileged_successful_workflow>")
    assert rendered.endswith("</privileged_successful_workflow>")
    assert rendered.count("<agent ") == len(records)
    assert all(record["response"] in rendered for record in records)


def _constant_gap_rows(gaps):
    old = torch.zeros((len(gaps), 3), dtype=torch.float32)
    teacher = torch.tensor([[gap, gap, 99.0] for gap in gaps], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0]] * len(gaps), dtype=torch.long)
    return old, teacher, mask


def test_workflow_sod_groups_agent_rows_and_collapses_padding_duplicates():
    # A's Search role recovers at its second action. Row 3 is a distributed
    # padding duplicate of A/Search/step1 and must not become an extra action.
    old, teacher, mask = _constant_gap_rows([1.0, 2.0, 0.5, 2.0, 0.25])
    advantages = torch.full_like(old, 10.0) * mask
    output = apply_workflow_sod_advantage(
        advantages=advantages,
        old_log_probs=old,
        teacher_log_probs=teacher,
        response_mask=mask,
        trajectory_ids=np.array(["A", "A", "A", "A", "B"], dtype=object),
        workflow_steps=np.array([0, 1, 2, 1, 0]),
        roles=np.array(["Plan", "Search", "Search", "Search", "Plan"], dtype=object),
        workflow_tool_steps=np.array([3, 3, 3, 3, 1]),
        coefficient=1.0,
        epsilon=1e-6,
        delta=0.2,
    )

    expected_weights = torch.tensor(
        [[1.0, 1.0, 0.0], [0.5, 0.5, 0.0], [1.2, 1.2, 0.0],
         [0.5, 0.5, 0.0], [1.0, 1.0, 0.0]]
    )
    torch.testing.assert_close(output.weights, expected_weights)
    torch.testing.assert_close(
        output.advantages[:, 0],
        torch.tensor([11.0, 11.0, 10.6, 11.0, 10.25]),
    )
    assert output.metrics["aux/workflow_sod/trajectory_count"] == 2
    assert output.metrics["aux/workflow_sod/action_count"] == 4
    assert output.metrics["aux/workflow_sod/duplicate_padding_rows"] == 1
    assert output.metrics["aux/workflow_sod/multi_step_trajectory_count"] == 1
    assert output.metrics["aux/workflow_sod/multi_action_same_role_stream_count"] == 1
    assert (
        output.metrics[
            "aux/workflow_sod/recovery_after_increase_trajectory_fraction"
        ]
        == 0.5
    )


def test_workflow_sod_keeps_negative_teacher_signal_and_masked_tokens_zero():
    old = torch.tensor([[0.0, 0.0, 7.0]])
    teacher = torch.tensor([[-2.0, 1.0, -9.0]])
    mask = torch.tensor([[1, 1, 0]])
    base = torch.tensor([[3.0, 3.0, 0.0]])
    output = apply_workflow_sod_advantage(
        advantages=base,
        old_log_probs=old,
        teacher_log_probs=teacher,
        response_mask=mask,
        trajectory_ids=np.array(["failed-workflow"], dtype=object),
        workflow_steps=np.array([0]),
        roles=np.array(["AnswerGenerationAgent"], dtype=object),
        coefficient=0.5,
    )

    # No success/outcome filter exists: a failed on-policy action is distilled
    # wherever the Teacher assigns a different sampled-token log probability.
    torch.testing.assert_close(output.advantages, torch.tensor([[2.0, 3.5, 0.0]]))
    assert output.metrics["aux/workflow_sod/local_advantage_positive_fraction"] == 0.5


def test_pure_workflow_sod_replaces_the_rl_advantage():
    old = torch.tensor([[0.0, 0.0, 0.0]])
    teacher = torch.tensor([[2.0, -1.0, 99.0]])
    mask = torch.tensor([[1, 1, 0]])
    output = apply_workflow_sod_advantage(
        advantages=torch.tensor([[50.0, 50.0, 0.0]]),
        old_log_probs=old,
        teacher_log_probs=teacher,
        response_mask=mask,
        trajectory_ids=np.array(["failed-workflow"], dtype=object),
        workflow_steps=np.array([0]),
        roles=np.array(["PlanningAgent"], dtype=object),
        coefficient=0.5,
        sod_only=True,
    )

    torch.testing.assert_close(output.advantages, torch.tensor([[1.0, -0.5, 0.0]]))
    assert output.metrics["aux/workflow_sod/sod_only"] == 1.0
    assert output.metrics["aux/workflow_sod/original_rl_advantage_rms"] > 0.0


def test_pure_sod_sign_reduces_exact_on_policy_reverse_kl():
    student = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    teacher = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    output = apply_workflow_sod_advantage(
        advantages=torch.zeros((3, 1), dtype=torch.float64),
        old_log_probs=student.log().unsqueeze(-1),
        teacher_log_probs=teacher.log().unsqueeze(-1),
        response_mask=torch.ones((3, 1), dtype=torch.long),
        trajectory_ids=np.array(["sample-0", "sample-1", "sample-2"]),
        workflow_steps=np.zeros(3, dtype=np.int64),
        roles=np.array(["PlanningAgent"] * 3),
        sod_only=True,
    )

    logits = student.log().clone().requires_grad_()
    current = logits.softmax(dim=0)
    ratio = current / student
    exact_on_policy_loss = -(
        student * output.advantages.squeeze(-1) * ratio
    ).sum()
    gradient = torch.autograd.grad(exact_on_policy_loss, logits)[0]
    updated = (logits - 0.1 * gradient).softmax(dim=0)
    old_kl = (student * (student.log() - teacher.log())).sum()
    new_kl = (updated * (updated.log() - teacher.log())).sum()
    assert new_kl < old_kl


def test_pure_sod_empty_signal_is_an_exact_noop():
    old = torch.tensor([[0.0, -1.0, 0.0]])
    output = apply_workflow_sod_advantage(
        advantages=torch.tensor([[4.0, -3.0, 0.0]]),
        old_log_probs=old,
        teacher_log_probs=old.clone(),
        response_mask=torch.tensor([[1, 1, 0]]),
        distillation_mask=torch.zeros((1, 3), dtype=torch.bool),
        trajectory_ids=np.array(["no-successful-sibling"], dtype=object),
        workflow_steps=np.array([0]),
        roles=np.array(["PlanningAgent"], dtype=object),
        sod_only=True,
    )

    torch.testing.assert_close(output.advantages, torch.zeros_like(old))
    assert output.metrics["aux/workflow_sod/response_token_count"] == 0.0


def test_pure_sod_requires_invalid_workflows_to_receive_zero_reward():
    assert required_invalid_workflow_reward_cap(sod_only=True) == 0.0
    assert required_invalid_workflow_reward_cap(sod_only=False) == 1.0


@pytest.mark.parametrize(
    ("target_success", "workflow_valid", "sod_only", "expected"),
    [
        (True, True, True, False),
        (True, True, False, False),
        (False, True, True, True),
        (False, True, False, True),
        (False, False, True, True),
        (False, False, False, False),
    ],
)
def test_pure_sod_keeps_invalid_failures_trainable(
    target_success, workflow_valid, sod_only, expected
):
    assert (
        should_score_workflow_sod_target(
            target_success=target_success,
            workflow_valid=workflow_valid,
            sod_only=sod_only,
        )
        is expected
    )


def test_prefix_aligned_reference_never_contains_future_actions():
    records = [
        {"step": 0, "role": "PlanningAgent", "node_id": "", "alignment_key": "plan", "response": "PLAN"},
        {"step": 1, "role": "QueryRewriteAgent", "node_id": "#1#", "alignment_key": "search", "response": "SEARCH"},
        {"step": 2, "role": "ReasonerAgent", "node_id": "#2#", "alignment_key": "reason", "response": "FUTURE_REASON"},
    ]

    references = build_prefix_aligned_references(records)
    assert "PLAN" in references["plan"]
    assert "SEARCH" not in references["plan"]
    assert "FUTURE_REASON" not in references["plan"]
    assert "SEARCH" in references["search"]
    assert "FUTURE_REASON" not in references["search"]


def test_future_reference_counter_detects_reordered_target_action():
    target_records = [
        {"step": 0, "alignment_key": "plan"},
        {"step": 1, "alignment_key": "search-a"},
        {"step": 2, "alignment_key": "search-b"},
    ]
    reference_records = [
        {"step": 0, "alignment_key": "plan"},
        {"step": 1, "alignment_key": "search-b"},
        {"step": 2, "alignment_key": "search-a"},
    ]

    assert count_target_future_actions_in_reference_prefix(
        reference_records,
        target_records,
        target_step=1,
        alignment_key="search-a",
    ) == 1
    assert count_target_future_actions_in_reference_prefix(
        reference_records,
        target_records,
        target_step=2,
        alignment_key="search-b",
    ) == 0


def test_fixed_bank_reference_filters_actions_later_in_student_order():
    target_records = [
        {"step": 0, "alignment_key": "plan"},
        {"step": 1, "alignment_key": "answer-b"},
        {"step": 2, "alignment_key": "answer-a"},
    ]
    reference_records = [
        {"step": 0, "alignment_key": "plan", "response": "PLAN"},
        {"step": 1, "alignment_key": "answer-a", "response": "FUTURE"},
        {"step": 2, "alignment_key": "answer-b", "response": "CURRENT"},
    ]
    rendered = render_causally_aligned_reference(
        reference_records,
        target_records,
        target_step=1,
        alignment_key="answer-b",
    )
    assert "PLAN" in rendered
    assert "CURRENT" in rendered
    assert "FUTURE" not in rendered


def test_missing_action_reference_marks_absence_without_future_leakage():
    records = [
        {"step": 0, "role": "PlanningAgent", "node_id": "", "alignment_key": "plan", "response": "PLAN"},
        {"step": 1, "role": "QueryRewriteAgent", "node_id": "#1#", "alignment_key": "search", "response": "SEARCH"},
        {"step": 2, "role": "AnswerGenerationAgent", "node_id": "#1#", "alignment_key": "answer", "response": "FUTURE"},
    ]
    target_records = [
        {"step": 0, "role": "PlanningAgent", "node_id": "", "alignment_key": "plan", "response": "TARGET_PLAN"},
        {"step": 1, "role": "QueryRewriteAgent", "node_id": "#1#", "alignment_key": "search", "response": "TARGET_SEARCH"},
        {"step": 2, "role": "QueryRewriteAgent", "node_id": "#2#", "alignment_key": "search-2", "response": "TARGET_SEARCH_2"},
        {"step": 3, "role": "ReasonerAgent", "node_id": "#3#", "alignment_key": "missing-reason", "response": "TARGET_REASON"},
    ]

    reference = render_missing_action_reference(
        records, 3, "missing-reason", target_records
    )
    assert "PLAN" in reference
    assert "SEARCH" in reference
    assert "FUTURE" not in reference
    assert '<aligned_action_absent key="missing-reason">' in reference


def test_fixed_sibling_prefers_alignment_and_does_not_switch_per_action():
    references = {
        "success-a": {"plan": "a", "search": "a"},
        "success-b": {"plan": "b", "search": "b", "reason": "b"},
    }
    selected = choose_fixed_aligned_sibling(
        ["success-a", "success-b"],
        "failed",
        {"plan", "search", "reason"},
        references,
    )
    assert selected == "success-b"


def test_workflow_sod_applies_only_privileged_self_teacher_rows():
    old, teacher, mask = _constant_gap_rows([1.0, 2.0])
    distillation_mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    output = apply_workflow_sod_advantage(
        advantages=torch.zeros_like(old),
        old_log_probs=old,
        teacher_log_probs=teacher,
        response_mask=mask,
        distillation_mask=distillation_mask,
        trajectory_ids=np.array(["A", "B"], dtype=object),
        workflow_steps=np.array([0, 0]),
        roles=np.array(["Plan", "Plan"], dtype=object),
    )
    torch.testing.assert_close(output.advantages[0], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(output.advantages[1], torch.zeros(3))
    assert output.metrics["aux/workflow_sod/action_count"] == 1


def test_self_teacher_reference_is_successful_and_never_the_target():
    successes = ["workflow-0", "workflow-2"]
    assert (
        choose_distinct_successful_sibling(successes, "workflow-0", rotation=0)
        == "workflow-2"
    )
    assert (
        choose_distinct_successful_sibling(successes, "failed-workflow", rotation=1)
        == "workflow-2"
    )
    assert choose_distinct_successful_sibling(["workflow-0"], "workflow-0") is None


def test_workflow_sod_rejects_conflicting_roles_for_one_logical_step():
    old, teacher, mask = _constant_gap_rows([1.0, 1.0])
    with pytest.raises(ValueError, match="inconsistent roles"):
        apply_workflow_sod_advantage(
            advantages=torch.zeros_like(old),
            old_log_probs=old,
            teacher_log_probs=teacher,
            response_mask=mask,
            trajectory_ids=np.array(["A", "A"], dtype=object),
            workflow_steps=np.array([0, 0]),
            roles=np.array(["Plan", "Search"], dtype=object),
        )


def test_workflow_sod_eq7_follows_the_complete_cross_role_action_sequence():
    old, teacher, mask = _constant_gap_rows([0.1, 10.0])
    output = apply_workflow_sod_advantage(
        advantages=torch.zeros_like(old),
        old_log_probs=old,
        teacher_log_probs=teacher,
        response_mask=mask,
        trajectory_ids=np.array(["A", "A"], dtype=object),
        workflow_steps=np.array([0, 1]),
        roles=np.array(["PlanningAgent", "ReasonerAgent"], dtype=object),
    )

    torch.testing.assert_close(
        output.weights[:, :2],
        torch.tensor([[1.0, 1.0], [0.01, 0.01]], dtype=torch.float32),
        rtol=1e-4,
        atol=1e-6,
    )
    assert output.metrics["aux/workflow_sod/same_role_stream_count"] == 2
    assert output.metrics["aux/workflow_sod/stepwise_stream_count"] == 1
    assert output.metrics["aux/workflow_sod/multi_action_same_role_stream_count"] == 0


def test_fixed_reference_bank_is_deterministic_and_prefix_aligned(tmp_path):
    bank_path = tmp_path / "accepted.jsonl"
    bank_path.write_text(
        '{"trajectory_id":"good-1","question":"  Who  won? ","sft_events":['
        '{"agent_name":"PlanningAgent","response":"PLAN","metadata":{}},'
        '{"agent_name":"AnswerGenerationAgent","response":"FIRST","metadata":{"node_id":"1"}},'
        '{"agent_name":"AnswerGenerationAgent","response":"RETRY","metadata":{"node_id":"1"}},'
        '{"agent_name":"AnswerSummarizationAgent","response":"FINAL","metadata":{}}]}'
        "\n",
        encoding="utf-8",
    )

    bank = load_privileged_reference_bank(bank_path)
    entry = bank["Who won?"]
    records = entry["action_records"]
    assert records[0]["alignment_key"] == workflow_action_match_key(
        "PlanningAgent", "planning", 0
    )
    assert records[1]["alignment_key"] == workflow_action_match_key(
        "AnswerGenerationAgent", "1", 0
    )
    assert records[2]["alignment_key"] == workflow_action_match_key(
        "AnswerGenerationAgent", "1", 1
    )
    first_answer_reference = entry["aligned_references"][
        workflow_action_match_key("AnswerGenerationAgent", "1", 0)
    ]
    assert "FIRST" in first_answer_reference
    assert "RETRY" not in first_answer_reference
    assert "FINAL" not in first_answer_reference


def test_fixed_reference_bank_rejects_duplicate_questions(tmp_path):
    bank_path = tmp_path / "duplicate.jsonl"
    event = (
        '[{"agent_name":"PlanningAgent","response":"PLAN","metadata":{}}]'
    )
    bank_path.write_text(
        '{"trajectory_id":"a","question":"Q","sft_events":' + event + "}\n"
        '{"trajectory_id":"b","question":" Q ","sft_events":' + event + "}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate question"):
        load_privileged_reference_bank(bank_path)


def test_workflow_sod_launcher_enables_only_the_new_joint_objective():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "exp" / "run_dagent_workflow_sod_grpo_elastic_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "WORKFLOW_SOD_ENABLED=true" in launcher
    assert "WORKFLOW_SOD_REQUIRE_MULTI_STEP=false" in launcher
    assert "FINAL_ANSWER_RETRY_LIMIT=0" in launcher
    assert "ADV_ESTIMATOR=workflow_grpo" in launcher
    assert "DAG_OPSD_ENABLED=false" in launcher
    assert "ONLINE_OPSD_ENABLED=false" in launcher
    assert "USE_KL_LOSS=false" in launcher
    assert "TEACHER_MODEL_PATH" not in launcher
    assert "DAGENT-Mixed-OutcomeEM" not in launcher
    assert "Teacher=current Student parameters" in launcher


def test_fixed_mixed3_formal_launcher_covers_the_complete_dataset_once():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "exp" / "run_dagent_self_sod_fixed_mixed3_formal_zju18.sh"
    ).read_text(encoding="utf-8")
    assert "rl_train_mixed_2wiki40_hotpot30_musique30_seed42.parquet" in launcher
    assert "validation_mixed_2wiki_hotpot_musique_seed42.parquet" in launcher
    assert "TRAIN_BATCH_SIZE=24" in launcher
    assert "TOTAL_TRAINING_STEPS=625" in launcher
    assert "WORKFLOW_ROLLOUT_N=4" in launcher
    assert "FINAL_ANSWER_RETRY_LIMIT=0" in launcher
    assert "RAY_TMPDIR" in launcher


def test_workflow_training_logs_per_batch_data_source_counts():
    root = Path(__file__).resolve().parents[2]
    trainer = (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
        encoding="utf-8"
    )
    assert 'record_metrics["data_source"] = normalized_data_source' in trainer
    assert 'train/data/questions/{data_source}' in trainer


def test_workflow_sod_supports_legacy_actor_and_frozen_reference_teachers():
    root = Path(__file__).resolve().parents[2]
    trainer = (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
        encoding="utf-8"
    )
    main = (root / "verl" / "trainer" / "main_ppo.py").read_text(encoding="utf-8")
    assert "def _prepare_workflow_sod_batch" in trainer
    assert "self.actor_rollout_wg.compute_log_prob(padded_probe_batch)" in trainer
    assert '"workflow_sod_teacher_log_probs"' in trainer
    assert '"online_opsd_workflow_role_texts"' in trainer
    assert '"workflow_sod_aligned_references"' in trainer
    assert '"workflow_sod_reference_id"' in trainer
    assert '"aux/workflow_sod/reference_switch_count"' in trainer
    assert '"aux/workflow_sod/future_reference_action_count"' in trainer
    assert "should_score_workflow_sod_target(" in trainer
    assert "score_invalid_target = sod_only and not forward_kl_only" in trainer
    assert "sod_only=score_invalid_target" in trainer
    assert '"aux/workflow_sod/scored_invalid_target_row_count"' in trainer
    assert '"aux/workflow_sod/format_prefix_teacher_gap_mean"' in trainer
    assert 'context["has_invalid_action"] = True' in trainer
    assert '"aux/workflow_sod/empty_signal_step"' in trainer
    assert '"aux/workflow_sod/actor_update_skipped_no_signal"' in trainer
    assert "and not skip_pure_sod_actor_update" in trainer
    assert "pure SOD produced no aligned or explicit-missing teacher probes" not in trainer
    assert "pure SOD requires at least one scored response token" not in trainer
    assert "workflow_sod_frozen_teacher" in main
    assert "self.ref_policy_wg.compute_ref_log_prob" in trainer
    assert 'reference_source == "fixed_bank"' in trainer


def test_sod_weighted_privileged_opsd_launcher_has_all_method_hard_gates():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root
        / "exp"
        / "run_dagent_sod_weighted_privileged_opsd_train1000_zju18_gpu2345.sh"
    ).read_text(encoding="utf-8")
    assert "WORKFLOW_SOD_TEACHER_POLICY=frozen_reference" in launcher
    assert "WORKFLOW_SOD_REFERENCE_SOURCE=fixed_bank" in launcher
    assert "WORKFLOW_SOD_OBJECTIVE=sampled_advantage" in launcher
    assert "WORKFLOW_SOD_ONLY=true" in launcher
    assert "ONLINE_OPSD_ENABLED=false" in launcher
    assert "accepted_trajectories.jsonl" in launcher
    assert "--require-count 1000" in launcher
    assert "SAVE_FREQ=\"${SAVE_FREQ:-10}\"" in launcher


def test_sod_faithful_privileged_launcher_changes_only_teacher_information():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root
        / "exp"
        / "run_dagent_sod_faithful_privileged_train1000_zju18_gpu2345.sh"
    ).read_text(encoding="utf-8")
    trainer = (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
        encoding="utf-8"
    )

    assert "WORKFLOW_SOD_COMPATIBILITY_MODE=true" in launcher
    assert "WORKFLOW_SOD_OBJECTIVE=sampled_advantage" in launcher
    assert "WORKFLOW_SOD_TEACHER_POLICY=frozen_reference" in launcher
    assert "WORKFLOW_SOD_REFERENCE_SOURCE=fixed_bank" in launcher
    assert "WORKFLOW_SOD_PRIVILEGED_REFERENCE_SCOPE=full_trajectory" in launcher
    assert "WORKFLOW_SOD_REQUIRE_FULL_PRIVILEGED_REFERENCE=true" in launcher
    assert "ACTOR_LOSS_AGG_MODE=token-mean" in launcher
    assert "ACTOR_CLIP_RATIO_LOW=0.2" in launcher
    assert "ACTOR_CLIP_RATIO_HIGH=0.28" in launcher
    assert "ONLINE_OPSD_ENABLED=false" in launcher
    assert "DAG_OPSD_ENABLED=false" in launcher
    assert "render_full_privileged_reference(reference_records)" in trainer
    assert '"actor/official_sod_token_mean_loss": 1.0' in trainer
    assert "SOD compatibility mode must score every on-policy action" in trainer


def test_pure_sod_launcher_has_isolated_objective_and_step10_checkpoints():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "exp" / "run_dagent_pure_sod_aligned_mixed3_formal_zju18.sh"
    ).read_text(encoding="utf-8")
    elastic = (
        root / "exp" / "run_dagent_pure_sod_elastic_gpu.sh"
    ).read_text(encoding="utf-8")
    assert "WORKFLOW_SOD_ONLY=true" in elastic
    assert "WORKFLOW_SOD_ENABLED=true" in elastic
    assert "DAG_OPSD_ENABLED=false" in elastic
    assert "ONLINE_OPSD_ENABLED=false" in elastic
    assert "USE_KL_LOSS=false" in elastic
    assert "FINAL_ANSWER_RETRY_LIMIT=0" in launcher
    assert "PPO_MINI_BATCH_SIZE=\"${PPO_MINI_BATCH_SIZE:-6}\"" in launcher
    assert "four actor updates per global step for both DP1 and DP2" in launcher
    assert 'ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"' in launcher
    assert 'SAVE_FREQ="${SAVE_FREQ:-10}"' in launcher


def test_zju53_pure_sod_uses_padded_sdpa_without_flash_attn_dependency():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "exp" / "run_dagent_pure_sod_aligned_mixed3_formal_zju53_gpu5.sh"
    ).read_text(encoding="utf-8")
    shared_launcher = (root / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(
        encoding="utf-8"
    )
    assert 'DAGENT_CUDA_VISIBLE_DEVICES="${DAGENT_CUDA_VISIBLE_DEVICES:-5}"' in launcher
    assert "MODEL_ATTN_IMPLEMENTATION=sdpa" in launcher
    assert "ACTOR_USE_REMOVE_PADDING=false" in launcher
    assert "DAGENT_MINIMUM_AVAILABLE_RAM_MIB:-16384" in launcher
    assert 'actor_rollout_ref.model.use_remove_padding="${ACTOR_USE_REMOVE_PADDING:-True}"' in shared_launcher


def test_zju53_gpu01_pure_sod_requires_node_memory_headroom():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "exp" / "run_dagent_pure_sod_aligned_mixed3_formal_zju53_gpu01.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_MINIMUM_AVAILABLE_RAM_MIB:-65536" in launcher


def test_zju46_gpu2_pure_sod_uses_single_a6000_safe_settings():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "exp" / "run_dagent_pure_sod_aligned_mixed3_formal_zju46_gpu2.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2" in launcher
    assert "rosa-vllm010" in launcher
    assert "MODEL_ATTN_IMPLEMENTATION=flash_attention_2" in launcher
    assert "ACTOR_USE_REMOVE_PADDING=true" in launcher
    assert "ACTOR_OPTIMIZER_OFFLOAD=true" in launcher
    assert "ACTOR_OPTIMIZER_FUSED=true" in launcher
    assert 'RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/r46s}"' in launcher
    assert 'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-24}"' in launcher
    assert 'PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}"' in launcher
    optimizer_config = (root / "verl" / "workers" / "config" / "optimizer.py").read_text(
        encoding="utf-8"
    )
    optimizer_yaml = (root / "verl" / "trainer" / "config" / "optim" / "fsdp.yaml").read_text(
        encoding="utf-8"
    )
    fsdp_worker = (root / "verl" / "workers" / "fsdp_workers.py").read_text(
        encoding="utf-8"
    )
    shared_launcher = (root / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(
        encoding="utf-8"
    )
    assert "fused: Optional[bool] = None" in optimizer_config
    assert "fused: null" in optimizer_yaml
    assert 'fused=optim_config.get("fused", None)' in fsdp_worker
    assert 'actor_rollout_ref.actor.optim.fused="${ACTOR_OPTIMIZER_FUSED}"' in shared_launcher
    dp_actor = (root / "verl" / "workers" / "actor" / "dp_actor.py").read_text(
        encoding="utf-8"
    )
    assert "load_fsdp_optimizer(self.actor_optimizer, get_device_id())" in dp_actor
    assert "offload_fsdp_optimizer(self.actor_optimizer)" in dp_actor
    update_actor = fsdp_worker.split("def update_actor(self, data: DataProto):", 1)[1]
    update_actor = update_actor.split("def start_profile", 1)[0]
    assert "if self._is_offload_optimizer:\n            load_fsdp_optimizer" not in update_actor
    checkpoint_manager = (
        root / "verl" / "utils" / "checkpoint" / "fsdp_checkpoint_manager.py"
    ).read_text(encoding="utf-8")
    load_checkpoint = checkpoint_manager.split("def load_checkpoint(", 1)[1]
    load_checkpoint = load_checkpoint.split("def save_checkpoint", 1)[0]
    assert "optimizer_offload: bool = False" in load_checkpoint
    assert "offload_fsdp_model_to_cpu(self.model)" in load_checkpoint
    assert "offload_fsdp_optimizer(self.optimizer)" in load_checkpoint
    assert "return model_offloaded_for_optimizer" in load_checkpoint
    actor_load_checkpoint = fsdp_worker.split(
        "def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):",
        1,
    )[1].split("def start_profile", 1)[0]
    assert "optimizer_offload=self._is_offload_optimizer" in actor_load_checkpoint
    assert "elif model_offloaded_for_optimizer:" in actor_load_checkpoint
    assert "load_fsdp_model_to_gpu(self.actor_module_fsdp)" in actor_load_checkpoint


def test_zju55_qwen15b_pure_sod_is_small_unsft_dp4_probe():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "exp" / "run_dagent_pure_sod_qwen15b_zju55_gpu0123.sh"
    ).read_text(encoding="utf-8")
    assert "Qwen2.5-1.5B-Instruct" in launcher
    assert "DAGENT_CUDA_VISIBLE_DEVICES:-0,1,2,3" in launcher
    assert "PPO_MINI_BATCH_SIZE=24" in launcher
    assert "ACTOR_RESPONSE_ONLY_LOGITS=true" in launcher
    assert "TOTAL_TRAINING_STEPS:-50" in launcher
    assert "run_dagent_pure_sod_elastic_gpu.sh" in launcher


def test_pure_sod_checkpoint_evaluator_requires_all_shards_and_both_suites():
    root = Path(__file__).resolve().parents[2]
    evaluator = (
        root / "exp" / "evaluate_dagent_pure_sod_checkpoint.sh"
    ).read_text(encoding="utf-8")
    assert "STEP must be a positive multiple of 10" in evaluator
    assert "model optim extra_state" in evaluator
    assert "fsdp_config.json" in evaluator
    assert "data.pt" in evaluator
    assert "verl.model_merger merge" in evaluator
    assert "evaluate_dagent_formal_v2_model.sh" in evaluator
    assert "evaluate_dagent_evidence_first_train1000_model.sh" in evaluator
    assert 'checkpoint_name="$(basename -- "${CHECKPOINT_ROOT}")"' in evaluator
    assert 'RUN_NAME="${RUN_NAME:-${checkpoint_slug}_step${STEP}}"' in evaluator
    assert 'FORMAL_TEST_MANIFEST_DIR="${FORMAL_TEST_MANIFEST_DIR:-' in evaluator
    assert 'int(summary.get("errors", -1)) != 0' in evaluator
    assert 'normalize_answers(row.golden_answers)' in evaluator
    assert 'normalize_answers(row.get("gold_answer"))' in evaluator
    assert 'actual_examples != expected_examples' in evaluator
    assert "formal_complete_with_retries" in evaluator
    assert "retrying (${attempt}/3)" in evaluator
    assert 'payload.get("metadata_joined", -1)' in evaluator
    assert 'payload.get("pipeline_error_rate", -1.0)' in evaluator
    assert '("by_reasoning_mode", "by_relation")' in evaluator
    assert "Evaluation lock ownership cannot be verified locally" in evaluator
    assert 'live_start="$(awk' in evaluator
    assert "--query-compute-apps=gpu_uuid,pid" in evaluator
    assert "has compute PID(s)" in evaluator
    assert "refusing to share" in evaluator
