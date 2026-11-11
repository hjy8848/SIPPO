import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts" / "build_dagent_search_e1_pairs.py"
TRAINER = ROOT / "scripts" / "train_dagent_search_e1_opsd.py"
DECIDER = ROOT / "scripts" / "decide_dagent_opsd_pilot.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def event(role, response):
    return {
        "agent_name": role,
        "messages": [
            {"role": "system", "content": f"Act as {role}."},
            {"role": "user", "content": "Question: q"},
        ],
        "response": response,
    }


def workflow_row(
    rollout_index,
    em,
    retrieval_calls,
    plan,
    answer,
    context="evidence",
    sample_index="q1",
    reason_to_search=False,
    grounded_by_nodes=True,
    plan_warnings=None,
):
    nodes = [
        {
            "id": "1",
            "type": "search",
            "query": "query",
            "search_context": context,
            "conclusion": answer,
        }
    ]
    if reason_to_search:
        nodes.extend(
            [
                {"id": "2", "type": "reason", "query": "select #1#", "conclusion": answer},
                {
                    "id": "3",
                    "type": "search",
                    "query": "#2# occupation",
                    "search_context": context,
                    "conclusion": answer,
                },
            ]
        )
    return {
        "sample_index": sample_index,
        "rollout_index": rollout_index,
        "question": "q",
        "gold_answer": ["SECRET_GOLD_FIELD"],
        "pred_answer": answer,
        "em": em,
        "retrieval_calls": retrieval_calls,
        "raw_trace": {
            "final_answer": answer,
            "verification": {
                "grounded_by_nodes": grounded_by_nodes,
                "direct_fallback_used": False,
            },
            "plan_warnings": plan_warnings or [],
            "sft_events": [
                event("PlanningAgent", plan),
                event("AnswerSummarizationAgent", f"<answer>{answer}</answer>"),
            ],
            "nodes_execution": nodes,
        },
    }


def test_pair_mining_uses_correct_valid_pool_without_efficiency_ranking(tmp_path):
    builder = load_module("search_e1_builder", BUILDER)
    rows = [
        workflow_row(0, 1, 3, "<plan>long correct</plan>", "correct-a"),
        workflow_row(1, 1, 1, "<plan>efficient correct</plan>", "correct-b"),
        workflow_row(2, 0, 2, "<plan>nearby failure</plan>", "wrong-a"),
        workflow_row(3, 0, 4, "<plan>completely different failed branch</plan>", "wrong-b"),
    ]
    log = tmp_path / "rollouts.json"
    log.write_text(builder.json.dumps(rows), encoding="utf-8")
    rejected = builder.Counter()
    workflows = builder.read_workflows(
        [str(log)], set(builder.DEFAULT_ROLES), 0, rejected
    )
    pair = builder.choose_pair(workflows)
    assert pair is not None
    reference, student = pair
    assert reference.rollout_index == 0
    assert reference.retrieval_calls == 3
    assert student.outcome == 0

    examples = builder.build_examples(workflows, rejected)
    assert {row["role"] for row in examples} == {
        "PlanningAgent",
        "AnswerSummarizationAgent",
    }
    assert all(row["student_response"] for row in examples)
    assert all(row["student_outcome"] == 0 for row in examples)
    assert all("SECRET_GOLD_FIELD" not in row["privileged_reference"] for row in examples)


def test_pair_mining_requires_an_incorrect_student(tmp_path):
    builder = load_module("search_e1_builder_all_correct", BUILDER)
    rows = [
        workflow_row(0, 1, 1, "<plan>short</plan>", "a"),
        workflow_row(1, 1, 2, "<plan>different correct trajectory</plan>", "a"),
    ]
    log = tmp_path / "all_correct.json"
    log.write_text(builder.json.dumps(rows), encoding="utf-8")
    rejected = builder.Counter()
    workflows = builder.read_workflows(
        [str(log)], set(builder.DEFAULT_ROLES), 0, rejected
    )
    assert builder.choose_pair(workflows) is None


def test_aligned_success_uses_success_action_and_redacts_current_block(tmp_path):
    builder = load_module("search_e1_builder_aligned", BUILDER)
    rows = [
        workflow_row(0, 1, 1, "<plan>successful plan</plan>", "right"),
        workflow_row(1, 0, 1, "<plan>failed plan</plan>", "wrong"),
    ]
    log = tmp_path / "aligned.json"
    log.write_text(builder.json.dumps(rows), encoding="utf-8")
    rejected = builder.Counter()
    workflows = builder.read_workflows(
        [str(log)],
        set(builder.DEFAULT_ROLES),
        0,
        rejected,
        require_grounded_by_nodes=True,
        require_no_plan_warnings=True,
        require_complete_nodes=True,
    )
    examples = builder.build_examples(workflows, rejected, mode="aligned_success")
    assert len(examples) == 2
    assert all(row["student_outcome"] == 1 for row in examples)
    assert all(row["target_source"] == "successful_on_policy_action" for row in examples)
    for row in examples:
        marker = f'<agent role="{row["role"]}" step="{row["workflow_step"]}">\n'
        block = row["privileged_reference"].split(marker, 1)[1].split("\n</agent>", 1)[0]
        assert block == "[CURRENT_ACTION_REDACTED]"
        assert row["student_response"] != block


def test_aligned_success_accepts_one_unique_successful_workflow(tmp_path):
    builder = load_module("search_e1_builder_one_success", BUILDER)
    rows = [workflow_row(0, 1, 1, "<plan>only plan</plan>", "right")]
    log = tmp_path / "one.json"
    log.write_text(builder.json.dumps(rows), encoding="utf-8")
    rejected = builder.Counter()
    workflows = builder.read_workflows(
        [str(log)], set(builder.DEFAULT_ROLES), 0, rejected
    )
    examples = builder.build_examples(workflows, rejected, mode="aligned_success")
    assert len(examples) == 2


def test_strict_teacher_gate_rejects_ungrounded_success(tmp_path):
    builder = load_module("search_e1_builder_strict_gate", BUILDER)
    rows = [
        workflow_row(
            0,
            1,
            1,
            "<plan>ungrounded</plan>",
            "right",
            grounded_by_nodes=False,
        )
    ]
    log = tmp_path / "ungrounded.json"
    log.write_text(builder.json.dumps(rows), encoding="utf-8")
    rejected = builder.Counter()
    workflows = builder.read_workflows(
        [str(log)],
        set(builder.DEFAULT_ROLES),
        0,
        rejected,
        require_grounded_by_nodes=True,
    )
    assert workflows[0].outcome == 0
    assert not builder.build_examples(workflows, rejected, mode="aligned_success")
    assert rejected["teacher_not_grounded_by_nodes"] == 1


def test_reason_search_name_does_not_change_teacher_eligibility(tmp_path):
    builder = load_module("search_e1_builder_reason_search", BUILDER)
    rows = [
        workflow_row(
            0,
            1,
            1,
            "<plan>shortcut</plan>",
            "a",
            sample_index="reason_search_case",
        ),
        workflow_row(
            1,
            1,
            2,
            "<plan>grounded reason then search</plan>",
            "a",
            sample_index="reason_search_case",
            reason_to_search=True,
        ),
        workflow_row(
            2,
            0,
            2,
            "<plan>failed</plan>",
            "b",
            sample_index="reason_search_case",
        ),
    ]
    log = tmp_path / "reason_search.json"
    log.write_text(builder.json.dumps(rows), encoding="utf-8")
    rejected = builder.Counter()
    workflows = builder.read_workflows(
        [str(log)], set(builder.DEFAULT_ROLES), 0, rejected
    )
    reference, _ = builder.choose_pair(workflows)
    assert reference.rollout_index == 0
    assert reference.reason_to_search is False


def test_full_vocabulary_forward_kl_has_student_only_gradient():
    trainer = load_module("search_e1_trainer", TRAINER)
    student = torch.tensor([[0.2, -0.1, 0.5]], requires_grad=True)
    teacher = torch.tensor([[-0.3, 0.4, 0.1]])
    token_loss = trainer.pointwise_forward_kl(
        student,
        teacher,
        temperature=1.0,
        pointwise_clip=10.0,
    )
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    student_logp = torch.log_softmax(student, dim=-1)
    expected = torch.sum(teacher_logp.exp() * (teacher_logp - student_logp), dim=-1)
    assert torch.allclose(token_loss, expected)
    token_loss.mean().backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_pointwise_clip_caps_each_vocabulary_contribution():
    trainer = load_module("search_e1_trainer_clip", TRAINER)
    student = torch.tensor([[-100.0, 100.0]])
    teacher = torch.tensor([[100.0, -100.0]])
    unclipped = trainer.pointwise_forward_kl(student, teacher, pointwise_clip=0.0)
    clipped = trainer.pointwise_forward_kl(student, teacher, pointwise_clip=0.25)
    assert clipped < unclipped
    assert clipped <= torch.tensor([0.5])


def test_teacher_gap_mask_only_keeps_supported_target_tokens():
    trainer = load_module("search_e1_trainer_gap_mask", TRAINER)
    gaps = torch.tensor([0.3, -0.2, 0.0, 0.1])
    assert trainer.teacher_gap_mask(gaps, "none", 0.0).tolist() == [
        True,
        True,
        True,
        True,
    ]
    assert trainer.teacher_gap_mask(gaps, "token", 0.0).tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert trainer.teacher_gap_mask(gaps, "row", 0.0).tolist() == [
        True,
        True,
        True,
        True,
    ]


def test_pilot_gate_requires_real_em_gain_and_zero_errors():
    decider = load_module("search_e1_pilot_decider", DECIDER)
    rows = [
        {
            "step": "0",
            "em": "0.500",
            "f1": "0.600",
            "api_errors": "0",
            "pipeline_errors": "0",
            "adapter": "BASE",
            "merged_model": "base",
        },
        {
            "step": "10",
            "em": "0.506",
            "f1": "0.610",
            "api_errors": "0",
            "pipeline_errors": "0",
            "adapter": "a10",
            "merged_model": "m10",
        },
        {
            "step": "20",
            "em": "0.515",
            "f1": "0.620",
            "api_errors": "0",
            "pipeline_errors": "0",
            "adapter": "a20",
            "merged_model": "m20",
        },
    ]
    accepted = decider.decide(rows, 0.01, {10, 20})
    assert accepted["formal_training_allowed"] is True
    assert accepted["best_candidate_step"] == 20
    rejected = decider.decide(rows, 0.02, {10, 20})
    assert rejected["formal_training_allowed"] is False


def test_response_logits_requests_only_response_window():
    trainer = load_module("search_e1_trainer_logits_window", TRAINER)

    class DummyModel:
        requested = None

        def __call__(self, input_ids, use_cache, logits_to_keep):
            del use_cache
            self.requested = logits_to_keep
            values = torch.arange(logits_to_keep * 3, dtype=torch.float32)
            return SimpleNamespace(logits=values.reshape(1, logits_to_keep, 3))

    model = DummyModel()
    prepared = (torch.tensor([[1, 2, 3, 4, 5]]), 2)
    logits = trainer.response_logits(model, prepared, response_length=3)
    assert model.requested == 4
    assert logits.shape == (3, 3)
    assert torch.equal(logits, torch.arange(9, dtype=torch.float32).reshape(3, 3))


def test_group_rows_preserves_agent_order():
    trainer = load_module("search_e1_trainer_group", TRAINER)
    rows = [
        {"group_id": "g", "workflow_step": 2, "role": "ReasonerAgent"},
        {"group_id": "g", "workflow_step": 0, "role": "PlanningAgent"},
    ]
    grouped = trainer.group_rows(rows)
    assert [row["role"] for row in grouped[0]] == ["PlanningAgent", "ReasonerAgent"]


def test_wandb_init_failure_does_not_call_blocking_teardown(monkeypatch, tmp_path):
    trainer = load_module("search_e1_trainer_wandb_failure", TRAINER)
    teardown_called = False

    def fail_init(**kwargs):
        del kwargs
        raise TimeoutError("offline")

    def teardown(**kwargs):
        nonlocal teardown_called
        del kwargs
        teardown_called = True

    fake_wandb = SimpleNamespace(
        init=fail_init,
        teardown=teardown,
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    args = SimpleNamespace(
        wandb_project="DAGENT",
        wandb_run_name="test",
        output_dir=tmp_path,
    )
    assert trainer.init_wandb(args) is None
    assert teardown_called is False
