from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

import torch
from torch import nn

try:
    import ray  # noqa: F401
except ModuleNotFoundError:
    import pytest

    pytest.skip("Ray is required for actor integration tests", allow_module_level=True)

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import set_actor_update_temperature


def test_outcome_launcher_preserves_explicit_online_opsd_mode():
    launcher = (
        Path(__file__).parents[2]
        / "exp"
        / "run_dagent_outcome_grpo_elastic_gpu.sh"
    ).read_text(encoding="utf-8")
    assert 'ONLINE_OPSD_ENABLED="${ONLINE_OPSD_ENABLED:-false}"' in launcher
    assert "export ONLINE_OPSD_ENABLED=false" not in launcher


def test_forward_kl_launcher_and_pair_builder_use_the_same_five_roles():
    root = Path(__file__).parents[2]
    launcher = (
        root
        / "exp"
        / "run_dagent_pure_sod_forward_kl_mixed3_formal_zju18_gpu2345.sh"
    ).read_text(encoding="utf-8")
    trainer = (root / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
        encoding="utf-8"
    )

    assert "ONLINE_OPSD_SELECTED_ROLES=" in launcher
    for role in (
        "PlanningAgent",
        "QueryRewriteAgent",
        "ReasonerAgent",
        "AnswerSummarizationAgent",
        "AnswerGenerationAgent",
    ):
        assert role in launcher
    assert "actor_rollout_ref.actor.online_opsd.selected_roles" in trainer
    assert "role not in forward_kl_selected_roles" in trainer
    assert (
        'MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}"'
        in launcher
    )
    assert 'ACTOR_USE_REMOVE_PADDING="${ACTOR_USE_REMOVE_PADDING:-true}"' in launcher
    assert 'CRITIC_USE_REMOVE_PADDING="${CRITIC_USE_REMOVE_PADDING:-true}"' in launcher


def test_forward_kl_excludes_invalid_students_and_dp_padding_duplicates():
    trainer = (
        Path(__file__).parents[2] / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert "score_invalid_target = sod_only and not forward_kl_only" in trainer
    assert "sod_only=score_invalid_target" in trainer
    assert "if target_row == canonical_rows[key]:" in trainer
    assert "online_opsd_eligible[target_row] = True" in trainer
    assert "forward_kl_duplicate_eligible_suppressed_count += 1" in trainer
    assert (
        '"aux/workflow_sod/forward_kl_duplicate_eligible_suppressed_count"'
        in trainer
    )


def _forward_kl_batch() -> DataProto:
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2]]),
            "response_mask": torch.ones(1, 2, dtype=torch.long),
            "input_ids": torch.tensor([[3, 4, 1, 2]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.arange(4).unsqueeze(0),
            "online_opsd_teacher_input_ids": torch.tensor([[9, 4, 1, 2]]),
            "online_opsd_teacher_attention_mask": torch.ones(1, 4, dtype=torch.long),
            "online_opsd_teacher_position_ids": torch.arange(4).unsqueeze(0),
            "online_opsd_token_mask": torch.ones(1, 2, dtype=torch.bool),
            "online_opsd_eligible": torch.ones(1, dtype=torch.bool),
            "online_opsd_role_code": torch.zeros(1, dtype=torch.long),
        }
    )
    batch.meta_info.update(
        {
            "temperature": 1.0,
            "global_steps": 1,
            "online_opsd_has_pairs": True,
        }
    )
    return batch


class _ScalarPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))


def _make_actor(
    *,
    min_kl: float,
    max_kl: float,
    max_rows_per_rank: int = 1,
    loss_token_start: int = 0,
    gate_on_full_response_kl: bool = True,
    exclude_opening_tag_from_loss: bool = False,
) -> DataParallelPPOActor:
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = _ScalarPolicy()
    actor.actor_optimizer = torch.optim.AdamW(
        actor.actor_module.parameters(), lr=0.1, weight_decay=0.5
    )
    actor.device_name = "cpu"
    actor.config = SimpleNamespace(
        grad_clip=10.0,
        fsdp_config=SimpleNamespace(optimizer_offload=False),
        online_opsd={
            "enabled": True,
            "standalone": True,
            "pair_source": "workflow_sod",
            "update_interval": 1,
            "max_rows_per_rank": max_rows_per_rank,
            "loss_weight": 1.0,
            "min_teacher_confidence": 0.0,
            "min_forward_kl": min_kl,
            "max_forward_kl": max_kl,
            "divergence_temperature": 1.0,
            "pointwise_kl_clip": 0.0,
            "loss_token_start": loss_token_start,
            "gate_on_full_response_kl": gate_on_full_response_kl,
            "exclude_opening_tag_from_loss": exclude_opening_tag_from_loss,
        },
    )

    def fake_logits(
        self,
        input_ids,
        attention_mask,
        position_ids,
        response_length,
        valid_response_length,
        temperature,
    ):
        del attention_mask, position_ids, response_length, temperature
        if int(input_ids[0, 0].item()) == 9:
            logits = torch.tensor([2.0, -1.0, -1.0])
        else:
            value = self.actor_module.logit
            logits = torch.stack((value, -value, value * 0.0))
        return logits.view(1, 1, 3).expand(1, valid_response_length, 3)

    actor._online_opsd_response_logits = MethodType(fake_logits, actor)
    return actor


def _repeat_forward_kl_batch(rows: int) -> DataProto:
    source = _forward_kl_batch()
    batch = DataProto.from_dict(
        tensors={
            key: value.repeat((rows,) + (1,) * (value.ndim - 1))
            for key, value in source.batch.items()
        }
    )
    batch.meta_info.update(source.meta_info)
    return batch


def test_rejected_forward_kl_pair_does_not_apply_adamw_weight_decay():
    actor = _make_actor(min_kl=100.0, max_kl=101.0)
    before = actor.actor_module.logit.detach().clone()

    with mock.patch(
        "verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")
    ):
        metrics = actor._update_online_opsd(_forward_kl_batch(), temperature=1.0)

    assert metrics["aux/workflow_sod_forward_kl/optimizer_step"] == [0.0]
    assert actor.actor_module.logit.detach().equal(before)
    assert actor.actor_optimizer.state == {}


def test_accepted_forward_kl_pair_updates_student_without_rl_loss():
    actor = _make_actor(min_kl=0.0, max_kl=100.0)
    before = actor.actor_module.logit.detach().clone()

    with mock.patch(
        "verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")
    ):
        metrics = actor._update_online_opsd(_forward_kl_batch(), temperature=1.0)

    assert metrics["aux/workflow_sod_forward_kl/optimizer_step"] == [1.0]
    assert metrics["aux/workflow_sod_forward_kl/forward_kl"][0] > 0.0
    assert not actor.actor_module.logit.detach().equal(before)


def test_forward_kl_loss_token_start_skips_a_short_response_without_weight_decay():
    actor = _make_actor(
        min_kl=0.0,
        max_kl=100.0,
        loss_token_start=2,
    )
    before = actor.actor_module.logit.detach().clone()

    with mock.patch(
        "verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")
    ):
        metrics = actor._update_online_opsd(_forward_kl_batch(), temperature=1.0)

    prefix = "aux/workflow_sod_forward_kl"
    assert metrics[f"{prefix}/optimizer_step"] == [0.0]
    assert metrics[f"{prefix}/selected_token_count"] == [0.0]
    assert metrics[f"{prefix}/full_selected_token_count"] == [2.0]
    assert metrics[f"{prefix}/loss_token_start"] == [2.0]
    assert actor.actor_module.logit.detach().equal(before)
    assert actor.actor_optimizer.state == {}


def test_opening_tag_excluded_loss_keeps_full_response_gate_weight():
    actor = _make_actor(
        min_kl=0.05,
        max_kl=100.0,
        exclude_opening_tag_from_loss=True,
    )

    def position_specific_logits(
        self,
        input_ids,
        attention_mask,
        position_ids,
        response_length,
        valid_response_length,
        temperature,
    ):
        del attention_mask, position_ids, response_length, temperature
        if int(input_ids[0, 0].item()) == 9:
            return torch.tensor(
                [[[4.0, -4.0, 0.0], [0.1, -0.1, 0.0]]]
            )[:, :valid_response_length]
        value = self.actor_module.logit
        logits = torch.stack((value, -value, value * 0.0))
        return logits.view(1, 1, 3).expand(1, valid_response_length, 3)

    actor._online_opsd_response_logits = MethodType(position_specific_logits, actor)

    batch = _forward_kl_batch()
    batch.batch["online_opsd_opening_tag_excluded_mask"] = torch.tensor(
        [[0, 1]], dtype=torch.bool
    )
    with mock.patch(
        "verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")
    ):
        metrics = actor._update_online_opsd(batch, temperature=1.0)

    prefix = "aux/workflow_sod_forward_kl"
    assert metrics[f"{prefix}/optimizer_step"] == [1.0]
    assert metrics[f"{prefix}/selected_token_count"] == [1.0]
    assert metrics[f"{prefix}/exclude_opening_tag_from_loss"] == [1.0]
    assert metrics[f"{prefix}/excluded_opening_tag_token_count"] == [1.0]
    assert metrics[f"{prefix}/gate_on_full_response_kl"] == [1.0]
    assert metrics[f"{prefix}/gate_forward_kl"][0] == metrics[
        f"{prefix}/full_response_forward_kl"
    ][0]
    assert metrics[f"{prefix}/gate_forward_kl"][0] > metrics[
        f"{prefix}/forward_kl"
    ][0]


def test_forward_kl_accumulates_pairs_into_one_offloaded_optimizer_step():
    actor = _make_actor(min_kl=0.0, max_kl=100.0, max_rows_per_rank=2)
    actor.config.fsdp_config.optimizer_offload = True

    with (
        mock.patch(
            "verl.workers.actor.dp_actor.get_device_id",
            return_value=torch.device("cpu"),
        ),
        mock.patch(
            "verl.workers.actor.dp_actor.load_fsdp_optimizer"
        ) as load_optimizer,
        mock.patch(
            "verl.workers.actor.dp_actor.offload_fsdp_optimizer"
        ) as offload_optimizer,
        mock.patch.object(
            actor.actor_optimizer,
            "step",
            wraps=actor.actor_optimizer.step,
        ) as optimizer_step,
    ):
        metrics = actor._update_online_opsd(
            _repeat_forward_kl_batch(2), temperature=1.0
        )

    assert metrics["aux/workflow_sod_forward_kl/optimizer_step"] == [1.0]
    assert metrics[
        "aux/workflow_sod_forward_kl/optimizer_accepted_pair_count"
    ] == [2.0]
    assert metrics["aux/workflow_sod_forward_kl/gradient_scale"] == [0.5]
    assert optimizer_step.call_count == 1
    load_optimizer.assert_called_once()
    offload_optimizer.assert_called_once()


def test_standalone_update_policy_does_not_require_ppo_tensors():
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = nn.Linear(1, 1)
    actor.actor_optimizer = mock.Mock()
    actor.config = SimpleNamespace(
        online_opsd={
            "enabled": True,
            "standalone": True,
            "pair_source": "workflow_sod",
        }
    )
    actor._update_online_opsd = mock.Mock(
        return_value={"aux/workflow_sod_forward_kl/optimizer_step": [1.0]}
    )

    metrics = actor.update_policy(_forward_kl_batch())

    actor._update_online_opsd.assert_called_once()
    assert metrics["actor/ppo_optimizer_step"] == 0.0
    assert metrics["actor/standalone_forward_kl"] == 1.0


def test_actor_update_temperature_matches_rollout_config():
    batch = _forward_kl_batch()
    batch.meta_info.pop("temperature")

    set_actor_update_temperature(batch, SimpleNamespace(temperature=0.8))

    assert batch.meta_info["temperature"] == 0.8


def test_standalone_actor_rejects_missing_temperature_metadata():
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = nn.Linear(1, 1)
    batch = _forward_kl_batch()
    batch.meta_info.pop("temperature")

    try:
        actor.update_policy(batch)
    except KeyError as error:
        assert error.args == ("temperature",)
    else:
        raise AssertionError("missing actor temperature metadata was silently accepted")
