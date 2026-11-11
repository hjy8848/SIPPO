# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, Qwen3Config
from transformers.modeling_outputs import CausalLMOutput

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import FSDPActorConfig, OptimizerConfig


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing DataParallelPPOActor"""

    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True), num_layers=2
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=0,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape

        embeddings = self.embedding(input_ids)
        hidden_states = self.transformer(embeddings)
        logits = self.lm_head(hidden_states)
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits.index_select(1, logits_to_keep)
        elif logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        self.last_projected_sequence_length = logits.shape[1]

        return CausalLMOutput(logits=logits)


class TestDataParallelPPOActor(unittest.TestCase):
    """Test DataParallelPPOActor compute_log_prob and update_policy methods"""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment"""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://"
            )

        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(cls.rank)
            cls.device = torch.device(f"cuda:{cls.rank}")
        else:
            cls.device = torch.device("cpu")

    def setUp(self):
        """Set up test fixtures"""
        self.config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            clip_ratio=0.2,
            entropy_coeff=0.01,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,  # Disable torch.compile for testing
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
        )

        self.mock_model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        self.mock_optimizer = torch.optim.Adam(self.mock_model.parameters(), lr=1e-4)

        self.actor = DataParallelPPOActor(
            config=self.config, actor_module=self.mock_model, actor_optimizer=self.mock_optimizer
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _create_test_data_for_compute_log_prob(self):
        """Create test DataProto for compute_log_prob method"""
        batch_size = 2
        prompt_length = 8
        response_length = 4
        total_length = prompt_length + response_length
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, total_length)).to(self.device)
        attention_mask = torch.ones(batch_size, total_length).to(self.device)
        position_ids = torch.arange(total_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        responses = input_ids[:, -response_length:]  # Last part is the response

        tensor_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
            },
            batch_size=[batch_size],
        )

        meta_info = {"micro_batch_size": batch_size, "temperature": 1.0, "use_dynamic_bsz": False}

        return DataProto(batch=tensor_dict, meta_info=meta_info)

    def _create_test_data_for_update_policy(self):
        """Create test DataProto for update_policy method"""
        batch_size = 4  # Must match ppo_mini_batch_size
        prompt_length = 8
        response_length = 4
        total_length = prompt_length + response_length
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, total_length)).to(self.device)
        attention_mask = torch.ones(batch_size, total_length).to(self.device)
        position_ids = torch.arange(total_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        responses = input_ids[:, -response_length:]
        response_mask = torch.ones(batch_size, response_length).to(self.device)
        old_log_probs = torch.randn(batch_size, response_length).to(self.device) * 0.1  # Small values
        advantages = torch.randn(batch_size, response_length).to(self.device) * 0.5

        tensor_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
                "old_log_probs": old_log_probs,
                "advantages": advantages,
            },
            batch_size=[batch_size],
        )

        meta_info = {"temperature": 1.0}

        return DataProto(batch=tensor_dict, meta_info=meta_info)

    def test_compute_log_prob(self):
        """Test compute_log_prob method"""
        data = self._create_test_data_for_compute_log_prob()

        log_probs, entropies = self.actor.compute_log_prob(data, calculate_entropy=True)

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        self.assertIsInstance(entropies, torch.Tensor)
        self.assertEqual(entropies.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(entropies)))
        self.assertTrue(torch.all(entropies >= 0))  # Entropy should be non-negative

    def test_compute_log_prob_without_entropy(self):
        """Test compute_log_prob method without entropy calculation"""
        data = self._create_test_data_for_compute_log_prob()

        log_probs, entropies = self.actor.compute_log_prob(data, calculate_entropy=False)

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        self.assertIsNone(entropies)

    def test_forced_single_row_dynamic_microbatches(self):
        config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=2,
            ppo_epochs=1,
            use_dynamic_bsz=True,
            ppo_max_token_len_per_gpu=24,
            force_single_row_dynamic_microbatches=True,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
        )
        actor = DataParallelPPOActor(
            config=config,
            actor_module=self.mock_model,
            actor_optimizer=self.mock_optimizer,
        )
        data = self._create_test_data_for_compute_log_prob()

        micro_batches, index_lists = actor._prepare_actor_dynamic_batch(
            data, max_token_len=24
        )

        self.assertEqual(len(micro_batches), len(data))
        self.assertTrue(all(len(micro_batch) == 1 for micro_batch in micro_batches))
        self.assertEqual(
            sorted(index for indices in index_lists for index in indices),
            list(range(len(data))),
        )

    def test_response_only_logits_matches_full_projection_with_remove_padding(self):
        input_ids = torch.randint(0, 1000, (2, 8), device=self.device)
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1]],
            device=self.device,
        )
        response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], device=self.device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": attention_mask.cumsum(dim=-1) - 1,
            "responses": input_ids[:, -3:],
            "response_mask": response_mask,
        }
        self.mock_model.eval()
        self.actor.use_remove_padding = True
        self.actor.response_only_logits = False
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

        def fake_unpad_input(hidden_states, mask):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            lengths = mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(lengths.cumsum(dim=0), (1, 0))
            return flat.index_select(0, indices), indices, cu_seqlens, int(lengths.max().item())

        def fake_index_first_axis(hidden_states, selected_indices):
            return hidden_states.index_select(0, selected_indices)

        def fake_pad_input(hidden_states, indices, batch, seqlen):
            output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
            output.index_copy_(0, indices, hidden_states)
            return output.reshape(batch, seqlen, *hidden_states.shape[1:])

        def fake_rearrange(hidden_states, pattern):
            self.assertEqual(pattern, "b s ... -> (b s) ...")
            return hidden_states.reshape(-1, *hidden_states.shape[2:])

        with (
            mock.patch("verl.workers.actor.dp_actor.unpad_input", fake_unpad_input),
            mock.patch("verl.workers.actor.dp_actor.index_first_axis", fake_index_first_axis),
            mock.patch("verl.workers.actor.dp_actor.pad_input", fake_pad_input),
            mock.patch("verl.workers.actor.dp_actor.rearrange", fake_rearrange),
        ):
            full_entropy, full_log_probs = self.actor._forward_micro_batch(
                model_inputs, temperature=1.0, calculate_entropy=True
            )

            self.actor.response_only_logits = True
            selected_entropy, selected_log_probs = self.actor._forward_micro_batch(
                model_inputs, temperature=1.0, calculate_entropy=True
            )

        valid_tokens = response_mask.bool()
        torch.testing.assert_close(selected_log_probs[valid_tokens], full_log_probs[valid_tokens])
        torch.testing.assert_close(selected_entropy[valid_tokens], full_entropy[valid_tokens])
        self.assertTrue(torch.equal(selected_log_probs[~valid_tokens], torch.zeros(1, device=self.device)))
        self.assertEqual(
            self.mock_model.last_projected_sequence_length,
            int(response_mask.sum().item()),
        )

    def test_update_policy(self):
        """Test update_policy method"""
        data = self._create_test_data_for_update_policy()

        metrics = self.actor.update_policy(data)

        self.assertIsInstance(metrics, dict)

        expected_metric_keys = [
            "actor/pg_loss",
            "actor/pg_clipfrac",
            "actor/ppo_kl",
            "actor/pg_clipfrac_lower",
            "actor/grad_norm",
        ]

        for key in expected_metric_keys:
            self.assertIn(key, metrics)
            if isinstance(metrics[key], list):
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics[key]))
            else:
                self.assertIsInstance(metrics[key], (float, int))
                self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))

    def test_planner_poc_parameter_gradient_audit_never_updates(self):
        config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            entropy_coeff=0.0,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
        )
        model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        actor = DataParallelPPOActor(
            config=config,
            actor_module=model,
            actor_optimizer=optimizer,
        )
        data = self._create_test_data_for_update_policy()
        response_mask = data.batch["response_mask"]
        real = torch.zeros_like(response_mask)
        real[0] = -0.25
        real[1] = 0.25
        shuffled = torch.zeros_like(response_mask)
        shuffled[2] = -0.25
        shuffled[3] = 0.25
        data.batch["workflow_loss_weights"] = response_mask / response_mask.sum()
        data.batch["planner_outcome_contrast_coefficients"] = real
        data.batch["planner_outcome_contrast_shuffled_coefficients"] = shuffled
        data.non_tensor_batch["trajectory_id"] = np.asarray(
            ["w0", "w1", "w2", "w3"], dtype=object
        )
        data.meta_info.update(
            {
                "planner_outcome_contrast_gradient_audit": True,
                "planner_outcome_contrast_loss_weight": 0.1,
            }
        )
        parameters_before = [
            parameter.detach().clone() for parameter in model.parameters()
        ]

        def cpu_logprobs(logits, labels):
            return torch.gather(
                torch.log_softmax(logits.float(), dim=-1),
                dim=-1,
                index=labels.unsqueeze(-1),
            ).squeeze(-1)

        with (
            mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
            mock.patch(
                "verl.workers.actor.dp_actor.logprobs_from_logits",
                cpu_logprobs,
            ),
        ):
            metrics = actor.update_policy(data)

        step.assert_not_called()
        self.assertEqual(metrics["actor/optimizer_steps_expected"], [0.0])
        self.assertEqual(metrics["actor/optimizer_steps_executed"], [0.0])
        self.assertEqual(
            metrics[
                "aux/planner_outcome_contrast/parameter_gradient_audit"
            ],
            [1.0],
        )
        self.assertIn(
            "aux/planner_outcome_contrast/parameter_real_poc_em_cosine",
            metrics,
        )
        self.assertIn(
            "aux/planner_outcome_contrast/parameter_real_shuffled_cosine",
            metrics,
        )
        for before, after in zip(parameters_before, model.parameters(), strict=True):
            torch.testing.assert_close(before, after)

    def test_question_preserving_poc_refreshes_before_every_optimizer_step(self):
        config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            entropy_coeff=0.0,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
        )
        model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        actor = DataParallelPPOActor(
            config=config,
            actor_module=model,
            actor_optimizer=optimizer,
        )
        data = self._create_test_data_for_update_policy()
        data.batch["actor_dp_valid_rows"] = torch.ones(
            len(data), dtype=torch.bool, device=self.device
        )
        data.non_tensor_batch.update(
            {
                "uid": np.asarray(["q0", "q0", "q1", "q1"], dtype=object),
                "trajectory_id": np.asarray(["w0", "w1", "w2", "w3"], dtype=object),
                "role": np.asarray(["PlanningAgent"] * 4, dtype=object),
                "outcome_exact": np.asarray([1.0, 0.0, 1.0, 0.0]),
                "workflow_step": np.zeros(4, dtype=np.int64),
            }
        )
        data.meta_info.update(
            {
                "planner_outcome_contrast_minibatch_mode": "question_preserving",
                "planner_outcome_contrast_mode": "real",
                "planner_outcome_contrast_beta": 1.0,
                "planner_outcome_contrast_loss_weight": 0.01,
                "planner_outcome_contrast_shuffle_seed": 17,
                "planner_outcome_contrast_require_signal": True,
                "planner_outcome_contrast_paired_shuffled_audit": True,
                "planner_outcome_contrast_data_parallel_size": 1,
            }
        )

        def cpu_logprobs(logits, labels):
            return torch.gather(
                torch.log_softmax(logits.float(), dim=-1),
                dim=-1,
                index=labels.unsqueeze(-1),
            ).squeeze(-1)

        parameter_snapshots = []
        original_current_log_probs = actor._current_minibatch_log_probs

        def tracked_current_log_probs(*args, **kwargs):
            parameter_snapshots.append(
                next(model.parameters()).detach().cpu().clone()
            )
            return original_current_log_probs(*args, **kwargs)

        with (
            mock.patch.object(
                actor,
                "_refresh_planner_outcome_contrast",
                wraps=actor._refresh_planner_outcome_contrast,
            ) as refresh,
            mock.patch.object(
                actor,
                "_current_minibatch_log_probs",
                side_effect=tracked_current_log_probs,
            ),
            mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
            mock.patch(
                "verl.workers.actor.dp_actor.logprobs_from_logits",
                cpu_logprobs,
            ),
        ):
            metrics = actor.update_policy(data)

        self.assertEqual(refresh.call_count, 2)
        self.assertEqual(len(parameter_snapshots), 2)
        self.assertFalse(torch.equal(parameter_snapshots[0], parameter_snapshots[1]))
        self.assertEqual(step.call_count, 2)
        self.assertEqual(metrics["actor/optimizer_steps_expected"], [2.0])
        self.assertEqual(metrics["actor/optimizer_steps_executed"], [2.0])
        self.assertEqual(
            metrics["actor/question_preserving_minibatch_count"], [2.0]
        )
        self.assertEqual(
            metrics["aux/planner_outcome_contrast/refreshed"], [1.0, 1.0]
        )
        ratios = metrics[
            "aux/planner_outcome_contrast/refreshed_weighted_poc_to_planner_em_direct_gradient_ratio"
        ]
        self.assertEqual(len(ratios), 2)
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in ratios))
        self.assertEqual(
            metrics["aux/planner_outcome_contrast/paired_shuffled_audit"],
            [1.0, 1.0],
        )
        self.assertTrue(
            all(
                value > 0
                for value in metrics[
                    "aux/planner_outcome_contrast/paired_shuffled_label_count"
                ]
            )
        )

    def test_question_preserving_poc_without_pairs_falls_back_to_finite_em_update(self):
        config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            entropy_coeff=0.0,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
        )
        model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        actor = DataParallelPPOActor(
            config=config,
            actor_module=model,
            actor_optimizer=optimizer,
        )
        data = self._create_test_data_for_update_policy()
        data.batch["actor_dp_valid_rows"] = torch.ones(
            len(data), dtype=torch.bool, device=self.device
        )
        data.non_tensor_batch.update(
            {
                "uid": np.asarray(["q0"] * 4, dtype=object),
                "trajectory_id": np.asarray(["w0", "w1", "w2", "w3"], dtype=object),
                "role": np.asarray(["PlanningAgent"] * 4, dtype=object),
                "outcome_exact": np.ones(4, dtype=np.float64),
                "workflow_step": np.zeros(4, dtype=np.int64),
            }
        )
        data.meta_info.update(
            {
                "planner_outcome_contrast_minibatch_mode": "question_preserving",
                "planner_outcome_contrast_mode": "real",
                "planner_outcome_contrast_beta": 1.0,
                "planner_outcome_contrast_loss_weight": 0.01,
                "planner_outcome_contrast_shuffle_seed": 17,
                "planner_outcome_contrast_require_signal": False,
                "planner_outcome_contrast_paired_shuffled_audit": True,
                "planner_outcome_contrast_data_parallel_size": 1,
            }
        )
        parameters_before = [
            parameter.detach().clone() for parameter in model.parameters()
        ]

        def cpu_logprobs(logits, labels):
            return torch.gather(
                torch.log_softmax(logits.float(), dim=-1),
                dim=-1,
                index=labels.unsqueeze(-1),
            ).squeeze(-1)

        with (
            mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
            mock.patch(
                "verl.workers.actor.dp_actor.logprobs_from_logits",
                cpu_logprobs,
            ),
        ):
            metrics = actor.update_policy(data)

        self.assertEqual(step.call_count, 1)
        self.assertEqual(metrics["actor/optimizer_steps_expected"], [1.0])
        self.assertEqual(metrics["actor/optimizer_steps_executed"], [1.0])
        self.assertEqual(
            metrics["aux/planner_outcome_contrast/empty_refreshed_minibatch"],
            [1.0],
        )
        self.assertEqual(
            metrics["aux/planner_outcome_contrast/global_signal_required"],
            [0.0],
        )
        self.assertEqual(
            metrics[
                "aux/planner_outcome_contrast/refreshed_weighted_poc_direct_gradient_norm"
            ],
            [0.0],
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(parameters_before, model.parameters(), strict=True)
            )
        )

    def test_question_preserving_current_policy_audit_never_updates(self):
        config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            entropy_coeff=0.0,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
        )
        model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        actor = DataParallelPPOActor(config=config, actor_module=model, actor_optimizer=optimizer)
        data = self._create_test_data_for_update_policy()
        data.batch["actor_dp_valid_rows"] = torch.ones(len(data), dtype=torch.bool, device=self.device)
        data.non_tensor_batch.update({
            "uid": np.asarray(["q0", "q0", "q1", "q1"], dtype=object),
            "trajectory_id": np.asarray(["w0", "w1", "w2", "w3"], dtype=object),
            "role": np.asarray(["PlanningAgent"] * 4, dtype=object),
            "outcome_exact": np.asarray([1.0, 0.0, 1.0, 0.0]),
            "workflow_step": np.zeros(4, dtype=np.int64),
        })
        data.meta_info.update({
            "planner_outcome_contrast_minibatch_mode": "question_preserving",
            "planner_outcome_contrast_mode": "real",
            "planner_outcome_contrast_beta": 1.0,
            "planner_outcome_contrast_loss_weight": 0.02,
            "planner_outcome_contrast_shuffle_seed": 17,
            "planner_outcome_contrast_require_signal": True,
            "planner_outcome_contrast_paired_shuffled_audit": True,
            "planner_outcome_contrast_current_policy_audit_only": True,
            "planner_outcome_contrast_data_parallel_size": 1,
        })
        before = [parameter.detach().clone() for parameter in model.parameters()]

        def cpu_logprobs(logits, labels):
            return torch.gather(torch.log_softmax(logits.float(), dim=-1), -1, labels.unsqueeze(-1)).squeeze(-1)

        with (
            mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
            mock.patch("verl.workers.actor.dp_actor.logprobs_from_logits", cpu_logprobs),
        ):
            metrics = actor.update_policy(data)

        self.assertEqual(step.call_count, 0)
        self.assertEqual(metrics["actor/optimizer_steps_expected"], [0.0])
        self.assertEqual(metrics["actor/optimizer_steps_executed"], [0.0])
        self.assertEqual(metrics["aux/planner_outcome_contrast/paired_shuffled_audit"], [1.0, 1.0])
        self.assertEqual(
            metrics["aux/planner_outcome_contrast/current_policy_audit_only"],
            [1.0],
        )
        for old, new in zip(before, model.parameters(), strict=True):
            torch.testing.assert_close(old, new)
        for values in metrics.values():
            if isinstance(values, list):
                self.assertTrue(all(np.isfinite(value) for value in values))

    def test_current_policy_log_probs_can_score_only_planner_rows(self):
        data = self._create_test_data_for_update_policy()
        planner_rows = np.asarray([True, False, True, False])
        self.mock_model.eval()

        def cpu_logprobs(logits, labels):
            return torch.gather(
                torch.log_softmax(logits.float(), dim=-1),
                dim=-1,
                index=labels.unsqueeze(-1),
            ).squeeze(-1)

        with mock.patch(
            "verl.workers.actor.dp_actor.logprobs_from_logits",
            cpu_logprobs,
        ):
            dense = self.actor._current_minibatch_log_probs(
                data,
                temperature=1.0,
            )
            sparse = self.actor._current_minibatch_log_probs(
                data,
                temperature=1.0,
                row_mask=planner_rows,
            )

        torch.testing.assert_close(sparse[planner_rows], dense[planner_rows])
        self.assertTrue(torch.count_nonzero(sparse[~planner_rows]).item() == 0)

    def test_current_policy_log_probs_empty_row_mask_skips_forward(self):
        data = self._create_test_data_for_update_policy()
        with mock.patch.object(
            self.actor,
            "_forward_micro_batch",
            wraps=self.actor._forward_micro_batch,
        ) as forward:
            sparse = self.actor._current_minibatch_log_probs(
                data,
                temperature=1.0,
                row_mask=np.zeros(len(data), dtype=np.bool_),
            )

        forward.assert_not_called()
        self.assertTrue(torch.count_nonzero(sparse).item() == 0)

    @unittest.skipUnless(torch.cuda.is_available(), "FSDP1 audit smoke requires CUDA")
    def test_fsdp1_output_container_baseline(self):
        class TensorOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(8, 8)

            def forward(self, values):
                return self.layer(values)

        class HuggingFaceOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(8, 8)

            def forward(self, values):
                return CausalLMOutput(logits=self.layer(values))

        for model_class, select_output in (
            (TensorOutputModel, lambda output: output),
            (HuggingFaceOutputModel, lambda output: output.logits),
        ):
            model = FSDP(
                model_class().to(self.device),
                device_id=self.device,
                use_orig_params=False,
            )
            output = model(torch.randn(2, 8, device=self.device))
            select_output(output).sum().backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.parameters())
            )

    @unittest.skipUnless(torch.cuda.is_available(), "FSDP1 audit smoke requires CUDA")
    def test_fsdp1_actor_forward_baseline(self):
        config = FSDPActorConfig(
            strategy="fsdp",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            entropy_coeff=0.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
        )
        model = FSDP(
            MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device),
            device_id=self.device,
            use_orig_params=False,
        )
        actor = DataParallelPPOActor(
            config=config,
            actor_module=model,
            actor_optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        )
        data = self._create_test_data_for_update_policy()[:2].to(self.device)
        model_inputs = {**data.batch, **data.non_tensor_batch}

        _, log_probs = actor._forward_micro_batch(
            model_inputs,
            temperature=1.0,
            calculate_entropy=False,
            preserve_fsdp_output_hook=True,
        )
        self.assertTrue(torch.all(torch.isfinite(log_probs)))
        log_probs.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "FSDP1 audit smoke requires CUDA")
    def test_planner_poc_parameter_gradient_audit_with_fsdp1(self):
        config = FSDPActorConfig(
            strategy="fsdp",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            entropy_coeff=0.0,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
        )
        model = FSDP(
            MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device),
            device_id=self.device,
            use_orig_params=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        actor = DataParallelPPOActor(
            config=config,
            actor_module=model,
            actor_optimizer=optimizer,
        )
        data = self._create_test_data_for_update_policy()
        response_mask = data.batch["response_mask"]
        real = torch.zeros_like(response_mask)
        real[0] = -0.25
        real[1] = 0.25
        shuffled = torch.zeros_like(response_mask)
        shuffled[2] = -0.25
        shuffled[3] = 0.25
        data.batch["workflow_loss_weights"] = response_mask / response_mask.sum()
        data.batch["planner_outcome_contrast_coefficients"] = real
        data.batch["planner_outcome_contrast_shuffled_coefficients"] = shuffled
        data.non_tensor_batch["trajectory_id"] = np.asarray(
            ["w0", "w1", "w2", "w3"], dtype=object
        )
        data.meta_info.update(
            {
                "planner_outcome_contrast_gradient_audit": True,
                "planner_outcome_contrast_loss_weight": 0.1,
            }
        )
        parameters_before = [
            parameter.detach().clone() for parameter in model.parameters()
        ]

        def standard_logprobs(logits, labels):
            return torch.gather(
                torch.log_softmax(logits.float(), dim=-1),
                dim=-1,
                index=labels.unsqueeze(-1),
            ).squeeze(-1)

        with (
            mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
            mock.patch(
                "verl.workers.actor.dp_actor.logprobs_from_logits",
                standard_logprobs,
            ),
        ):
            metrics = actor.update_policy(data)

        step.assert_not_called()
        self.assertEqual(metrics["actor/optimizer_steps_expected"], [0.0])
        self.assertEqual(metrics["actor/optimizer_steps_executed"], [0.0])
        self.assertGreater(
            metrics[
                "aux/planner_outcome_contrast/parameter_em_gradient_norm"
            ][0],
            0.0,
        )
        self.assertGreater(
            metrics[
                "aux/planner_outcome_contrast/parameter_real_poc_gradient_norm"
            ][0],
            0.0,
        )
        for before, after in zip(parameters_before, model.parameters(), strict=True):
            torch.testing.assert_close(before, after)

    def test_dataparallelppoactor_initialization(self):
        """Test DataParallelPPOActor initialization"""
        self.assertIsNotNone(self.actor.actor_module)
        self.assertIsNotNone(self.actor.actor_optimizer)
        self.assertEqual(self.actor.config, self.config)

        self.assertEqual(self.actor.config.strategy, "fsdp2")
        self.assertEqual(self.actor.config.ppo_mini_batch_size, 4)
        self.assertEqual(self.actor.config.clip_ratio, 0.2)

    def test_dataparallelppoactor_with_qwen3_model(self):
        """Test DataParallelPPOActor with real Qwen3ForCausalLM model"""
        qwen_config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            torch_dtype=torch.float32,
            use_cache=False,
        )

        with torch.device(self.device):
            qwen_model = AutoModelForCausalLM.from_config(config=qwen_config, torch_dtype=torch.float32).to(self.device)

        qwen_optimizer = torch.optim.Adam(qwen_model.parameters(), lr=1e-4)

        qwen_actor = DataParallelPPOActor(config=self.config, actor_module=qwen_model, actor_optimizer=qwen_optimizer)

        data = self._create_test_data_for_compute_log_prob()
        log_probs, entropies = qwen_actor.compute_log_prob(data, calculate_entropy=True)

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        self.assertIsInstance(entropies, torch.Tensor)
        self.assertEqual(entropies.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(entropies)))
        self.assertTrue(torch.all(entropies >= 0))

        qwen_actor.response_only_logits = True
        selected_log_probs, selected_entropies = qwen_actor.compute_log_prob(data, calculate_entropy=True)
        torch.testing.assert_close(selected_log_probs, log_probs)
        torch.testing.assert_close(selected_entropies, entropies)

        policy_data = self._create_test_data_for_update_policy()
        metrics = qwen_actor.update_policy(policy_data)

        self.assertIsInstance(metrics, dict)

        expected_metric_keys = [
            "actor/pg_loss",
            "actor/pg_clipfrac",
            "actor/ppo_kl",
            "actor/pg_clipfrac_lower",
            "actor/grad_norm",
        ]

        for key in expected_metric_keys:
            self.assertIn(key, metrics)
            if isinstance(metrics[key], list):
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics[key]))
            else:
                self.assertIsInstance(metrics[key], (float, int))
                self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))


if __name__ == "__main__":
    unittest.main()
