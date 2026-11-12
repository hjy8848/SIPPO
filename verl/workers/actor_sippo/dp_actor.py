# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import json
import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.planner_outcome_contrast import (
    PLANNER_ROLE,
    build_planner_outcome_contrast_coefficients,
    planner_outcome_contrast_loss,
    summarize_parameter_gradient_audit,
)
from verl.trainer.ppo.workflow_dag_opsd_grpo import (
    build_complete_workflow_minibatch_indices,
    build_question_preserving_minibatch_indices,
    validate_linear_full_trajectory_minibatches,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    FSDPModule,
    fsdp2_clip_grad_norm_,
    load_fsdp_optimizer,
    offload_fsdp_optimizer,
)
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.actor.loss_scaling import compute_microbatch_loss_scale
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.response_only_logits = self.config.get("response_only_logits", False)
        self.trim_padded_sequences = self.config.get("trim_padded_sequences", False)

    @staticmethod
    def _trim_padded_micro_batch(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Remove padding shared by every row while preserving valid causal tokens."""
        full_response_length = responses.size(-1)
        prompt_length = input_ids.size(-1) - full_response_length
        if prompt_length <= 0:
            return input_ids, attention_mask, position_ids, responses, full_response_length

        prompt_columns = attention_mask[:, :prompt_length].bool().any(dim=0)
        active_prompt_columns = torch.nonzero(prompt_columns, as_tuple=False).flatten()
        left = int(active_prompt_columns[0].item()) if active_prompt_columns.numel() else 0

        response_mask = response_mask.bool()
        response_lengths = response_mask.sum(dim=-1)
        expected_prefix_mask = (
            torch.arange(full_response_length, device=response_mask.device).unsqueeze(0)
            < response_lengths.unsqueeze(1)
        )
        if torch.equal(response_mask, expected_prefix_mask):
            local_response_length = max(1, int(response_lengths.max().item()))
        else:
            local_response_length = full_response_length

        right = prompt_length + local_response_length
        if left == 0 and right == input_ids.size(-1):
            return input_ids, attention_mask, position_ids, responses, full_response_length

        input_ids = input_ids[:, left:right]
        attention_mask = attention_mask[:, left:right]
        position_ids = position_ids[..., left:right]
        responses = responses[:, :local_response_length]
        return input_ids, attention_mask, position_ids, responses, local_response_length

    def _prepare_actor_dynamic_batch(self, data: DataProto, max_token_len: int):
        min_num_micro_batch = None
        if self.config.get("force_single_row_dynamic_microbatches", False):
            min_num_micro_batch = len(data)
        return prepare_dynamic_batch(
            data,
            max_token_len=max_token_len,
            min_num_micro_batch=min_num_micro_batch,
        )

    @staticmethod
    def _response_prediction_mask(response_mask: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """Mark sequence positions whose logits predict valid response tokens."""
        response_length = response_mask.shape[-1]
        if response_length >= sequence_length:
            raise ValueError(
                f"response length ({response_length}) must be smaller than sequence length ({sequence_length})"
            )
        response_token_mask = torch.zeros(
            (*response_mask.shape[:-1], sequence_length),
            dtype=torch.bool,
            device=response_mask.device,
        )
        response_token_mask[..., -response_length:] = response_mask.bool()
        prediction_mask = torch.zeros_like(response_token_mask)
        prediction_mask[..., :-1] = response_token_mask[..., 1:]
        return prediction_mask

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        preserve_fsdp_output_hook=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        full_response_length = micro_batch["responses"].size(-1)
        response_length = full_response_length
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            responses = micro_batch["responses"]
            entropy = None
            if self.trim_padded_sequences and not self.use_remove_padding and not multi_modal_inputs:
                response_mask = micro_batch.get(
                    "response_mask", attention_mask[:, -full_response_length:]
                )
                input_ids, attention_mask, position_ids, responses, response_length = (
                    self._trim_padded_micro_batch(
                        input_ids,
                        attention_mask,
                        position_ids,
                        responses,
                        response_mask,
                    )
                )
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                response_mask = micro_batch.get("response_mask", attention_mask[:, -response_length:]).bool()
                response_logit_indices = None
                use_response_only_logits = (
                    self.response_only_logits
                    and not self.use_fused_kernels
                    and not self.use_ulysses_sp
                    and not multi_modal_inputs
                )
                if use_response_only_logits:
                    prediction_mask = self._response_prediction_mask(response_mask, seqlen)
                    prediction_mask_rmpad = index_first_axis(
                        prediction_mask.reshape(-1, 1), indices
                    ).squeeze(-1)
                    response_logit_indices = torch.nonzero(
                        prediction_mask_rmpad, as_tuple=False
                    ).flatten()

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                elif use_response_only_logits:
                    extra_args["logits_to_keep"] = response_logit_indices

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    if preserve_fsdp_output_hook:
                        logits_rmpad = logits_rmpad / temperature
                    else:
                        logits_rmpad.div_(temperature)

                    if use_response_only_logits:
                        labels = input_ids_rmpad_rolled.squeeze(0).index_select(
                            0, response_logit_indices
                        )
                        expected_labels = micro_batch["responses"][response_mask]
                        if not torch.equal(labels, expected_labels):
                            raise RuntimeError("response-only logits are not aligned with response tokens")
                        selected_log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=labels,
                            inplace_backward=not calculate_entropy,
                        )
                        log_probs = torch.zeros_like(
                            micro_batch["responses"], dtype=selected_log_probs.dtype
                        ).masked_scatter(response_mask, selected_log_probs)
                        if calculate_entropy:
                            selected_entropy = self.compute_entropy_from_logits(logits_rmpad)
                            entropy = torch.zeros_like(
                                micro_batch["responses"], dtype=selected_entropy.dtype
                            ).masked_scatter(response_mask, selected_entropy)
                        return entropy, log_probs

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                elif self.response_only_logits and not multi_modal_inputs:
                    extra_args["logits_to_keep"] = response_length + 1

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    if preserve_fsdp_output_hook:
                        logits = logits / temperature
                    else:
                        logits.div_(temperature)
                    if self.response_only_logits and not multi_modal_inputs:
                        if logits.shape[1] != response_length + 1:
                            raise RuntimeError(
                                "response-only projection returned an unexpected sequence length: "
                                f"{logits.shape[1]} != {response_length + 1}"
                            )
                        logits = logits[:, :-1, :]
                    else:
                        logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, responses)
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

        if response_length < full_response_length:
            log_probs = F.pad(log_probs, (0, full_response_length - response_length))
            if entropy is not None:
                entropy = F.pad(entropy, (0, full_response_length - response_length))

        return entropy, log_probs

    def _online_opsd_response_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_length: int,
        valid_response_length: int,
        temperature: float,
    ) -> torch.Tensor:
        """Return full-vocabulary logits aligned to the valid response tokens."""
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            try:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=response_length + 1,
                )
            except TypeError:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
        logits = output.logits
        if logits.shape[1] <= response_length + 1:
            response_logits = logits[:, :response_length]
        else:
            response_logits = logits[:, -(response_length + 1) : -1]
        if response_logits.shape[1] != response_length:
            raise RuntimeError(
                "Online OPSD response/logit alignment failed: "
                f"{response_logits.shape[1]} != {response_length}"
            )
        return response_logits[:, :valid_response_length] / temperature

    def _update_online_opsd(self, data: DataProto, temperature: float) -> dict:
        """Run one synchronized, role-gated OPSD optimizer update."""
        config = self.config.online_opsd or {}
        if not bool(config.get("enabled", False)):
            return {}
        standalone = bool(config.get("standalone", False))
        pair_source = str(config.get("pair_source", "legacy"))
        metric_prefix = (
            "aux/workflow_sod_forward_kl"
            if pair_source == "workflow_sod"
            else "aux/online_opsd"
        )
        interval = int(config.get("update_interval", 4))
        global_steps = int(data.meta_info.get("global_steps", 0))
        if interval < 1:
            raise ValueError("online_opsd.update_interval must be positive")
        if standalone and interval != 1:
            raise ValueError("standalone online_opsd requires update_interval=1")
        if global_steps % interval != 0:
            return {f"{metric_prefix}/optimizer_step": 0.0}
        if not bool(data.meta_info.get("online_opsd_has_pairs", False)):
            return {
                f"{metric_prefix}/optimizer_step": 0.0,
                f"{metric_prefix}/no_contrast_batch": 1.0,
            }

        required = [
            "online_opsd_teacher_input_ids",
            "online_opsd_teacher_attention_mask",
            "online_opsd_teacher_position_ids",
            "online_opsd_token_mask",
            "online_opsd_eligible",
        ]
        missing = [key for key in required if key not in data.batch]
        if missing:
            raise ValueError(f"Online OPSD actor tensors are missing: {missing}")

        eligible_rows = torch.nonzero(
            data.batch["online_opsd_eligible"].detach().cpu().bool(),
            as_tuple=False,
        ).flatten()
        if "online_opsd_role_code" in data.batch and eligible_rows.numel() > 1:
            role_codes = data.batch["online_opsd_role_code"].detach().cpu().long()
            buckets = {
                int(code): eligible_rows[role_codes[eligible_rows] == code].tolist()
                for code in torch.unique(role_codes[eligible_rows], sorted=True).tolist()
            }
            balanced_rows = []
            while any(buckets.values()):
                for code in sorted(buckets):
                    if buckets[code]:
                        balanced_rows.append(buckets[code].pop(0))
            eligible_rows = torch.tensor(balanced_rows, dtype=torch.long)
        max_rows = int(config.get("max_rows_per_rank", 1))
        loss_weight = float(config.get("loss_weight", 0.05))
        min_confidence = float(config.get("min_teacher_confidence", 0.1))
        min_kl = float(config.get("min_forward_kl", 1e-4))
        max_kl = float(config.get("max_forward_kl", 2.0))
        divergence_temperature = float(
            config.get("divergence_temperature", 1.0)
        )
        pointwise_clip = float(config.get("pointwise_kl_clip", 10.0))
        loss_token_start = int(config.get("loss_token_start", 0))
        gate_on_full_response_kl = bool(
            config.get("gate_on_full_response_kl", True)
        )
        exclude_opening_tag_from_loss = bool(
            config.get("exclude_opening_tag_from_loss", False)
        )
        if max_rows < 1:
            raise ValueError("online_opsd.max_rows_per_rank must be positive")
        if loss_weight < 0:
            raise ValueError("online_opsd.loss_weight must be non-negative")
        if standalone and loss_weight <= 0:
            raise ValueError("standalone online_opsd requires a positive loss_weight")
        if not 0 <= min_confidence < 1:
            raise ValueError(
                "online_opsd.min_teacher_confidence must be in [0, 1)"
            )
        if not 0 <= min_kl < max_kl:
            raise ValueError("online_opsd KL gates must satisfy 0 <= min < max")
        if divergence_temperature <= 0:
            raise ValueError(
                "online_opsd.divergence_temperature must be positive"
            )
        if loss_token_start < 0:
            raise ValueError("online_opsd.loss_token_start must be non-negative")

        metrics = {}
        self.actor_optimizer.zero_grad()
        accepted_pair_total = 0.0
        accepted_slot_count = 0
        for slot in range(max_rows):
            real_pair = eligible_rows.numel() > slot
            row_index = int(eligible_rows[slot].item()) if real_pair else 0
            row = data[row_index : row_index + 1].to(get_device_id())
            role_code = int(
                row.batch.get("online_opsd_role_code", torch.tensor([-1]))[0].item()
            )
            response_mask = row.batch["response_mask"][0].bool()
            valid_response_length = int(response_mask.sum().item())
            if valid_response_length < 1:
                raise RuntimeError("Online OPSD selected an empty response")
            response_length = row.batch["responses"].shape[-1]

            with torch.no_grad():
                teacher_logits = self._online_opsd_response_logits(
                    row.batch["online_opsd_teacher_input_ids"],
                    row.batch["online_opsd_teacher_attention_mask"],
                    row.batch["online_opsd_teacher_position_ids"],
                    response_length,
                    valid_response_length,
                    temperature,
                ).detach()
                teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
                teacher_probs = teacher_logp.exp()
                teacher_entropy = -(teacher_probs * teacher_logp).sum(dim=-1)
                teacher_confidence = (
                    1.0
                    - teacher_entropy.mean()
                    / math.log(max(2, teacher_logits.shape[-1]))
                ).clamp(0.0, 1.0)

            student_logits = self._online_opsd_response_logits(
                row.batch["input_ids"],
                row.batch["attention_mask"],
                row.batch["position_ids"],
                response_length,
                valid_response_length,
                temperature,
            )
            student_logp = F.log_softmax(student_logits.float(), dim=-1)
            contribution = teacher_probs * (teacher_logp - student_logp)
            if pointwise_clip > 0:
                contribution = contribution.clamp_max(pointwise_clip)
            token_kl = contribution.sum(dim=-1) * (temperature**2)
            full_selected_mask = row.batch["online_opsd_token_mask"][
                0, :valid_response_length
            ].float()
            token_positions = torch.arange(
                valid_response_length, device=full_selected_mask.device
            )
            if exclude_opening_tag_from_loss:
                selected_mask = row.batch[
                    "online_opsd_opening_tag_excluded_mask"
                ][0, :valid_response_length].float()
                selected_mask = selected_mask * full_selected_mask
            else:
                selected_mask = full_selected_mask * (
                    token_positions >= loss_token_start
                ).float()
            denominator = selected_mask.sum().clamp_min(1.0)
            forward_kl = (token_kl[0] * selected_mask).sum() / denominator
            full_response_forward_kl = (
                token_kl[0] * full_selected_mask
            ).sum() / full_selected_mask.sum().clamp_min(1.0)
            gate_forward_kl = (
                full_response_forward_kl
                if gate_on_full_response_kl
                else forward_kl
            )
            prefix_mask = full_selected_mask * (token_positions < 8).float()
            body_mask = full_selected_mask * (token_positions >= 8).float()

            def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return (values * mask).sum() / mask.sum().clamp_min(1.0)

            format_prefix_forward_kl = masked_mean(token_kl[0], prefix_mask)
            response_body_forward_kl = masked_mean(token_kl[0], body_mask)
            target_ids = row.batch["responses"][0, :valid_response_length]
            teacher_target_logp = teacher_logp[0].gather(
                dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            teacher_top1_match = (
                teacher_logits[0].argmax(dim=-1) == target_ids
            ).float()
            teacher_target_logp_mean = masked_mean(
                teacher_target_logp, full_selected_mask
            )
            teacher_top1_match_rate = masked_mean(
                teacher_top1_match, full_selected_mask
            )

            accepted = (
                real_pair
                and loss_weight > 0
                and bool(selected_mask.sum().item() > 0)
                and float(teacher_confidence.detach().item()) >= min_confidence
                and min_kl <= float(gate_forward_kl.detach().item()) <= max_kl
            )
            confidence_weight = (
                (teacher_confidence.detach() - min_confidence)
                / max(1e-6, 1.0 - min_confidence)
            ).clamp(0.0, 1.0)
            divergence_weight = torch.exp(
                -gate_forward_kl.detach().clamp_min(0.0)
                / divergence_temperature
            )
            gate_weight = (
                confidence_weight * divergence_weight
                if accepted
                else forward_kl.detach().new_zeros(())
            )
            loss = forward_kl * gate_weight * loss_weight

            accepted_count = loss.detach().new_tensor(float(accepted))
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    accepted_count, op=torch.distributed.ReduceOp.SUM
                )
            global_accepted = bool(accepted_count.item() > 0)
            if global_accepted:
                # Rejected ranks backpropagate a graph-connected zero so every
                # FSDP rank participates in the same synchronized update.
                loss.backward()
                accepted_pair_total += float(accepted_count.item())
                accepted_slot_count += 1

            append_to_dict(
                metrics,
                {
                    f"{metric_prefix}/real_pair_on_rank": float(real_pair),
                    f"{metric_prefix}/accepted_pair_on_rank": float(accepted),
                    f"{metric_prefix}/accepted_pair_count": float(
                        accepted_count.item()
                    ),
                    f"{metric_prefix}/selected_role_code": float(role_code),
                    f"{metric_prefix}/forward_kl": float(
                        forward_kl.detach().item()
                    ),
                    f"{metric_prefix}/full_response_forward_kl": float(
                        full_response_forward_kl.detach().item()
                    ),
                    f"{metric_prefix}/gate_forward_kl": float(
                        gate_forward_kl.detach().item()
                    ),
                    f"{metric_prefix}/teacher_confidence": float(
                        teacher_confidence.detach().item()
                    ),
                    f"{metric_prefix}/format_prefix_forward_kl": float(
                        format_prefix_forward_kl.detach().item()
                    ),
                    f"{metric_prefix}/response_body_forward_kl": float(
                        response_body_forward_kl.detach().item()
                    ),
                    f"{metric_prefix}/teacher_target_logp": float(
                        teacher_target_logp_mean.detach().item()
                    ),
                    f"{metric_prefix}/teacher_top1_match_rate": float(
                        teacher_top1_match_rate.detach().item()
                    ),
                    f"{metric_prefix}/selected_token_count": float(
                        selected_mask.sum().item()
                    ),
                    f"{metric_prefix}/full_selected_token_count": float(
                        full_selected_mask.sum().item()
                    ),
                    f"{metric_prefix}/loss_token_start": float(loss_token_start),
                    f"{metric_prefix}/exclude_opening_tag_from_loss": float(
                        exclude_opening_tag_from_loss
                    ),
                    f"{metric_prefix}/excluded_opening_tag_token_count": float(
                        (full_selected_mask - selected_mask).clamp_min(0).sum().item()
                    ),
                    f"{metric_prefix}/gate_on_full_response_kl": float(
                        gate_on_full_response_kl
                    ),
                    f"{metric_prefix}/gate_weight": float(
                        gate_weight.detach().item()
                    ),
                    f"{metric_prefix}/loss": float(loss.detach().item()),
                },
            )

        if accepted_pair_total > 0:
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
            gradient_scale = world_size / accepted_pair_total
            for parameter in self.actor_module.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)
            grad_norm = self._optimizer_step()
        else:
            # Skipping optimizer.step is essential: AdamW weight decay would
            # otherwise move the policy when every pair was rejected.
            gradient_scale = 0.0
            grad_norm = loss.detach().new_zeros(())
        self.actor_optimizer.zero_grad()
        metrics[f"{metric_prefix}/optimizer_step"] = [
            float(accepted_pair_total > 0)
        ]
        metrics[f"{metric_prefix}/optimizer_accepted_pair_count"] = [
            accepted_pair_total
        ]
        metrics[f"{metric_prefix}/accepted_slot_count"] = [
            float(accepted_slot_count)
        ]
        metrics[f"{metric_prefix}/gradient_scale"] = [float(gradient_scale)]
        metrics[f"{metric_prefix}/grad_norm"] = [float(grad_norm.detach().item())]
        return metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            optimizer_offload = bool(self.config.fsdp_config.optimizer_offload)
            if optimizer_offload:
                load_fsdp_optimizer(self.actor_optimizer, get_device_id())
            try:
                self.actor_optimizer.step()
            finally:
                if optimizer_offload:
                    offload_fsdp_optimizer(self.actor_optimizer)
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "response_mask" in data.batch:
            select_keys.append("response_mask")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = self._prepare_actor_dynamic_batch(
                data, max_token_len=max_token_len
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    def _capture_parameter_gradients(
        self,
        parameters: list[nn.Parameter],
        *,
        references: dict[str, list[torch.Tensor | None]] | None = None,
        keep_snapshot: bool,
    ) -> tuple[list[torch.Tensor | None] | None, float, dict[str, float]]:
        references = references or {}
        for name, snapshot in references.items():
            if len(snapshot) != len(parameters):
                raise ValueError(f"POC {name} gradient snapshot has the wrong length")

        snapshot = [] if keep_snapshot else None
        squared_norm = None
        dots = {name: None for name in references}
        chunk_size = 1 << 20
        for index, parameter in enumerate(parameters):
            gradient = parameter.grad
            if isinstance(gradient, DTensor):
                gradient = gradient.to_local()
            if gradient is None:
                if any(reference[index] is not None for reference in references.values()):
                    raise ValueError("POC gradient presence changed between audit passes")
                if snapshot is not None:
                    snapshot.append(None)
                continue

            cpu_gradient = (
                gradient.detach().to(device="cpu", copy=True)
                if snapshot is not None
                else None
            )
            gradient_flat = gradient.detach().reshape(-1)
            if squared_norm is None:
                squared_norm = torch.zeros(
                    (), dtype=torch.float64, device=gradient.device
                )
                dots = {
                    name: torch.zeros(
                        (), dtype=torch.float64, device=gradient.device
                    )
                    for name in references
                }
            for start in range(0, gradient_flat.numel(), chunk_size):
                end = min(start + chunk_size, gradient_flat.numel())
                gradient_chunk = gradient_flat[start:end].float()
                squared_norm.add_(torch.dot(gradient_chunk, gradient_chunk).double())
                for name, reference in references.items():
                    reference_gradient = reference[index]
                    if reference_gradient is None:
                        raise ValueError(
                            "POC gradient presence changed between audit passes"
                        )
                    reference_chunk = reference_gradient.reshape(-1)[start:end].to(
                        device=gradient.device,
                        dtype=torch.float32,
                        non_blocking=False,
                    )
                    dots[name].add_(
                        torch.dot(gradient_chunk, reference_chunk).double()
                    )
            if snapshot is not None:
                snapshot.append(cpu_gradient)
        squared_norm_value = (
            float(squared_norm.item()) if squared_norm is not None else 0.0
        )
        dot_values = {
            name: float(value.item()) if value is not None else 0.0
            for name, value in dots.items()
        }
        return snapshot, squared_norm_value, dot_values

    def _audit_planner_outcome_contrast_parameter_gradients(
        self,
        *,
        micro_batches: list[DataProto],
        minibatch_loss_mask: torch.Tensor,
        temperature: float,
        loss_weight: float,
    ) -> dict[str, list[float]]:
        """Measure EM, real-POC, and shuffled-POC gradients without updating."""

        if self.config.entropy_coeff != 0 or self.config.use_kl_loss:
            raise ValueError("POC gradient audit requires entropy and KL losses disabled")
        if self.config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
            raise ValueError("POC gradient audit requires vanilla policy loss")

        parameters = list(self.actor_module.parameters())
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if self.device_name == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(get_device_id())

        def restore_rng() -> None:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, get_device_id())

        def backward_pass(kind: str) -> float:
            restore_rng()
            self.actor_optimizer.zero_grad()
            total_loss = 0.0
            for microbatch_index, micro_batch in enumerate(micro_batches):
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                _, log_prob = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    preserve_fsdp_output_hook=True,
                )
                if kind == "em":
                    old_log_prob = (
                        model_inputs["old_log_probs"]
                        if self.config.use_rollout_log_probs
                        else log_prob.detach()
                    )
                    loss_mask = model_inputs["workflow_loss_weights"]
                    loss_scale_factor = compute_microbatch_loss_scale(
                        loss_mask,
                        minibatch_loss_mask,
                        "normalized-weighted-sum",
                    )
                    policy_loss_fn = get_policy_loss_fn("vanilla")
                    loss, _, _, _ = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=model_inputs["advantages"],
                        response_mask=loss_mask,
                        loss_agg_mode="normalized-weighted-sum",
                        config=self.config,
                        rollout_is_weights=None,
                    )
                    loss = loss * loss_scale_factor
                else:
                    coefficient_key = (
                        "planner_outcome_contrast_coefficients"
                        if kind == "real"
                        else "planner_outcome_contrast_shuffled_coefficients"
                    )
                    loss = planner_outcome_contrast_loss(
                        log_prob,
                        model_inputs[coefficient_key],
                    )
                try:
                    loss.backward()
                except (RuntimeError, ValueError) as error:
                    raise RuntimeError(
                        f"POC gradient audit {kind} backward failed at microbatch "
                        f"{microbatch_index}/{len(micro_batches)}"
                    ) from error
                total_loss += float(loss.detach().item())
            return total_loss

        em_loss = backward_pass("em")
        em_snapshot, em_squared_norm, _ = self._capture_parameter_gradients(
            parameters, keep_snapshot=True
        )
        self.actor_optimizer.zero_grad()

        real_loss = backward_pass("real")
        real_snapshot, real_squared_norm, real_dots = (
            self._capture_parameter_gradients(
                parameters,
                references={"em": em_snapshot},
                keep_snapshot=True,
            )
        )
        self.actor_optimizer.zero_grad()

        shuffled_loss = backward_pass("shuffled")
        _, shuffled_squared_norm, shuffled_dots = self._capture_parameter_gradients(
            parameters,
            references={"em": em_snapshot, "real": real_snapshot},
            keep_snapshot=False,
        )
        self.actor_optimizer.zero_grad()

        statistics = torch.tensor(
            [
                em_squared_norm,
                real_squared_norm,
                shuffled_squared_norm,
                real_dots["em"],
                shuffled_dots["em"],
                shuffled_dots["real"],
            ],
            dtype=torch.float64,
            device=parameters[0].device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
        summary = summarize_parameter_gradient_audit(
            em_squared_norm=float(statistics[0].item()),
            real_squared_norm=float(statistics[1].item()),
            shuffled_squared_norm=float(statistics[2].item()),
            real_em_dot=float(statistics[3].item()),
            shuffled_em_dot=float(statistics[4].item()),
            real_shuffled_dot=float(statistics[5].item()),
            loss_weight=loss_weight,
        )
        summary.update(
            {
                "aux/planner_outcome_contrast/parameter_gradient_audit": 1.0,
                "aux/planner_outcome_contrast/parameter_em_loss": em_loss,
                "aux/planner_outcome_contrast/parameter_real_poc_loss": real_loss,
                "aux/planner_outcome_contrast/parameter_weighted_real_poc_loss": (
                    real_loss * loss_weight
                ),
                "aux/planner_outcome_contrast/parameter_shuffled_poc_loss": shuffled_loss,
            }
        )
        return {key: [float(value)] for key, value in summary.items()}

    def _current_minibatch_log_probs(
        self,
        mini_batch: DataProto,
        *,
        temperature: float,
        row_mask: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Score one optimizer minibatch under the current actor parameters."""

        selected_rows = None
        scoring_batch = mini_batch
        if row_mask is not None:
            if isinstance(row_mask, torch.Tensor):
                row_mask = row_mask.detach().cpu().numpy()
            row_mask = np.asarray(row_mask, dtype=np.bool_).reshape(-1)
            if row_mask.shape[0] != len(mini_batch):
                raise ValueError("current-policy row_mask must match the minibatch")
            selected_rows = np.flatnonzero(row_mask).tolist()
            if not selected_rows:
                return torch.zeros_like(
                    mini_batch.batch["response_mask"],
                    dtype=torch.float32,
                    device="cpu",
                )
            scoring_batch = mini_batch[selected_rows]

        if self.config.use_dynamic_bsz:
            max_token_len = (
                self.config.ppo_max_token_len_per_gpu
                * self.ulysses_sequence_parallel_size
            )
            micro_batches, batch_idx_list = self._prepare_actor_dynamic_batch(
                scoring_batch, max_token_len=max_token_len
            )
        else:
            micro_batches = scoring_batch.split(
                self.config.ppo_micro_batch_size_per_gpu
            )
            batch_idx_list = None

        current = []
        with torch.no_grad():
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                _, log_probs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                )
                current.append(log_probs.detach())
        output = torch.cat(current, dim=0)
        if batch_idx_list is not None:
            output = restore_dynamic_batch(output, batch_idx_list)
        output = output.cpu()
        if selected_rows is None:
            return output
        sparse_output = torch.zeros_like(
            mini_batch.batch["response_mask"],
            dtype=output.dtype,
            device="cpu",
        )
        sparse_output[selected_rows] = output
        return sparse_output

    def _refresh_planner_outcome_contrast(
        self,
        mini_batch: DataProto,
        *,
        temperature: float,
        mode: str,
        beta: float,
        loss_weight: float,
        shuffle_seed: int,
        require_signal: bool,
        paired_shuffled_audit: bool,
        data_parallel_size: int,
        objective_ablation_variants: list[str] | None = None,
        permutation_count: int = 0,
        permutation_artifact_path: str = "",
    ) -> dict[str, float]:
        """Refresh the exact local POC linearization at current parameters."""

        roles = np.asarray(mini_batch.non_tensor_batch["role"], dtype=object)
        valid_rows = mini_batch.batch["actor_dp_valid_rows"].detach().cpu().numpy().astype(
            np.bool_, copy=False
        )
        planner_score_rows = (roles == PLANNER_ROLE) & valid_rows
        current_log_probs = self._current_minibatch_log_probs(
            mini_batch,
            temperature=temperature,
            row_mask=planner_score_rows,
        )
        common_builder_kwargs = dict(
            response_mask=mini_batch.batch["response_mask"].cpu(),
            old_log_probs=current_log_probs,
            question_ids=mini_batch.non_tensor_batch["uid"],
            trajectory_ids=mini_batch.non_tensor_batch["trajectory_id"],
            roles=mini_batch.non_tensor_batch["role"],
            outcomes=mini_batch.non_tensor_batch["outcome_exact"],
            chosen_eligibility=mini_batch.non_tensor_batch.get("psc_chosen_eligible"),
            workflow_steps=mini_batch.non_tensor_batch["workflow_step"],
            valid_rows=mini_batch.batch["actor_dp_valid_rows"].cpu(),
            beta=beta,
            data_parallel_size=1,
        )
        result = build_planner_outcome_contrast_coefficients(
            **common_builder_kwargs,
            shuffled=mode == "shuffled",
            shuffle_seed=shuffle_seed,
        )

        local_informative = float(
            result.metrics[
                "aux/planner_outcome_contrast/informative_question_count"
            ]
        )
        counts = torch.tensor(
            [
                local_informative,
                result.metrics["aux/planner_outcome_contrast/pair_count"],
            ],
            dtype=torch.float64,
            device=get_device_id(),
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        global_informative = float(counts[0].item())
        global_pairs = float(counts[1].item())
        # FSDP averages rank gradients. This scale reconstructs one equal-weight
        # average over all informative questions represented in this update.
        scale = (
            local_informative * data_parallel_size / global_informative
            if global_informative > 0
            else 0.0
        )
        coefficients = result.coefficients * scale
        mini_batch.batch["planner_outcome_contrast_coefficients"] = coefficients
        valid_token_mask = mini_batch.batch["response_mask"].float() * mini_batch.batch[
            "actor_dp_valid_rows"
        ].unsqueeze(-1).float()
        token_mass = valid_token_mask.sum().clamp_min(1.0)
        em_direct_gradient = (
            -mini_batch.batch["advantages"].float()
            * valid_token_mask
            / token_mass
        )
        planner_rows = torch.as_tensor(
            np.asarray(mini_batch.non_tensor_batch["role"], dtype=object)
            == "PlanningAgent",
            dtype=torch.bool,
        ).unsqueeze(-1)
        planner_em_gradient = em_direct_gradient * planner_rows
        weighted_poc_gradient = coefficients * loss_weight
        planner_em_norm = torch.linalg.vector_norm(planner_em_gradient)
        weighted_poc_norm = torch.linalg.vector_norm(weighted_poc_gradient)
        direct_denominator = planner_em_norm.clamp_min(1e-12)
        cosine_denominator = (planner_em_norm * weighted_poc_norm).clamp_min(1e-12)
        direct_cosine = (
            torch.sum(planner_em_gradient * weighted_poc_gradient)
            / cosine_denominator
        )
        metrics = dict(result.metrics)
        variant_options = {
            "real": {},
            "shuffled": {"shuffled": True, "shuffle_seed": shuffle_seed},
            "cross_question": {
                "pairing_mode": "cross_question",
                "question_weighting": "pair",
            },
            "no_length_normalization": {"length_normalization": False},
            "pair_weighted": {"question_weighting": "pair"},
            "one_random_pair": {"pair_selection": "one_random"},
        }
        for variant_name in objective_ablation_variants or []:
            if variant_name not in variant_options:
                raise ValueError(
                    f"Unknown planner outcome contrast objective ablation: {variant_name}"
                )
            variant_result = build_planner_outcome_contrast_coefficients(
                **common_builder_kwargs,
                **variant_options[variant_name],
            )
            variant_gradient = variant_result.coefficients * scale * loss_weight
            variant_norm = torch.linalg.vector_norm(variant_gradient)
            variant_denominator = (planner_em_norm * variant_norm).clamp_min(1e-12)
            variant_cosine = (
                torch.sum(planner_em_gradient * variant_gradient)
                / variant_denominator
            )
            metric_prefix = f"aux/planner_outcome_contrast/ablation/{variant_name}/"
            metrics.update(
                {
                    metric_prefix + "pair_count": float(
                        variant_result.metrics["aux/planner_outcome_contrast/pair_count"]
                    ),
                    metric_prefix + "informative_group_count": float(
                        variant_result.metrics[
                            "aux/planner_outcome_contrast/informative_question_count"
                        ]
                    ),
                    metric_prefix + "gradient_norm": float(variant_norm.item()),
                    metric_prefix + "planner_em_norm_ratio": float(
                        (variant_norm / planner_em_norm.clamp_min(1e-12)).item()
                    ),
                    metric_prefix + "planner_em_cosine": float(
                        variant_cosine.item()
                        if planner_em_norm.item() > 0 and variant_norm.item() > 0
                        else 0.0
                    ),
                }
            )
        shuffled_result = None
        weighted_shuffled_gradient = None
        if paired_shuffled_audit and mode == "real":
            shuffled_result = build_planner_outcome_contrast_coefficients(
                **common_builder_kwargs,
                shuffled=True,
                shuffle_seed=shuffle_seed,
            )
            weighted_shuffled_gradient = (
                shuffled_result.coefficients * scale * loss_weight
            )
        metrics.update(
            {
                "aux/planner_outcome_contrast/refreshed": 1.0,
                "aux/planner_outcome_contrast/global_informative_question_count": global_informative,
                "aux/planner_outcome_contrast/global_pair_count": global_pairs,
                "aux/planner_outcome_contrast/current_policy_scored_rows": float(
                    planner_score_rows.sum()
                ),
                "aux/planner_outcome_contrast/local_coefficient_scale": scale,
                "aux/planner_outcome_contrast/empty_refreshed_minibatch": float(
                    global_pairs <= 0
                ),
                "aux/planner_outcome_contrast/global_signal_required": float(
                    require_signal
                ),
                "aux/planner_outcome_contrast/refreshed_planner_em_direct_gradient_norm": float(
                    planner_em_norm.item()
                ),
                "aux/planner_outcome_contrast/refreshed_weighted_poc_direct_gradient_norm": float(
                    weighted_poc_norm.item()
                ),
                "aux/planner_outcome_contrast/refreshed_weighted_poc_to_planner_em_direct_gradient_ratio": float(
                    (weighted_poc_norm / direct_denominator).item()
                ),
                "aux/planner_outcome_contrast/refreshed_weighted_poc_planner_em_direct_gradient_cosine": float(
                    direct_cosine.item()
                    if planner_em_norm.item() > 0 and weighted_poc_norm.item() > 0
                    else 0.0
                ),
            }
        )
        if shuffled_result is not None and weighted_shuffled_gradient is not None:
            shuffled_norm = torch.linalg.vector_norm(weighted_shuffled_gradient)
            shuffled_em_denominator = (
                planner_em_norm * shuffled_norm
            ).clamp_min(1e-12)
            real_shuffled_denominator = (
                weighted_poc_norm * shuffled_norm
            ).clamp_min(1e-12)
            metrics.update(
                {
                    "aux/planner_outcome_contrast/paired_shuffled_audit": 1.0,
                    "aux/planner_outcome_contrast/paired_shuffled_label_count": shuffled_result.metrics[
                        "aux/planner_outcome_contrast/shuffled_label_count"
                    ],
                    "aux/planner_outcome_contrast/paired_shuffled_loss_at_current_policy": shuffled_result.metrics[
                        "aux/planner_outcome_contrast/loss_at_old_policy"
                    ],
                    "aux/planner_outcome_contrast/paired_shuffled_weighted_direct_gradient_norm": float(
                        shuffled_norm.item()
                    ),
                    "aux/planner_outcome_contrast/paired_shuffled_planner_em_direct_gradient_cosine": float(
                        (
                            torch.sum(
                                planner_em_gradient * weighted_shuffled_gradient
                            )
                            / shuffled_em_denominator
                        ).item()
                        if planner_em_norm.item() > 0 and shuffled_norm.item() > 0
                        else 0.0
                    ),
                    "aux/planner_outcome_contrast/paired_real_shuffled_direct_gradient_cosine": float(
                        (
                            torch.sum(
                                weighted_poc_gradient * weighted_shuffled_gradient
                            )
                            / real_shuffled_denominator
                        ).item()
                        if weighted_poc_norm.item() > 0 and shuffled_norm.item() > 0
                        else 0.0
                    ),
                }
            )
        if permutation_count > 0 and mode == "real":
            permutation_cosines: list[float] = []
            for permutation_index in range(permutation_count):
                permutation_result = build_planner_outcome_contrast_coefficients(
                    **common_builder_kwargs,
                    shuffled=True,
                    shuffle_seed=shuffle_seed + permutation_index + 1,
                )
                permutation_gradient = (
                    permutation_result.coefficients * scale * loss_weight
                )
                permutation_norm = torch.linalg.vector_norm(permutation_gradient)
                permutation_denominator = (
                    planner_em_norm * permutation_norm
                ).clamp_min(1e-12)
                permutation_cosine = (
                    torch.sum(planner_em_gradient * permutation_gradient)
                    / permutation_denominator
                )
                permutation_cosines.append(
                    float(
                        permutation_cosine.item()
                        if planner_em_norm.item() > 0 and permutation_norm.item() > 0
                        else 0.0
                    )
                )
            permutation_array = np.asarray(permutation_cosines, dtype=np.float64)
            real_cosine = float(direct_cosine.item())
            exceedances = int(np.count_nonzero(permutation_array >= real_cosine))
            permutation_p_value = (1.0 + exceedances) / (permutation_count + 1.0)
            metrics.update(
                {
                    "aux/planner_outcome_contrast/permutation_count": float(
                        permutation_count
                    ),
                    "aux/planner_outcome_contrast/permutation_mean_cosine": float(
                        permutation_array.mean()
                    ),
                    "aux/planner_outcome_contrast/permutation_std_cosine": float(
                        permutation_array.std()
                    ),
                    "aux/planner_outcome_contrast/permutation_p_value": float(
                        permutation_p_value
                    ),
                    "aux/planner_outcome_contrast/permutation_q10_cosine": float(
                        np.quantile(permutation_array, 0.10)
                    ),
                    "aux/planner_outcome_contrast/permutation_q50_cosine": float(
                        np.quantile(permutation_array, 0.50)
                    ),
                    "aux/planner_outcome_contrast/permutation_q90_cosine": float(
                        np.quantile(permutation_array, 0.90)
                    ),
                    "aux/planner_outcome_contrast/permutation_exceedances": float(
                        exceedances
                    ),
                }
            )
            if permutation_artifact_path and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                artifact = {
                    "mode": "frozen_current_policy_permutation_null",
                    "permutation_count": permutation_count,
                    "shuffle_seed": shuffle_seed,
                    "real_planner_em_cosine": real_cosine,
                    "null_cosines": permutation_cosines,
                    "exceedances_ge_real": exceedances,
                    "p_value": permutation_p_value,
                    "workflows_in_minibatch": int(
                        mini_batch.batch["response_mask"].shape[0]
                    ),
                    "optimizer_steps_expected": 0,
                    "optimizer_steps_executed": 0,
                }
                artifact_parent = os.path.dirname(permutation_artifact_path)
                if artifact_parent:
                    os.makedirs(artifact_parent, exist_ok=True)
                with open(permutation_artifact_path, "w", encoding="utf-8") as handle:
                    json.dump(artifact, handle, indent=2)
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        online_opsd_data = None
        online_opsd_config = self.config.online_opsd or {}
        online_opsd_enabled = bool(online_opsd_config.get("enabled", False))
        online_opsd_standalone = bool(online_opsd_config.get("standalone", False))
        if online_opsd_enabled:
            online_keys = [
                "responses",
                "response_mask",
                "input_ids",
                "attention_mask",
                "position_ids",
                "online_opsd_teacher_input_ids",
                "online_opsd_teacher_attention_mask",
                "online_opsd_teacher_position_ids",
                "online_opsd_token_mask",
                "online_opsd_eligible",
            ]
            if "online_opsd_role_code" in data.batch:
                online_keys.append("online_opsd_role_code")
            if bool(online_opsd_config.get("exclude_opening_tag_from_loss", False)):
                online_keys.append("online_opsd_opening_tag_excluded_mask")
            if all(key in data.batch for key in online_keys):
                online_opsd_data = data.select(batch_keys=online_keys)

        if online_opsd_standalone:
            if online_opsd_data is None:
                raise ValueError(
                    "standalone online_opsd is missing its teacher-pair tensors"
                )
            metrics = self._update_online_opsd(
                online_opsd_data, temperature=temperature
            )
            metric_prefix = (
                "aux/workflow_sod_forward_kl"
                if str(online_opsd_config.get("pair_source", "legacy"))
                == "workflow_sod"
                else "aux/online_opsd"
            )
            optimizer_steps = metrics.get(f"{metric_prefix}/optimizer_step", [])
            if not isinstance(optimizer_steps, (list, tuple)):
                optimizer_steps = [optimizer_steps]
            metrics["actor/ppo_optimizer_step"] = 0.0
            metrics["actor/standalone_forward_kl"] = 1.0
            metrics["actor/optimizer_steps_expected"] = float(
                sum(float(value) for value in optimizer_steps)
            )
            self.actor_optimizer.zero_grad()
            return metrics

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        planner_outcome_contrast_minibatch_mode = str(
            data.meta_info.get(
                "planner_outcome_contrast_minibatch_mode", "workflow_single"
            )
        )
        question_preserving_minibatches = (
            planner_outcome_contrast_minibatch_mode == "question_preserving"
        )
        planner_outcome_contrast_mode = str(
            data.meta_info.get("planner_outcome_contrast_mode", "off")
        )
        legacy_planner_outcome_contrast = (
            "planner_outcome_contrast_coefficients" in data.batch
        )
        planner_outcome_contrast_enabled = legacy_planner_outcome_contrast or (
            question_preserving_minibatches
            and planner_outcome_contrast_mode in {"real", "shuffled"}
        )
        planner_outcome_contrast_gradient_audit = bool(
            data.meta_info.get("planner_outcome_contrast_gradient_audit", False)
        )
        if legacy_planner_outcome_contrast:
            select_keys.append("planner_outcome_contrast_coefficients")
        if planner_outcome_contrast_gradient_audit:
            if "planner_outcome_contrast_shuffled_coefficients" not in data.batch:
                raise ValueError("POC gradient audit is missing shuffled coefficients")
            select_keys.append("planner_outcome_contrast_shuffled_coefficients")
        workflow_normalized_loss = "workflow_loss_weights" in data.batch
        if workflow_normalized_loss:
            select_keys.append("workflow_loss_weights")
        if question_preserving_minibatches:
            if "actor_dp_valid_rows" not in data.batch:
                raise ValueError(
                    "question-preserving actor minibatches require actor_dp_valid_rows"
                )
            select_keys.append("actor_dp_valid_rows")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        linear_full_trajectory = bool(data.meta_info.get("linear_full_trajectory", False))
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if linear_full_trajectory or workflow_normalized_loss:
            non_tensor_select_keys.append("trajectory_id")
        if question_preserving_minibatches:
            for key in ("uid", "trajectory_id", "role", "outcome_exact", "workflow_step"):
                if key not in data.non_tensor_batch:
                    raise ValueError(
                        f"question-preserving actor minibatches require {key} metadata"
                    )
                if key not in non_tensor_select_keys:
                    non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if linear_full_trajectory:
            if workflow_normalized_loss:
                raise ValueError(
                    "linear full trajectories must use row-level minibatches, not workflow loss weights"
                )
            if not self.config.use_rollout_log_probs:
                raise ValueError(
                    "linear full trajectories require rollout log-probabilities as PPO old_log_probs"
                )
            linear_minibatch_count = validate_linear_full_trajectory_minibatches(
                data.non_tensor_batch["trajectory_id"],
                self.config.ppo_mini_batch_size,
            )
        workflow_minibatch_count = 0
        question_minibatch_count = 0
        if question_preserving_minibatches:
            if workflow_normalized_loss:
                raise ValueError(
                    "question-preserving token-mean minibatches cannot use workflow loss weights"
                )
            question_indices = build_question_preserving_minibatch_indices(
                data.non_tensor_batch["uid"],
                data.non_tensor_batch["trajectory_id"],
                self.config.ppo_mini_batch_size,
            )
            mini_batches = [data[indices] for indices in question_indices]
            question_minibatch_count = len(mini_batches)
        elif workflow_normalized_loss:
            workflow_indices = build_complete_workflow_minibatch_indices(
                data.non_tensor_batch["trajectory_id"],
                self.config.ppo_mini_batch_size,
            )
            total_workflows = len(set(data.non_tensor_batch["trajectory_id"].tolist()))
            mini_batches = []
            for indices in workflow_indices:
                mini_batch = data[indices]
                minibatch_workflows = len(
                    set(mini_batch.non_tensor_batch["trajectory_id"].tolist())
                )
                # Preserve equal-workflow normalization while giving each optimizer
                # update the same total loss mass as the former full-batch update.
                mini_batch.batch["workflow_loss_weights"] = (
                    mini_batch.batch["workflow_loss_weights"]
                    * (total_workflows / minibatch_workflows)
                )
                mini_batches.append(mini_batch)
            workflow_minibatch_count = len(mini_batches)
        else:
            mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        if legacy_planner_outcome_contrast:
            if not workflow_normalized_loss:
                raise ValueError(
                    "Planner outcome contrast requires workflow-normalized loss"
                )
            if not on_policy:
                raise ValueError(
                    "Planner outcome contrast requires one on-policy minibatch and one PPO epoch"
                )
            if online_opsd_data is not None:
                raise ValueError(
                    "Planner outcome contrast cannot be combined with online OPSD"
                )
        if question_preserving_minibatches:
            if self.config.ppo_epochs != 1:
                raise ValueError(
                    "question-preserving actor minibatches require ppo_epochs=1"
                )
            if self.config.loss_agg_mode != "token-mean":
                raise ValueError(
                    "question-preserving actor minibatches require token-mean loss"
                )
            if online_opsd_data is not None:
                raise ValueError(
                    "question-preserving actor minibatches cannot be combined with online OPSD"
                )
        planner_outcome_contrast_weight = float(
            data.meta_info.get("planner_outcome_contrast_loss_weight", 0.0)
        )

        metrics = {}
        optimizer_steps_expected = len(mini_batches) * self.config.ppo_epochs
        optimizer_steps_executed = 0
        if workflow_normalized_loss:
            metrics["actor/workflow_complete_workflows_per_minibatch"] = [
                float(self.config.ppo_mini_batch_size)
            ]
            metrics["actor/workflow_minibatch_count"] = [
                float(workflow_minibatch_count)
            ]
        if question_preserving_minibatches:
            metrics["actor/question_preserving_minibatch_count"] = [
                float(question_minibatch_count)
            ]
            metrics["actor/question_preserving_target_action_rows"] = [
                float(self.config.ppo_mini_batch_size)
            ]
            metrics["actor/question_preserving_min_action_rows"] = [
                float(min(len(item) for item in mini_batches))
            ]
            metrics["actor/question_preserving_max_action_rows"] = [
                float(max(len(item) for item in mini_batches))
            ]
        if linear_full_trajectory:
            metrics["actor/linear_complete_workflows_per_minibatch"] = [
                float(self.config.ppo_mini_batch_size)
            ]
            metrics["actor/linear_minibatch_count"] = [
                float(linear_minibatch_count)
            ]
        if planner_outcome_contrast_gradient_audit:
            if len(mini_batches) != 1:
                raise ValueError("POC gradient audit requires exactly one minibatch")
            mini_batch = mini_batches[0]
            minibatch_loss_mask = mini_batch.batch["workflow_loss_weights"]
            if self.config.use_dynamic_bsz:
                max_token_len = (
                    self.config.ppo_max_token_len_per_gpu
                    * self.ulysses_sequence_parallel_size
                )
                micro_batches, _ = self._prepare_actor_dynamic_batch(
                    mini_batch, max_token_len=max_token_len
                )
            else:
                micro_batches = mini_batch.split(
                    self.config.ppo_micro_batch_size_per_gpu
                )
            audit_metrics = self._audit_planner_outcome_contrast_parameter_gradients(
                micro_batches=micro_batches,
                minibatch_loss_mask=minibatch_loss_mask,
                temperature=temperature,
                loss_weight=planner_outcome_contrast_weight,
            )
            metrics.update(audit_metrics)
            metrics["actor/optimizer_steps_expected"] = [0.0]
            metrics["actor/optimizer_steps_executed"] = [0.0]
            metrics["actor/ppo_optimizer_step"] = [0.0]
            self.actor_optimizer.zero_grad()
            return metrics
        planner_outcome_contrast_beta = float(
            data.meta_info.get("planner_outcome_contrast_beta", 1.0)
        )
        planner_outcome_contrast_shuffle_seed = int(
            data.meta_info.get("planner_outcome_contrast_shuffle_seed", 0)
        )
        planner_outcome_contrast_require_signal = bool(
            data.meta_info.get("planner_outcome_contrast_require_signal", True)
        )
        planner_outcome_contrast_data_parallel_size = int(
            data.meta_info.get("planner_outcome_contrast_data_parallel_size", 1)
        )
        planner_outcome_contrast_paired_shuffled_audit = bool(
            data.meta_info.get(
                "planner_outcome_contrast_paired_shuffled_audit", False
            )
        )
        planner_outcome_contrast_current_policy_audit_only = bool(
            data.meta_info.get(
                "planner_outcome_contrast_current_policy_audit_only", False
            )
        )
        planner_outcome_contrast_objective_ablation_variants = list(
            data.meta_info.get(
                "planner_outcome_contrast_objective_ablation_variants", []
            )
        )
        planner_outcome_contrast_permutation_count = int(
            data.meta_info.get("planner_outcome_contrast_permutation_count", 0)
        )
        planner_outcome_contrast_permutation_artifact_path = str(
            data.meta_info.get("planner_outcome_contrast_permutation_artifact_path", "")
        )
        if planner_outcome_contrast_current_policy_audit_only:
            if not question_preserving_minibatches:
                raise ValueError(
                    "Current-policy POC audit requires question-preserving minibatches"
                )
            if (
                planner_outcome_contrast_mode != "real"
                or not planner_outcome_contrast_paired_shuffled_audit
            ):
                raise ValueError(
                    "Current-policy POC audit requires real and shuffled paired coefficients"
                )
            for mini_batch in mini_batches:
                refreshed_metrics = self._refresh_planner_outcome_contrast(
                    mini_batch,
                    temperature=temperature,
                    mode=planner_outcome_contrast_mode,
                    beta=planner_outcome_contrast_beta,
                    loss_weight=planner_outcome_contrast_weight,
                    shuffle_seed=planner_outcome_contrast_shuffle_seed,
                    require_signal=planner_outcome_contrast_require_signal,
                    paired_shuffled_audit=True,
                    data_parallel_size=planner_outcome_contrast_data_parallel_size,
                    objective_ablation_variants=planner_outcome_contrast_objective_ablation_variants,
                    permutation_count=planner_outcome_contrast_permutation_count,
                    permutation_artifact_path=planner_outcome_contrast_permutation_artifact_path,
                )
                append_to_dict(metrics, refreshed_metrics)
            metrics["actor/optimizer_steps_expected"] = [0.0]
            metrics["actor/optimizer_steps_executed"] = [0.0]
            metrics["actor/ppo_optimizer_step"] = [0.0]
            metrics[
                "aux/planner_outcome_contrast/current_policy_audit_only"
            ] = [1.0]
            self.actor_optimizer.zero_grad()
            return metrics
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if question_preserving_minibatches and planner_outcome_contrast_enabled:
                    refreshed_metrics = self._refresh_planner_outcome_contrast(
                        mini_batch,
                        temperature=temperature,
                        mode=planner_outcome_contrast_mode,
                        beta=planner_outcome_contrast_beta,
                        loss_weight=planner_outcome_contrast_weight,
                        shuffle_seed=planner_outcome_contrast_shuffle_seed,
                        require_signal=planner_outcome_contrast_require_signal,
                        paired_shuffled_audit=planner_outcome_contrast_paired_shuffled_audit,
                        data_parallel_size=planner_outcome_contrast_data_parallel_size,
                        objective_ablation_variants=planner_outcome_contrast_objective_ablation_variants,
                        permutation_count=planner_outcome_contrast_permutation_count,
                        permutation_artifact_path=planner_outcome_contrast_permutation_artifact_path,
                    )
                    append_to_dict(metrics, refreshed_metrics)
                minibatch_loss_mask = mini_batch.batch.get(
                    "workflow_loss_weights", mini_batch.batch["response_mask"]
                )
                if question_preserving_minibatches:
                    minibatch_loss_mask = minibatch_loss_mask * mini_batch.batch[
                        "actor_dp_valid_rows"
                    ].unsqueeze(-1)
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = self._prepare_actor_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                    microbatch_row_counts = [len(item) for item in micro_batches]
                    append_to_dict(
                        metrics,
                        {
                            "actor/dynamic_microbatch_count": float(len(micro_batches)),
                            "actor/dynamic_microbatch_max_rows": float(
                                max(microbatch_row_counts)
                            ),
                            "actor/forced_single_row_dynamic_microbatches": float(
                                self.config.get(
                                    "force_single_row_dynamic_microbatches", False
                                )
                            ),
                        },
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    loss_mask = model_inputs.get("workflow_loss_weights", response_mask)
                    if question_preserving_minibatches:
                        loss_mask = loss_mask * model_inputs[
                            "actor_dp_valid_rows"
                        ].unsqueeze(-1)
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    if workflow_normalized_loss:
                        loss_agg_mode = "normalized-weighted-sum"

                    loss_scale_factor = compute_microbatch_loss_scale(
                        loss_mask,
                        minibatch_loss_mask,
                        loss_agg_mode,
                    )

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout importance sampling weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (all functions return 4 values)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=loss_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if planner_outcome_contrast_enabled:
                        contrast_loss = planner_outcome_contrast_loss(
                            log_prob,
                            model_inputs["planner_outcome_contrast_coefficients"],
                        )
                        weighted_contrast_loss = (
                            contrast_loss * planner_outcome_contrast_weight
                        )
                        loss = loss + weighted_contrast_loss
                        micro_batch_metrics[
                            "aux/planner_outcome_contrast/loss"
                        ] = contrast_loss.detach().item()
                        micro_batch_metrics[
                            "aux/planner_outcome_contrast/weighted_loss"
                        ] = weighted_contrast_loss.detach().item()
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                optimizer_steps_executed += int(torch.isfinite(grad_norm).item())
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        if online_opsd_data is not None:
            online_opsd_metrics = self._update_online_opsd(
                online_opsd_data, temperature=temperature
            )
            for key, values in online_opsd_metrics.items():
                if isinstance(values, list):
                    metrics.setdefault(key, []).extend(values)
                else:
                    append_to_dict(metrics, {key: values})
        metrics["actor/optimizer_steps_expected"] = [float(optimizer_steps_expected)]
        metrics["actor/optimizer_steps_executed"] = [float(optimizer_steps_executed)]
        metrics["actor/ppo_optimizer_step"] = [float(optimizer_steps_executed > 0)]
        self.actor_optimizer.zero_grad()
        return metrics
