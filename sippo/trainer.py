"""SIPPO trainer: JADE's workflow-RLOO / PSC training step on top of the verl 0.9.1 trainer.

Inherits the 0.9.1 ``RayPPOTrainer`` (worker groups, dataloader, rollout servers, checkpoint manager, checkpointing)
and the verbatim JADE workflow rollout (``WorkflowRolloutMixin``). ``fit`` is the JADE step restricted to the path the
W and P runs execute:

    qa_manager workflow rollouts -> question-preserving DP partition (+ valid-row mask) -> old log-probs
    -> workflow-RLOO advantage -> PSC metadata (P arm) -> SIPPO policy update -> rollout weight sync

Disabled-by-config JADE branches (online OPSD, SOD, CCW, typed-selection credit, node rewards, critic, KL, audits,
linear single-policy mode) are not reproduced; enabling any of them raises instead of being ignored.
"""

from __future__ import annotations

import logging
import os
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.nrpsc_objective import optional_psc_metadata
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.trainer.ppo.workflow_dag_opsd_grpo import (
    build_complete_question_dp_partitions,
    compute_actor_data_parallel_size,
)
from verl.trainer.ppo.workflow_rloo import compute_workflow_rloo_outcome_advantage
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean

from sippo.actor_update import NON_TENSOR_KEYS, TENSOR_KEYS
from sippo.workflow_rollout import WorkflowRolloutMixin, get_config_bool

logger = logging.getLogger(__name__)

try:  # register the workflow advantage under its JADE name (0.9.1 registers by string key)
    core_algos.get_adv_estimator_fn("workflow_rloo")
except Exception:
    core_algos.register_adv_est("workflow_rloo")(compute_workflow_rloo_outcome_advantage)

# Features that are off in the W/P runs. Turning one on must fail loudly instead of silently training a different objective.
_UNSUPPORTED_ON = (
    "algorithm.dag_opsd.enabled",
    "algorithm.workflow_sod.enabled",
    "actor_rollout_ref.actor.online_opsd.enabled",
    "algorithm.typed_selection_credit.enabled",
    "qa_manager.node_reward.enabled",
    "actor_rollout_ref.actor.use_kl_loss",
    "algorithm.use_kl_in_reward",
)


class _RolloutManagerAdapter:
    """The verbatim JADE rollout code wakes/sleeps the rollout engine itself (``wake_up()`` before sampling,
    ``sleep()`` after). In verl 0.9.1 that lifecycle belongs to ``checkpoint_manager``: replicas are awake after
    ``update_weights`` and ``fit`` calls ``sleep_replicas()`` once all rollouts are done. So both calls are no-ops
    here; every other attribute is forwarded unchanged."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def wake_up(self):
        return None

    def sleep(self):
        return None


class SippoRayPPOTrainer(WorkflowRolloutMixin, RayPPOTrainer):
    def init_workers(self):
        super().init_workers()
        if not isinstance(self.async_rollout_manager, _RolloutManagerAdapter):
            self.async_rollout_manager = _RolloutManagerAdapter(self.async_rollout_manager)

    def _compute_old_log_prob(self, batch):
        """Old-policy log-probs from the training engine.

        Same as the verl 0.9.1 method, except that the entropy over the full vocabulary is computed only when
        ``actor_rollout_ref.actor.calculate_old_log_prob_entropy`` is true (the W/P runs set it to false, as the JADE
        trainer did); 0.9.1 hard-codes True. Returns ``(DataProto with old_log_probs[, entropys], mfu)``.
        """
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

        calculate_entropy = get_config_bool(
            self.config, "actor_rollout_ref.actor.calculate_old_log_prob_entropy", default=True
        )
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=calculate_entropy, calculate_sum_pi_squared=False,
                             compute_loss=False)
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
        result = {"old_log_probs": log_probs.float()}
        if calculate_entropy:
            result["entropys"] = no_padding_2_padding(tu.get(output, "entropy"), batch_td).float()
        mfu = tu.get(output, "metrics")["mfu"]
        return DataProto.from_tensordict(tu.get_tensordict(result)), mfu

    # ---- configuration guards ------------------------------------------------------------------------------------
    def _check_supported_config(self) -> None:
        for key in _UNSUPPORTED_ON:
            if get_config_bool(self.config, key, default=False):
                raise NotImplementedError(f"{key}=true is not implemented in the SIPPO W/P trainer")
        if str(self._planner_ccw_mode()) != "off":
            raise NotImplementedError("algorithm.planner_ccw.mode must be off")
        estimator = self.config.algorithm.adv_estimator
        estimator = estimator.value if hasattr(estimator, "value") else str(estimator)
        if estimator != "workflow_rloo":
            raise NotImplementedError("only algorithm.adv_estimator=workflow_rloo is implemented")
        if self.use_critic:
            raise NotImplementedError("workflow_rloo trains without a critic")
        poc = OmegaConf.select(self.config, "algorithm.planner_outcome_contrast", default={}) or {}
        if str(poc.get("minibatch_mode", "workflow_single")) != "question_preserving":
            raise NotImplementedError("algorithm.planner_outcome_contrast.minibatch_mode must be question_preserving")
        aux = OmegaConf.select(self.config, "algorithm.planner_aux_control", default={}) or {}
        aux_mode = str(aux.get("mode", "off"))
        if aux_mode not in {"off", "pointwise_rloo_matched"}:
            raise NotImplementedError(f"algorithm.planner_aux_control.mode={aux_mode!r}")
        poc_mode = str(poc.get("mode", "off")).lower()
        if poc_mode == "nrpsc" and int(poc.get("nrpsc_replay_budget", 0)) != 0:
            raise NotImplementedError("NPU NR-PSC n=8 baseline requires replay budget zero")
        if poc_mode == "nrpsc" and float(poc.get("nrpsc_tiebreak_weight", 0.0)) < 0:
            raise ValueError("algorithm.planner_outcome_contrast.nrpsc_tiebreak_weight must be non-negative")
        if aux_mode != "off" and poc_mode not in {"off", "false", "none"}:
            raise ValueError("R control (planner_aux_control) and PSC (planner_outcome_contrast.mode) are separate arms")
        for audit in ("gradient_audit", "coverage_audit_only", "current_policy_audit_only", "paired_shuffled_audit"):
            if bool(poc.get(audit, False)):
                raise NotImplementedError(f"planner_outcome_contrast.{audit} is not implemented")

    # ---- frozen real batch: capture, and gradient comparison on it ---------------------------------------------------
    def _freeze_and_compare(self, out_dir, tensors, non_tensors, meta):
        """Write the real batch that is about to be used for the policy update and, when SIPPO_GRAD_COMPARE=1, compare the
        base / PSC / NR-PSC gradients on it. Parameters and optimizer state are untouched by the comparison."""
        import json
        import pickle

        os.makedirs(out_dir, exist_ok=True)

        import time

        def trace(message):  # direct file append: survives even if the Ray log forwarder drops the last lines
            with open(os.path.join(out_dir, "trace.txt"), "a") as trace_handle:
                trace_handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")

        trace(f"enter _freeze_and_compare; SIPPO_GRAD_COMPARE={os.environ.get('SIPPO_GRAD_COMPARE')!r}")
        with open(os.path.join(out_dir, "frozen_batch.pkl"), "wb") as handle:
            pickle.dump({"tensors": tensors, "non_tensors": non_tensors, "meta": meta}, handle, protocol=4)
        print(f"[sippo] frozen batch written to {out_dir}/frozen_batch.pkl ({len(tensors['input_ids'])} rows)")
        if os.environ.get("SIPPO_GRAD_COMPARE") == "1":
            weight = float(os.environ.get("SIPPO_PSC_WEIGHT", "0.02"))
            import traceback

            try:
                trace("calling grad_compare_sippo")
                print("[sippo] gradient comparison started", flush=True)
                result = self.actor_rollout_wg.grad_compare_sippo(tensors, non_tensors, meta, weight)
            except BaseException:
                with open(os.path.join(out_dir, "error.txt"), "w") as handle:
                    handle.write(traceback.format_exc())
                raise
            if isinstance(result, (list, tuple)):
                result = result[0]
            trace("grad_compare_sippo returned")
            with open(os.path.join(out_dir, "grad_compare.json"), "w") as handle:
                json.dump(result, handle, indent=1, default=float)
            trace("grad_compare.json written")
        else:
            trace("SKIPPED gradient comparison: SIPPO_GRAD_COMPARE is not 1")
            print(f"[sippo] gradient comparison written to {out_dir}/grad_compare.json")

    # ---- the training loop ---------------------------------------------------------------------------------------
    def fit(self):
        from qa_manager.ROSA import ROSA_AgentManager
        from verl.utils.tracking import Tracking

        self._check_supported_config()
        from sippo.seed import seed_process_from_env

        seed_process_from_env()
        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        if os.environ.get("SIPPO_REPLAY_FROZEN"):  # gradient audit on an already frozen real batch: no rollout, no update
            import pickle

            with open(os.environ["SIPPO_REPLAY_FROZEN"], "rb") as handle:
                frozen = pickle.load(handle)
            os.environ["SIPPO_GRAD_COMPARE"] = "1"
            self.checkpoint_manager.sleep_replicas()  # free the rollout engine's NPU memory first, exactly as fit() does before an update
            self._freeze_and_compare(os.environ["SIPPO_REPLAY_OUT"], frozen["tensors"], frozen["non_tensors"], frozen["meta"])
            return

        test_freq = int(self.config.trainer.get("test_freq", -1))
        if self.config.trainer.get("val_before_train", False):
            val_metrics = self.test_validate()  # verbatim JADE validation (see sippo/workflow_rollout.py)
            pprint(f"Initial validation metrics: {val_metrics}")
            tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        qa_manager = ROSA_AgentManager(self.tokenizer, self.config)
        max_workflow_concurrency = int(
            OmegaConf.select(self.config, "qa_manager.max_workflow_concurrency", default=self.config.data.train_batch_size)
        )
        if max_workflow_concurrency < 1:
            raise ValueError(f"qa_manager.max_workflow_concurrency must be positive, got {max_workflow_concurrency}")
        temperature = float(self.config.actor_rollout_ref.rollout.temperature)
        save_freq = int(self.config.trainer.get("save_freq", -1))
        max_prompt_length = int(self.config.data.max_prompt_length)

        for _epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # 1) workflow rollouts (multi-agent DAG through qa_manager; every role step is one training row)
                batch_list, train_reward_metrics = self.async_rollout_batch(batch_dict, qa_manager, max_workflow_concurrency, True, False)
                metrics.update(train_reward_metrics)
                for dataproto in batch_list:
                    dataproto.meta_info = {}
                original_len = len(batch_list)
                batch_list[:] = [dp for dp in batch_list if dp.batch["prompts"][0].size()[0] == max_prompt_length]
                if original_len - len(batch_list) > 0:
                    print(f"Dropped {original_len - len(batch_list)} dataproto(s) with prompts batch_size != {max_prompt_length}")
                batch = DataProto.concat(batch_list)
                # rollout engines are idle from here on: free their memory for the training step
                self.checkpoint_manager.sleep_replicas()

                # 2) question-preserving data-parallel partition + valid-row mask
                actor_world_size = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
                sequence_parallel_size = int(
                    OmegaConf.select(self.config, "actor_rollout_ref.actor.ulysses_sequence_parallel_size", default=1))
                actor_dp_size = compute_actor_data_parallel_size(actor_world_size, sequence_parallel_size)
                missing = {"uid", "trajectory_id"} - set(batch.non_tensor_batch)
                if missing:
                    raise ValueError(f"question-preserving DP partitioning is missing metadata: {sorted(missing)}")
                original_rows = len(batch)
                partitions, real_row_counts = build_complete_question_dp_partitions(
                    batch.non_tensor_batch["uid"], batch.non_tensor_batch["trajectory_id"], actor_dp_size)
                shards = []
                for indices, real_rows in zip(partitions, real_row_counts, strict=True):
                    shard = batch[indices]
                    valid_rows = torch.zeros(len(shard), dtype=torch.bool, device=shard.batch["response_mask"].device)
                    valid_rows[:real_rows] = True
                    shard.batch["actor_dp_valid_rows"] = valid_rows
                    shards.append(shard)
                batch = DataProto.concat(shards)
                metrics.update({
                    "actor/question_dp_size": float(actor_dp_size),
                    "actor/question_dp_rows_per_rank": float(partitions[0].size),
                    "actor/question_dp_padding_rows": float(len(batch) - original_rows),
                    "actor/question_dp_min_questions_per_rank": float(min(
                        len(set(batch.non_tensor_batch["uid"][r * partitions[0].size:(r + 1) * partitions[0].size].tolist()))
                        for r in range(actor_dp_size))),
                })
                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                batch.meta_info["temperature"] = temperature

                # 3) old-policy log-probs (and entropy metric) from the training engine
                old_log_prob, _old_log_prob_mfu = self._compute_old_log_prob(batch)
                if "entropys" in old_log_prob.batch.keys():
                    entropys = old_log_prob.batch.pop("entropys")
                    metrics["actor/entropy"] = masked_mean(entropys, batch.batch["response_mask"]).detach().item()
                else:
                    metrics["actor/old_log_prob_entropy_skipped"] = 1.0
                batch = batch.union(old_log_prob)

                # 4) rewards -> workflow-RLOO advantage
                batch.batch["token_level_scores"] = batch.batch["reward_tensor"]
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                advantages, returns = compute_workflow_rloo_outcome_advantage(
                    token_level_rewards=batch.batch["token_level_rewards"],
                    response_mask=batch.batch["response_mask"],
                    index=batch.non_tensor_batch["uid"],
                    trajectory_ids=batch.non_tensor_batch["trajectory_id"],
                    config=self.config.algorithm,
                )
                batch.batch["advantages"] = advantages
                batch.batch["returns"] = returns
                metrics.update({"actor/workflow_loss_normalization_enabled": 0.0, "actor/question_preserving_token_mean_loss": 1.0})

                # 5) PSC metadata (P arm) -- verbatim JADE
                batch, poc_metrics = self._apply_planner_outcome_contrast(batch)
                metrics.update(poc_metrics)

                # 6) policy update
                tensors = {k: batch.batch[k].cpu() for k in TENSOR_KEYS}
                non_tensor_keys = list(NON_TENSOR_KEYS)
                for key in optional_psc_metadata(batch.non_tensor_batch):
                    if key not in non_tensor_keys:
                        non_tensor_keys.append(key)
                for key in ("nrpsc_plan_view", "nrpsc_successes", "nrpsc_trials"):
                    if key in batch.non_tensor_batch and key not in non_tensor_keys:
                        non_tensor_keys.append(key)
                non_tensors = {k: np.asarray(batch.non_tensor_batch[k]) for k in non_tensor_keys}
                meta = dict(batch.meta_info)
                meta["temperature"] = temperature
                if os.environ.get("SIPPO_FREEZE_DIR"):
                    self._freeze_and_compare(os.environ["SIPPO_FREEZE_DIR"], tensors, non_tensors, meta)
                    if os.environ.get("SIPPO_FREEZE_ONLY") == "1":
                        print("[sippo] frozen batch written; stopping before the policy update (SIPPO_FREEZE_ONLY=1)")
                        progress_bar.close()
                        return
                aux = OmegaConf.select(self.config, "algorithm.planner_aux_control", default={}) or {}
                if str(aux.get("mode", "off")) != "off":  # R arm settings for SippoPolicyUpdater
                    meta.update({
                        "planner_aux_mode": str(aux.get("mode")),
                        "planner_aux_target_psc_loss_weight": float(aux.get("target_psc_loss_weight", 0.02)),
                        "planner_aux_beta": float(aux.get("beta", 1.0)),
                        "planner_aux_epsilon": float(aux.get("epsilon", 1e-12)),
                        "planner_aux_max_scale": aux.get("max_scale", None),
                        "planner_aux_record_planner_base_grad": bool(aux.get("record_planner_base_grad", True)),
                    })
                actor_out = self.actor_rollout_wg.update_actor_sippo(tensors, non_tensors, meta)
                if isinstance(actor_out, (list, tuple)):  # one result per worker; the SIPPO update is single data-parallel rank
                    actor_out = actor_out[0]
                metrics.update(reduce_metrics(actor_out))

                # 7) push the updated weights to the rollout servers
                self.checkpoint_manager.update_weights(self.global_steps)

                # validation: same position and condition as the JADE trainer (after the update and the weight sync)
                if test_freq > 0 and (is_last_step or self.global_steps % test_freq == 0):
                    val_metrics = self.test_validate()
                    metrics.update(val_metrics)

                extra_save = {int(x) for x in os.environ.get("SIPPO_EXTRA_SAVE_STEPS", "").split(",") if x.strip()}
                if save_freq > 0 and (is_last_step or self.global_steps % save_freq == 0 or self.global_steps in extra_save):
                    # The rollout engines are awake here (update_weights woke them, validation used them). Gathering the
                    # full state dict for hf_model on top of resident params + optimizer state + an awake vLLM does not
                    # fit in HBM, so release vLLM first and re-sync it afterwards (weights are unchanged by saving).
                    self.checkpoint_manager.sleep_replicas()
                    self._save_checkpoint()
                    if not is_last_step:
                        self.checkpoint_manager.update_weights(self.global_steps)
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                tracker.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint("Training finished")
                    progress_bar.close()
                    return
