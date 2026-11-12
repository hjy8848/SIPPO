"""SIPPO policy update for the NPU-native tree (workflow-RLOO "W" arm and workflow-RLOO + PSC "P" arm).

This is a from-scratch re-statement of ``JADE DataParallelPPOActor.update_policy`` restricted to the one
configuration the W and P runs actually use:

    minibatch_mode = question_preserving     (complete sibling-question groups per optimizer step)
    loss_agg_mode  = token-mean              (micro-batch loss scaled to reconstruct the full-minibatch mean)
    ppo_epochs     = 1, no KL loss, no entropy bonus, vanilla PPO clip (dual-clip c)
    P arm only     : + ``loss_weight * sum(log_prob * coefficient)`` with coefficients refreshed from the current policy

Everything that is switched off in those runs (online OPSD, SOD, CCW, node reward, gradient audits, linear full
trajectories, workflow-weighted loss, Ulysses, fused kernels, multimodal) is deliberately NOT implemented and is
rejected loudly if requested, so a silently different objective can never be trained.

The PSC coefficient builder and the question-preserving mini-batch packer are the original JADE functions, copied
verbatim (see PORT_COPIED_FROM_JADE.md5).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.planner_outcome_contrast import (
    PLANNER_ROLE,
    build_planner_outcome_contrast_coefficients,
    planner_outcome_contrast_loss,
)
from verl.trainer.ppo.nrpsc_objective import (
    FrozenReliabilityTable, FrozenTfidf, PlanObservation,
    actual_reward_eligibility, batch_node_coefficients, optional_psc_metadata,
)
from verl.trainer.ppo.workflow_dag_opsd_grpo import build_question_preserving_minibatch_indices
from sippo.r_control import build_planner_pointwise_rloo_coefficients
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.seqlen_balancing import rearrange_micro_batches, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits, masked_mean

TENSOR_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_mask",
    "old_log_probs",
    "advantages",
    "actor_dp_valid_rows",
)
NON_TENSOR_KEYS = ("uid", "trajectory_id", "role", "outcome_exact", "workflow_step")


@dataclass
class SippoActorConfig:
    ppo_mini_batch_size: int = 128
    ppo_max_token_len_per_gpu: int = 8192
    ppo_micro_batch_size_per_gpu: int = 4  # only used when use_dynamic_bsz is False
    use_dynamic_bsz: bool = True
    force_single_row_dynamic_microbatches: bool = False
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_c: float = 3.0
    grad_clip: float = 1.0
    loss_agg_mode: str = "token-mean"
    ppo_epochs: int = 1
    use_remove_padding: bool = True
    response_only_logits: bool = True
    use_rollout_log_probs: bool = False
    entropy_coeff: float = 0.0
    use_kl_loss: bool = False
    autocast_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_actor_config(cls, cfg: dict[str, Any]) -> "SippoActorConfig":
        """Build from a JADE/verl ``actor`` config dict, rejecting settings this implementation does not cover."""
        get = cfg.get
        clip = get("clip_ratio", 0.2)
        out = cls(
            ppo_mini_batch_size=int(get("ppo_mini_batch_size")),
            ppo_max_token_len_per_gpu=int(get("ppo_max_token_len_per_gpu", 8192)),
            ppo_micro_batch_size_per_gpu=int(get("ppo_micro_batch_size_per_gpu") or 1),
            use_dynamic_bsz=bool(get("use_dynamic_bsz", True)),
            force_single_row_dynamic_microbatches=bool(get("force_single_row_dynamic_microbatches", False)),
            clip_ratio_low=float(get("clip_ratio_low") if get("clip_ratio_low") is not None else clip),
            clip_ratio_high=float(get("clip_ratio_high") if get("clip_ratio_high") is not None else clip),
            clip_ratio_c=float(get("clip_ratio_c", 3.0)),
            grad_clip=float(get("grad_clip", 1.0)),
            loss_agg_mode=str(get("loss_agg_mode", "token-mean")),
            ppo_epochs=int(get("ppo_epochs", 1)),
            use_remove_padding=bool(get("use_remove_padding", True)),
            response_only_logits=bool(get("response_only_logits", False)),
            use_rollout_log_probs=bool(get("use_rollout_log_probs", False)),
            entropy_coeff=float(get("entropy_coeff", 0.0)),
            use_kl_loss=bool(get("use_kl_loss", False)),
        )
        out.validate_supported(cfg)
        return out

    def validate_supported(self, raw: Optional[dict[str, Any]] = None) -> None:
        if self.loss_agg_mode != "token-mean":
            raise NotImplementedError("SIPPO W/P path uses loss_agg_mode='token-mean'")
        if self.ppo_epochs != 1:
            raise NotImplementedError("SIPPO W/P path uses ppo_epochs=1 (question-preserving mini-batches)")
        if self.use_kl_loss or self.entropy_coeff != 0:
            raise NotImplementedError("KL loss / entropy bonus are off in the W/P runs and not implemented")
        if raw is not None:
            if int(raw.get("ulysses_sequence_parallel_size", 1) or 1) != 1:
                raise NotImplementedError("Ulysses sequence parallelism is not used by the W/P runs")
            opsd = raw.get("online_opsd") or {}
            if isinstance(opsd, dict) and opsd.get("enabled", False):
                raise NotImplementedError("online_opsd is off in the W/P runs and not implemented")
            if str((raw.get("policy_loss") or {}).get("loss_mode", "vanilla")) != "vanilla":
                raise NotImplementedError("only the vanilla PPO policy loss is implemented")
            if raw.get("use_fused_kernels", False):
                raise NotImplementedError("fused kernels are not used by the W/P runs")


def _append(metrics: dict[str, list], key: str, value) -> None:
    metrics.setdefault(key, []).append(value)


def _append_dict(metrics: dict[str, list], new: dict[str, Any]) -> None:
    for k, v in new.items():
        _append(metrics, k, v)


def _global_grad_norm(model: torch.nn.Module) -> float:
    """L2 norm over the gradients of all trainable parameters (FSDP2 DTensor gradients are gathered per parameter)."""
    total = None
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        if hasattr(g, "full_tensor"):
            g = g.full_tensor()
        sq = g.detach().float().pow(2).sum()
        total = sq if total is None else total + sq.to(total.device)
    return 0.0 if total is None else float(total.sqrt().item())


def ppo_clip_loss(old_log_prob, log_prob, advantages, mask, clip_low, clip_high, clip_c):
    """Vanilla PPO objective with dual clip, token-mean aggregation (same math as JADE compute_policy_loss_vanilla)."""
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, mask)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), mask)
    pg_losses3 = -advantages * clip_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), mask)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = masked_mean(pg_losses, mask)  # token-mean
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def _response_prediction_mask(response_mask: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Mark sequence positions whose logits predict valid response tokens."""
    response_length = response_mask.shape[-1]
    if response_length >= sequence_length:
        raise ValueError(f"response length ({response_length}) must be smaller than sequence length ({sequence_length})")
    token_mask = torch.zeros((*response_mask.shape[:-1], sequence_length), dtype=torch.bool, device=response_mask.device)
    token_mask[..., -response_length:] = response_mask.bool()
    prediction_mask = torch.zeros_like(token_mask)
    prediction_mask[..., :-1] = token_mask[..., 1:]
    return prediction_mask


def forward_log_probs(model, micro: dict[str, torch.Tensor], temperature: float, cfg: SippoActorConfig, device_type: str):
    """Per-token log-probabilities of ``micro['responses']`` under ``model``; shape (rows, response_length).

    Remove-padding path packs all rows into one sequence and relies on the attention implementation's varlen kernel
    (flash_attention_2) to keep rows apart: ``sdpa`` would leak context across rows (see _checks/packing_integrity.py).
    """
    responses = micro["responses"]
    response_length = responses.size(-1)
    with torch.autocast(device_type=device_type, dtype=cfg.autocast_dtype):
        input_ids = micro["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro["attention_mask"]
        position_ids = micro["position_ids"]
        if cfg.use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)
            rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
            response_mask = micro.get("response_mask", attention_mask[:, -response_length:]).bool()
            extra = {}
            logit_indices = None
            if cfg.response_only_logits:
                prediction_mask = _response_prediction_mask(response_mask, seqlen)
                prediction_mask_rmpad = index_first_axis(prediction_mask.reshape(-1, 1), indices).squeeze(-1)
                logit_indices = torch.nonzero(prediction_mask_rmpad, as_tuple=False).flatten()
                extra["logits_to_keep"] = logit_indices
            out = model(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad,
                        use_cache=False, **extra)
            logits = out.logits.squeeze(0)
            logits.div_(temperature)
            labels = rolled.squeeze(0)
            if cfg.response_only_logits:
                labels = labels.index_select(0, logit_indices)
                if not torch.equal(labels, micro["responses"][response_mask]):
                    raise RuntimeError("response-only logits are not aligned with response tokens")
                selected = logprobs_from_logits(logits=logits, labels=labels, inplace_backward=True)
                return torch.zeros_like(responses, dtype=selected.dtype).masked_scatter(response_mask, selected)
            log_probs = logprobs_from_logits(logits=logits, labels=labels, inplace_backward=True)
            full = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
            return full.squeeze(-1)[:, -response_length - 1 : -1]
        out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        logits = out.logits
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]
        return logprobs_from_logits(logits, responses)


class SippoPolicyUpdater:
    """Runs one SIPPO policy update on a model + optimizer owned by the caller."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: SippoActorConfig,
        *,
        device: str | torch.device,
        clip_grad_fn: Optional[Callable[[torch.nn.Module, float], torch.Tensor]] = None,
        before_step: Optional[Callable[[], None]] = None,
        after_step: Optional[Callable[[], None]] = None,
        optimizer_step_fn: Optional[Callable[[], float]] = None,
        dp_size: int = 1,
        no_sync: Optional[Callable[[], Any]] = None,
        allreduce_grads: Optional[Callable[[], None]] = None,
    ):
        """``optimizer_step_fn`` (e.g. ``engine.optimizer_step``) must clip gradients, skip non-finite updates, step the
        optimizer and return the pre-clip gradient norm; when given it replaces ``clip_grad_fn``/``before/after_step``."""
        cfg.validate_supported()
        self.model, self.optimizer, self.cfg = model, optimizer, cfg
        self.device = torch.device(device)
        self.device_type = self.device.type
        self._clip = clip_grad_fn or (lambda m, c: torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=c))
        self._before_step, self._after_step = before_step, after_step
        self._optimizer_step_fn = optimizer_step_fn
        # data parallelism (one full model replica per rank): backward runs without gradient synchronisation and the
        # gradients are averaged over the ranks once per mini-batch, so micro-batch counts may differ between ranks
        self._dp_size = int(dp_size)
        self._no_sync = no_sync
        self._allreduce_grads = allreduce_grads
        if self._dp_size > 1 and (no_sync is None or allreduce_grads is None):
            raise ValueError("dp_size > 1 needs no_sync and allreduce_grads")

    # ---- batching -----------------------------------------------------------------------------------------------
    @staticmethod
    def _take(tensors: dict, non_tensors: dict, idx) -> tuple[dict, dict]:
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return ({k: v[idx_t] for k, v in tensors.items()}, {k: np.asarray(v)[np.asarray(idx)] for k, v in non_tensors.items()})

    def _micro_batches(self, tensors: dict, non_tensors: dict):
        """Split one mini-batch into micro-batches. Returns (list of (tensors, non_tensors), index lists or None)."""
        n = next(iter(tensors.values())).shape[0]
        if self.cfg.use_dynamic_bsz:
            td = TensorDict({k: v for k, v in tensors.items()}, batch_size=[n])
            min_num = n if self.cfg.force_single_row_dynamic_microbatches else None
            mbs, idx_lists = rearrange_micro_batches(td, max_token_len=self.cfg.ppo_max_token_len_per_gpu,
                                                     min_num_micro_batch=min_num)
            out = []
            for mb, idx in zip(mbs, idx_lists):
                out.append((dict(mb), {k: np.asarray(v)[np.asarray(idx)] for k, v in non_tensors.items()}))
            return out, idx_lists
        size = self.cfg.ppo_micro_batch_size_per_gpu
        out = []
        for s in range(0, n, size):
            idx = list(range(s, min(s + size, n)))
            out.append(self._take(tensors, non_tensors, idx))
        return out, None

    # ---- PSC coefficient refresh ---------------------------------------------------------------------------------
    def _current_log_probs(self, tensors: dict, non_tensors: dict, row_mask: np.ndarray, temperature: float):
        """Score the selected rows of a mini-batch with the CURRENT parameters (no grad)."""
        rows = np.flatnonzero(row_mask).tolist()
        zero = torch.zeros_like(tensors["response_mask"], dtype=torch.float32, device="cpu")
        if not rows:
            return zero
        sub_t, sub_n = self._take(tensors, non_tensors, rows)
        micro, idx_lists = self._micro_batches(sub_t, sub_n)
        outs = []
        with torch.no_grad():
            for mt, _ in micro:
                mt = {k: v.to(self.device) for k, v in mt.items()}
                outs.append(forward_log_probs(self.model, mt, temperature, self.cfg, self.device_type).detach())
        out = torch.cat(outs, dim=0)
        if idx_lists is not None:
            out = restore_dynamic_batch(out, idx_lists)
        out = out.cpu()
        sparse = torch.zeros_like(tensors["response_mask"], dtype=out.dtype, device="cpu")
        sparse[rows] = out
        return sparse

    @staticmethod
    def _allreduce_sum(values: list[float]) -> list[float]:
        """SUM all-reduce of small non-negative counts; identity on one rank (HCCL has no float64)."""
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return values
        t = torch.tensor(values, dtype=torch.float32, device="cpu" if torch.distributed.get_backend() == "gloo" else None)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return [float(x) for x in t.tolist()]

    def _refresh_nrpsc(self, tensors: dict, non_tensors: dict, meta: dict, *, temperature, beta, dp_size):
        if "nrpsc_plan_view" not in non_tensors or meta.get("nrpsc_idf") is None:
            raise ValueError("NR-PSC requires Planner views and frozen TF-IDF")
        roles = np.asarray(non_tensors["role"], dtype=object)
        valid = tensors["actor_dp_valid_rows"].detach().cpu().numpy().astype(np.bool_, copy=False)
        planner_rows = (roles == PLANNER_ROLE) & valid
        current = self._current_log_probs(tensors, non_tensors, planner_rows, temperature)
        eligibility = non_tensors.get("psc_chosen_eligible")
        if meta.get("planner_outcome_contrast_use_reward_eligibility", False):
            eligibility = actual_reward_eligibility(non_tensors)
        observations, representatives, duplicate_rows = [], {}, {}
        for row in np.flatnonzero(planner_rows):
            view = non_tensors["nrpsc_plan_view"][row]
            if view is None:
                continue
            outcome = float(non_tensors["outcome_exact"][row])
            if outcome not in (0.0, 1.0):
                raise ValueError("NR-PSC requires binary terminal EM")
            successes = int(non_tensors["nrpsc_successes"][row]) if "nrpsc_successes" in non_tensors else int(outcome)
            trials = int(non_tensors["nrpsc_trials"][row]) if "nrpsc_trials" in non_tensors else 1
            eligible = eligibility is None or outcome == 0 or bool(eligibility[row])
            trajectory = non_tensors["trajectory_id"][row]
            if trajectory in representatives:
                previous = representatives[trajectory]
                original = previous.row
                if (previous.question != non_tensors["uid"][row] or previous.successes != successes
                        or previous.trials != trials or previous.eligible != eligible
                        or previous.view.response_ids != view.response_ids
                        or previous.view.prompt_ids != view.prompt_ids
                        or not torch.equal(tensors["response_mask"][original], tensors["response_mask"][row])
                        or not torch.allclose(current[original], current[row], atol=1e-6, rtol=0.0)):
                    raise ValueError("duplicate NR-PSC Planner rows disagree")
                duplicate_rows[trajectory].append(int(row))
                continue
            observation = PlanObservation(non_tensors["uid"][row], int(row), view, successes, trials, eligible)
            representatives[trajectory] = observation
            duplicate_rows[trajectory] = [int(row)]
            observations.append(observation)
        questions = len(set(np.asarray(non_tensors["uid"], dtype=object)[valid].tolist()))
        scorer = FrozenTfidf(meta["nrpsc_idf"], meta.get("nrpsc_unknown_idf"))
        table = meta.get("nrpsc_reliability_table", {})
        coefficients, counts = batch_node_coefficients(
            current, tensors["response_mask"].cpu(), observations, scorer,
            float(meta["nrpsc_similarity_threshold"]), question_count=max(questions, 1), beta=beta,
            reliability=FrozenReliabilityTable(table) if table else None,
        )
        for trajectory, rows in duplicate_rows.items():
            if len(rows) > 1:
                shared = coefficients[representatives[trajectory].row].clone() / len(rows)
                for row in rows:
                    coefficients[row] = shared
        global_questions = self._allreduce_sum([float(questions)])[0]
        scale = questions * dp_size / global_questions if global_questions else 0.0
        metrics = {"aux/nrpsc/" + key: float(value) for key, value in counts.items()
                   if key != "candidate_counts"}
        metrics.update({"aux/nrpsc/global_question_count": global_questions,
                        "aux/nrpsc/coefficient_scale": scale})
        return coefficients * scale, metrics

    def _refresh_poc(self, tensors: dict, non_tensors: dict, *, temperature, mode, beta, loss_weight, shuffle_seed, require_signal, dp_size):
        roles = np.asarray(non_tensors["role"], dtype=object)
        valid = tensors["actor_dp_valid_rows"].detach().cpu().numpy().astype(np.bool_, copy=False)
        current = self._current_log_probs(tensors, non_tensors, (roles == PLANNER_ROLE) & valid, temperature)
        result = build_planner_outcome_contrast_coefficients(
            response_mask=tensors["response_mask"].cpu(),
            old_log_probs=current,
            question_ids=non_tensors["uid"],
            trajectory_ids=non_tensors["trajectory_id"],
            roles=non_tensors["role"],
            outcomes=non_tensors["outcome_exact"],
            chosen_eligibility=non_tensors.get("psc_chosen_eligible"),
            workflow_steps=non_tensors["workflow_step"],
            valid_rows=tensors["actor_dp_valid_rows"].cpu(),
            beta=beta,
            data_parallel_size=1,
            shuffled=mode == "shuffled",
            shuffle_seed=shuffle_seed,
        )
        local_informative = float(result.metrics["aux/planner_outcome_contrast/informative_question_count"])
        global_informative, global_pairs = self._allreduce_sum(
            [local_informative, float(result.metrics["aux/planner_outcome_contrast/pair_count"])]
        )
        scale = local_informative * dp_size / global_informative if global_informative > 0 else 0.0
        coefficients = result.coefficients * scale
        # diagnostics identical to the original (planner EM direct gradient vs weighted PSC gradient)
        valid_token_mask = tensors["response_mask"].float() * tensors["actor_dp_valid_rows"].unsqueeze(-1).float()
        token_mass = valid_token_mask.sum().clamp_min(1.0)
        em_direct = -tensors["advantages"].float() * valid_token_mask / token_mass
        planner_rows = torch.as_tensor(roles == PLANNER_ROLE, dtype=torch.bool).unsqueeze(-1)
        planner_em = em_direct * planner_rows
        weighted = coefficients * loss_weight
        em_norm, poc_norm = torch.linalg.vector_norm(planner_em), torch.linalg.vector_norm(weighted)
        cosine = torch.sum(planner_em * weighted) / (em_norm * poc_norm).clamp_min(1e-12)
        metrics = dict(result.metrics)
        metrics.update({
            "aux/planner_outcome_contrast/refreshed": 1.0,
            "aux/planner_outcome_contrast/global_informative_question_count": global_informative,
            "aux/planner_outcome_contrast/global_pair_count": global_pairs,
            "aux/planner_outcome_contrast/current_policy_scored_rows": float(((roles == PLANNER_ROLE) & valid).sum()),
            "aux/planner_outcome_contrast/local_coefficient_scale": scale,
            "aux/planner_outcome_contrast/empty_refreshed_minibatch": float(global_pairs <= 0),
            "aux/planner_outcome_contrast/global_signal_required": float(require_signal),
            "aux/planner_outcome_contrast/refreshed_planner_em_direct_gradient_norm": float(em_norm.item()),
            "aux/planner_outcome_contrast/refreshed_weighted_poc_direct_gradient_norm": float(poc_norm.item()),
            "aux/planner_outcome_contrast/refreshed_weighted_poc_to_planner_em_direct_gradient_ratio": float(
                (poc_norm / em_norm.clamp_min(1e-12)).item()),
            "aux/planner_outcome_contrast/refreshed_weighted_poc_planner_em_direct_gradient_cosine": float(
                cosine.item() if em_norm.item() > 0 and poc_norm.item() > 0 else 0.0),
        })
        return coefficients, metrics

    # ---- R arm: shadow gradients and norm matching ---------------------------------------------------------------
    def _shadow_grad_norm(self, tensors: dict, non_tensors: dict, coef: torch.Tensor, temperature: float) -> float:
        """Global gradient norm of ``sum(log_prob * coef)`` over the rows where ``coef`` is non-zero.

        A *shadow* gradient: parameters and optimizer state are untouched and gradients are cleared afterwards.
        """
        rows = torch.nonzero(coef.abs().sum(dim=-1) != 0).flatten().tolist()
        if not rows:
            return 0.0
        sub_t, sub_n = self._take(tensors, non_tensors, rows)
        sub_t["_shadow_coef"] = coef[rows]
        micro, _ = self._micro_batches(sub_t, sub_n)
        self.optimizer.zero_grad()
        for mt, _mn in micro:
            mt = {k: v.to(self.device) for k, v in mt.items()}
            lp = forward_log_probs(self.model, mt, temperature, self.cfg, self.device_type)
            planner_outcome_contrast_loss(lp, mt["_shadow_coef"]).backward()
        norm = _global_grad_norm(self.model)
        self.optimizer.zero_grad()
        return norm

    def _planner_base_grad_norm(self, tensors: dict, non_tensors: dict, temperature: float, on_policy: bool, mb_mass: float) -> float:
        """Global gradient norm of the Planner rows' share of the base (EM / workflow-RLOO) objective of this mini-batch."""
        roles = np.asarray(non_tensors["role"], dtype=object)
        valid = tensors["actor_dp_valid_rows"].detach().cpu().numpy().astype(np.bool_, copy=False)
        rows = np.flatnonzero((roles == PLANNER_ROLE) & valid).tolist()
        if not rows:
            return 0.0
        sub_t, sub_n = self._take(tensors, non_tensors, rows)
        micro, _ = self._micro_batches(sub_t, sub_n)
        self.optimizer.zero_grad()
        for mt, _mn in micro:
            mt = {k: v.to(self.device) for k, v in mt.items()}
            loss_mask = mt["response_mask"] * mt["actor_dp_valid_rows"].unsqueeze(-1)
            scale = float(loss_mask.sum().detach().item()) / mb_mass
            lp = forward_log_probs(self.model, mt, temperature, self.cfg, self.device_type)
            old = mt["old_log_probs"] if (self.cfg.use_rollout_log_probs or not on_policy) else lp.detach()
            pg_loss, *_ = ppo_clip_loss(old, lp, mt["advantages"], loss_mask, self.cfg.clip_ratio_low,
                                        self.cfg.clip_ratio_high, self.cfg.clip_ratio_c)
            (pg_loss * scale).backward()
        norm = _global_grad_norm(self.model)
        self.optimizer.zero_grad()
        return norm

    def _r_control_coefficients(self, mb_tensors: dict, mb_non: dict, *, temperature: float, on_policy: bool, mb_mass: float, r: dict):
        """Pointwise-RLOO coefficients for this mini-batch, scaled so that  ||grad(aux)|| = target_weight * ||grad(PSC)||.

        The PSC shadow is built exactly as the P arm would build it (refreshed at the current policy, beta, unweighted),
        its gradient norm is measured on all trainable parameters, and the R coefficients are multiplied by
        ``c = target_weight * ||g_PSC|| / (||g_R|| + eps)`` (``c`` is a constant: no gradient flows through it).
        A mini-batch without signal (no informative question, or a zero norm) gets ``c = 0`` and is recorded -- no
        large coefficient is ever used to manufacture a gradient. If ``max_scale`` is set and binding, ``c`` is clipped
        and the resulting matching error is reported.
        """
        psc_coef, psc_metrics = self._refresh_poc(
            mb_tensors, mb_non, temperature=temperature, mode="real", beta=r["beta"], loss_weight=r["target_weight"],
            shuffle_seed=0, require_signal=False, dp_size=r["dp_size"])
        r_coef, r_metrics = build_planner_pointwise_rloo_coefficients(
            response_mask=mb_tensors["response_mask"].cpu(),
            question_ids=mb_non["uid"],
            trajectory_ids=mb_non["trajectory_id"],
            roles=mb_non["role"],
            outcomes=mb_non["outcome_exact"],
            workflow_steps=mb_non["workflow_step"],
            chosen_eligibility=mb_non.get("psc_chosen_eligible"),
            valid_rows=mb_tensors["actor_dp_valid_rows"].cpu(),
            data_parallel_size=r["dp_size"],
        )
        psc_inf = psc_metrics["aux/planner_outcome_contrast/informative_question_count"]
        if int(psc_inf) != int(r_metrics["aux/planner_control/informative_question_count"]):
            raise RuntimeError(
                f"R control and PSC disagree on informative questions ({r_metrics['aux/planner_control/informative_question_count']} "
                f"vs {psc_inf}); the grouping rules must be identical")
        g_psc = self._shadow_grad_norm(mb_tensors, mb_non, psc_coef, temperature)
        g_r = self._shadow_grad_norm(mb_tensors, mb_non, r_coef, temperature)
        target = r["target_weight"] * g_psc
        scale_c = target / (g_r + r["epsilon"]) if (g_r > 0.0 and target > 0.0) else 0.0
        hit_cap = False
        if r["max_scale"] is not None and scale_c > r["max_scale"]:
            scale_c, hit_cap = float(r["max_scale"]), True
        matched = scale_c * g_r
        out = dict(r_metrics)
        out.update({
            "aux/planner_control/scale_c": float(scale_c),
            "aux/planner_control/psc_shadow_grad_norm": float(g_psc),
            "aux/planner_control/r_grad_norm": float(g_r),
            "aux/planner_control/target_grad_norm": float(target),
            "aux/planner_control/matched_grad_norm": float(matched),
            "aux/planner_control/matching_relative_error": float(abs(matched - target) / target) if target > 0 else 0.0,
            "aux/planner_control/hit_scale_cap": float(hit_cap),
            "aux/planner_control/zero_norm_minibatch": float(scale_c == 0.0),
            "aux/planner_control/psc_pair_count": float(psc_metrics["aux/planner_outcome_contrast/pair_count"]),
        })
        if r["record_planner_base_grad"]:
            base = self._planner_base_grad_norm(mb_tensors, mb_non, temperature, on_policy, mb_mass)
            out["aux/planner_control/planner_base_grad_norm"] = float(base)
            out["aux/planner_control/aux_to_planner_base_grad_ratio"] = float(matched / base) if base > 0 else 0.0
        return r_coef * scale_c, out

    # ---- the update -----------------------------------------------------------------------------------------------
    def update(self, tensors: dict[str, torch.Tensor], non_tensors: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, list]:
        cfg = self.cfg
        for key in TENSOR_KEYS:
            if key not in tensors:
                raise ValueError(f"missing batch tensor: {key}")
        for key in NON_TENSOR_KEYS:
            if key not in non_tensors:
                raise ValueError(f"question-preserving actor minibatches require {key} metadata")
        if str(meta.get("planner_outcome_contrast_minibatch_mode", "workflow_single")) != "question_preserving":
            raise NotImplementedError("only question_preserving mini-batches are implemented (the W/P runs use them)")
        for unsupported in ("linear_full_trajectory", "planner_outcome_contrast_gradient_audit",
                            "planner_outcome_contrast_current_policy_audit_only"):
            if meta.get(unsupported, False):
                raise NotImplementedError(f"{unsupported} is not used by the W/P runs")
        if "workflow_loss_weights" in tensors:
            raise ValueError("question-preserving token-mean minibatches cannot use workflow loss weights")

        temperature = float(meta["temperature"])
        mode = str(meta.get("planner_outcome_contrast_mode", "off"))
        poc_enabled = mode in {"real", "shuffled", "nrpsc"}
        beta = float(meta.get("planner_outcome_contrast_beta", 1.0))
        shuffle_seed = int(meta.get("planner_outcome_contrast_shuffle_seed", 0))
        require_signal = bool(meta.get("planner_outcome_contrast_require_signal", True))
        dp_size = int(meta.get("planner_outcome_contrast_data_parallel_size", 1))
        loss_weight = float(meta.get("planner_outcome_contrast_loss_weight", 0.0))

        # R arm (pointwise-RLOO Planner control with PSC-matched gradient norm); mutually exclusive with PSC
        aux_mode = str(meta.get("planner_aux_mode", "off"))
        if aux_mode not in {"off", "pointwise_rloo_matched"}:
            raise NotImplementedError(f"planner_aux_mode={aux_mode!r}")
        r_enabled = aux_mode == "pointwise_rloo_matched"
        if r_enabled and poc_enabled:
            raise ValueError("the R control and PSC are different arms: planner_outcome_contrast.mode must be off for R")
        r_cfg = {
            "target_weight": float(meta.get("planner_aux_target_psc_loss_weight", 0.02)),
            "beta": float(meta.get("planner_aux_beta", 1.0)),
            "epsilon": float(meta.get("planner_aux_epsilon", 1e-12)),
            "max_scale": (None if meta.get("planner_aux_max_scale") in (None, "", "null") else float(meta["planner_aux_max_scale"])),
            "record_planner_base_grad": bool(meta.get("planner_aux_record_planner_base_grad", True)),
            "dp_size": dp_size,
        }
        aux_active = poc_enabled or r_enabled
        aux_weight = loss_weight if poc_enabled else 1.0  # for R the matched scale c is already inside the coefficients
        aux_prefix = "aux/planner_outcome_contrast" if poc_enabled else "aux/planner_control"

        self.model.train()
        tensors = {k: tensors[k] for k in TENSOR_KEYS}
        nt_keys = list(NON_TENSOR_KEYS) + [key for key in optional_psc_metadata(non_tensors)
                                           if key not in NON_TENSOR_KEYS]
        if mode == "nrpsc":
            nt_keys += [key for key in ("nrpsc_plan_view", "nrpsc_successes", "nrpsc_trials")
                        if key in non_tensors and key not in nt_keys]
        non_tensors = {k: np.asarray(non_tensors[k]) for k in nt_keys}

        index_sets = build_question_preserving_minibatch_indices(
            non_tensors["uid"], non_tensors["trajectory_id"], cfg.ppo_mini_batch_size)
        mini_batches = [self._take(tensors, non_tensors, idx) for idx in index_sets]
        on_policy = len(mini_batches) == 1 and cfg.ppo_epochs == 1

        metrics: dict[str, list] = {}
        metrics["actor/question_preserving_minibatch_count"] = [float(len(mini_batches))]
        metrics["actor/question_preserving_target_action_rows"] = [float(cfg.ppo_mini_batch_size)]
        sizes = [len(mb[0]["input_ids"]) for mb in mini_batches]
        metrics["actor/question_preserving_min_action_rows"] = [float(min(sizes))]
        metrics["actor/question_preserving_max_action_rows"] = [float(max(sizes))]
        steps_expected, steps_executed = len(mini_batches) * cfg.ppo_epochs, 0
        if self._dp_size > 1:  # every rank must run the same number of collectives; a mismatch would hang, so fail loudly
            counts = self._allreduce_sum([float(len(mini_batches)), float(len(mini_batches) ** 2)])
            if abs(counts[0] / self._dp_size - len(mini_batches)) > 1e-6 or abs(counts[1] / self._dp_size - len(mini_batches) ** 2) > 1e-6:
                raise RuntimeError(f"data-parallel ranks disagree on the mini-batch count (this rank {len(mini_batches)})")

        for mb_tensors, mb_non in mini_batches:
            if poc_enabled:
                if mode == "nrpsc":
                    coefficients, refreshed = self._refresh_nrpsc(
                        mb_tensors, mb_non, meta, temperature=temperature, beta=beta, dp_size=dp_size)
                else:
                    coefficients, refreshed = self._refresh_poc(
                        mb_tensors, mb_non, temperature=temperature, mode=mode, beta=beta, loss_weight=loss_weight,
                        shuffle_seed=shuffle_seed, require_signal=require_signal, dp_size=dp_size)
                mb_tensors["planner_outcome_contrast_coefficients"] = coefficients
                _append_dict(metrics, refreshed)
            mb_mask = mb_tensors["response_mask"] * mb_tensors["actor_dp_valid_rows"].unsqueeze(-1)
            if r_enabled:
                r_coefficients, r_metrics = self._r_control_coefficients(
                    mb_tensors, mb_non, temperature=temperature, on_policy=on_policy,
                    mb_mass=float(mb_mask.sum().detach().item()), r=r_cfg)
                mb_tensors["planner_outcome_contrast_coefficients"] = r_coefficients
                _append_dict(metrics, r_metrics)
            micro, _ = self._micro_batches(mb_tensors, mb_non)
            _append_dict(metrics, {
                "actor/dynamic_microbatch_count": float(len(micro)),
                "actor/dynamic_microbatch_max_rows": float(max(len(m[0]["input_ids"]) for m in micro)),
                "actor/forced_single_row_dynamic_microbatches": float(cfg.force_single_row_dynamic_microbatches),
            } if cfg.use_dynamic_bsz else {})
            self.optimizer.zero_grad()
            mb_mass = float(mb_mask.sum().detach().item())
            if mb_mass <= 0.0:
                raise ValueError("minibatch loss mask has no active mass")
            with (self._no_sync() if self._no_sync is not None else contextlib.nullcontext()):
                for mt, _mn in micro:
                    mt = {k: v.to(self.device) for k, v in mt.items()}
                    loss_mask = mt["response_mask"] * mt["actor_dp_valid_rows"].unsqueeze(-1)
                    scale = float(loss_mask.sum().detach().item()) / mb_mass
                    log_prob = forward_log_probs(self.model, mt, temperature, cfg, self.device_type)
                    old_log_prob = mt["old_log_probs"] if (cfg.use_rollout_log_probs or not on_policy) else log_prob.detach()
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = ppo_clip_loss(
                        old_log_prob, log_prob, mt["advantages"], loss_mask, cfg.clip_ratio_low, cfg.clip_ratio_high, cfg.clip_ratio_c)
                    loss = pg_loss * scale
                    mm: dict[str, float] = {}
                    if aux_active:
                        contrast = planner_outcome_contrast_loss(log_prob, mt["planner_outcome_contrast_coefficients"])
                        weighted = contrast * aux_weight
                        loss = loss + weighted
                        mm[f"{aux_prefix}/loss"] = contrast.detach().item()
                        mm[f"{aux_prefix}/weighted_loss"] = weighted.detach().item()
                    loss.backward()
                    mm.update({
                        "actor/pg_loss": pg_loss.detach().item() * scale,
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    })
                    _append_dict(metrics, mm)
            if self._dp_size > 1:
                self._allreduce_grads()
            if self._optimizer_step_fn is not None:
                grad_norm = torch.tensor(float(self._optimizer_step_fn()))
            else:
                grad_norm = self._clip(self.model, cfg.grad_clip)
                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.full_tensor()
                if not torch.isfinite(grad_norm):
                    self.optimizer.zero_grad()
                else:
                    if self._before_step:
                        self._before_step()
                    try:
                        self.optimizer.step()
                    finally:
                        if self._after_step:
                            self._after_step()
            steps_executed += int(torch.isfinite(grad_norm).item())
            _append(metrics, "actor/grad_norm", grad_norm.detach().item())

        metrics["actor/optimizer_steps_expected"] = [float(steps_expected)]
        metrics["actor/optimizer_steps_executed"] = [float(steps_executed)]
        metrics["actor/ppo_optimizer_step"] = [float(steps_executed > 0)]
        self.optimizer.zero_grad()
        return metrics

    # ---- frozen-batch gradient comparison (NR-PSC vs PSC); no optimizer step, parameters are never changed ----------
    def _grad_list(self, mt_all: dict, mn_all: dict, temperature: float, *, coef=None, on_policy: bool = False, planner_only: bool = False):
        """Full-parameter gradient (detached CPU fp32, one entry per parameter, None if no gradient) of either the base
        W objective of this mini-batch (``coef is None``) or ``sum(log_prob * coef)`` (the unweighted PSC/NR-PSC loss)."""
        cfg = self.cfg
        mb_mass = float((mt_all["response_mask"] * mt_all["actor_dp_valid_rows"].unsqueeze(-1)).sum().item())
        sub_t, sub_n = mt_all, mn_all
        if planner_only:  # Planner share of the base W (workflow-RLOO) loss, normalised by the full mini-batch mass
            roles = np.asarray(mn_all["role"], dtype=object)
            validm = mt_all["actor_dp_valid_rows"].detach().cpu().numpy().astype(np.bool_, copy=False)
            prow = np.flatnonzero((roles == PLANNER_ROLE) & validm).tolist()
            if not prow:
                return None
            sub_t, sub_n = self._take(mt_all, mn_all, prow)
        if coef is not None:
            rows = torch.nonzero(coef.abs().sum(dim=-1) != 0).flatten().tolist()
            if not rows:
                return None
            sub_t, sub_n = self._take(mt_all, mn_all, rows)
            sub_t["_shadow_coef"] = coef[rows]
        micro, _ = self._micro_batches(sub_t, sub_n)
        self.optimizer.zero_grad()
        for mt, _mn in micro:
            mt = {k: v.to(self.device) for k, v in mt.items()}
            lp = forward_log_probs(self.model, mt, temperature, cfg, self.device_type)
            if coef is None:
                loss_mask = mt["response_mask"] * mt["actor_dp_valid_rows"].unsqueeze(-1)
                scale = float(loss_mask.sum().detach().item()) / mb_mass
                old = mt["old_log_probs"] if (cfg.use_rollout_log_probs or not on_policy) else lp.detach()
                pg_loss, *_ = ppo_clip_loss(old, lp, mt["advantages"], loss_mask, cfg.clip_ratio_low,
                                            cfg.clip_ratio_high, cfg.clip_ratio_c)
                (pg_loss * scale).backward()
            else:
                planner_outcome_contrast_loss(lp, mt["_shadow_coef"]).backward()
        grads = []
        for p in self.model.parameters():
            g = p.grad
            if g is not None and hasattr(g, "full_tensor"):
                g = g.full_tensor()
            grads.append(None if g is None else g.detach().float().cpu())
        self.optimizer.zero_grad()
        return grads

    @staticmethod
    def _grad_dot(a, b) -> float:
        if a is None or b is None:
            return 0.0
        total = 0.0
        for x, y in zip(a, b):
            if x is not None and y is not None:
                total += float(torch.dot(x.reshape(-1).double(), y.reshape(-1).double()).item())
        return total

    def compare_gradients(self, tensors: dict, non_tensors: dict, meta: dict, psc_weight: float = 0.02) -> dict:
        """Gradients of the base W loss, original PSC (mode=real, P-arm settings) and NR-PSC on a frozen real batch.

        Walks the same question-preserving mini-batches as ``update`` but never steps the optimizer. Per mini-batch it
        reports full-parameter gradient norms (unweighted losses), pairwise cosines, the planner-support check of each
        coefficient tensor and the NR-PSC weight that matches the weighted PSC gradient (norm match and projection).
        """
        cfg = self.cfg
        temperature = float(meta["temperature"])
        beta = float(meta.get("planner_outcome_contrast_beta", 1.0))
        dp_size = int(meta.get("planner_outcome_contrast_data_parallel_size", 1))
        shuffle_seed = int(meta.get("planner_outcome_contrast_shuffle_seed", 0))
        index_sets = build_question_preserving_minibatch_indices(
            non_tensors["uid"], non_tensors["trajectory_id"], cfg.ppo_mini_batch_size)
        mini_batches = [self._take(tensors, non_tensors, idx) for idx in index_sets]
        on_policy = len(mini_batches) == 1 and cfg.ppo_epochs == 1
        import os as _os
        lean = _os.environ.get("SIPPO_COMPARE_LEAN") == "1"  # keep only Planner-RLOO, PSC and NR-PSC gradients (less memory/time)
        meta_no_elig = dict(meta)
        meta_no_elig["planner_outcome_contrast_use_reward_eligibility"] = False
        records = []
        self.model.train()
        for number, (mb_t, mb_n) in enumerate(mini_batches):
            roles = np.asarray(mb_n["role"], dtype=object)
            planner_rows = torch.as_tensor(roles == PLANNER_ROLE, dtype=torch.bool)
            valid = mb_t["actor_dp_valid_rows"].bool()
            coefs = {}
            psc_coef, psc_metrics = self._refresh_poc(
                mb_t, mb_n, temperature=temperature, mode="real", beta=beta, loss_weight=psc_weight,
                shuffle_seed=shuffle_seed, require_signal=False, dp_size=dp_size)
            coefs["psc"] = psc_coef
            coefs["nrpsc"], nr_metrics = self._refresh_nrpsc(
                mb_t, mb_n, meta, temperature=temperature, beta=beta, dp_size=dp_size)
            nr_off_metrics = {}
            if not lean:
                coefs["nrpsc_no_reward_eligibility"], nr_off_metrics = self._refresh_nrpsc(
                    mb_t, mb_n, meta_no_elig, temperature=temperature, beta=beta, dp_size=dp_size)
            rec = {"minibatch": number, "rows": int(len(mb_t["input_ids"])),
                   "planner_rows": int(planner_rows.sum()),
                   "questions": len(set(np.asarray(mb_n["uid"], dtype=object)[valid.numpy()].tolist()))}
            for name, coef in coefs.items():
                nonzero = coef.abs() > 0
                rec[f"{name}/nonzero_tokens"] = int(nonzero.sum())
                rec[f"{name}/nonzero_tokens_outside_planner_rows"] = int((nonzero & ~planner_rows.unsqueeze(-1)).sum())
                rec[f"{name}/nonzero_tokens_outside_response_mask"] = int((nonzero & ~mb_t["response_mask"].bool()).sum())
                rec[f"{name}/nonzero_tokens_in_invalid_rows"] = int((nonzero & ~valid.unsqueeze(-1)).sum())
                rec[f"{name}/coef_abs_sum"] = float(coef.abs().sum())
            rec["psc/informative_question_count"] = float(psc_metrics["aux/planner_outcome_contrast/informative_question_count"])
            rec["psc/pair_count"] = float(psc_metrics["aux/planner_outcome_contrast/pair_count"])
            for key, value in nr_metrics.items():
                rec[f"nrpsc_metric/{key}"] = value
            rec["nrpsc_no_reward_eligibility_metric/informative_class_pairs"] = nr_off_metrics.get(
                "aux/nrpsc/informative_class_pairs")
            grads = {} if lean else {"base": self._grad_list(mb_t, mb_n, temperature, coef=None, on_policy=on_policy)}
            grads |= {"planner_rloo": self._grad_list(mb_t, mb_n, temperature, coef=None, on_policy=on_policy, planner_only=True)}
            for name, coef in coefs.items():
                grads[name] = self._grad_list(mb_t, mb_n, temperature, coef=coef)
            norms = {name: (self._grad_dot(g, g) ** 0.5 if g is not None else 0.0) for name, g in grads.items()}
            for name, value in norms.items():
                rec[f"grad_norm/{name}"] = value
            names = [n for n in grads if grads[n] is not None]
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    dot = self._grad_dot(grads[left], grads[right])
                    denom = norms[left] * norms[right]
                    rec[f"cosine/{left}__{right}"] = dot / denom if denom > 0 else 0.0
                    rec[f"dot/{left}__{right}"] = dot
            for name in ("nrpsc", "nrpsc_no_reward_eligibility"):
                if name in norms and norms[name] > 0 and norms["psc"] > 0:
                    rec[f"matched_weight_norm/{name}"] = psc_weight * norms["psc"] / norms[name]
                    rec[f"matched_weight_projection/{name}"] = psc_weight * rec["dot/psc__" + name] / (norms[name] ** 2)
            if norms.get("base", 0) > 0:
                for name in ("psc", "nrpsc", "nrpsc_no_reward_eligibility"):
                    if name not in norms:
                        continue
                    rec[f"weighted_to_base_ratio_at_{psc_weight}/{name}"] = psc_weight * norms[name] / norms["base"]
            records.append(rec)
            del grads
            with open(_os.environ.get('SIPPO_PROGRESS_FILE', '/tmp/sippo_progress.txt'), 'a') as _f:
                _f.write(f'minibatch {number + 1}/{len(mini_batches)} done\n')
        return {"psc_weight": psc_weight, "minibatches": records}
