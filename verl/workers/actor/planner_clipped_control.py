"""Single-rank Planner PPO reweighting with a detached PSC-matched budget."""
import math

import numpy as np
import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn


def planner_ppo_share(actor, inputs, log_prob, loss_mask, minibatch_mass, on_policy):
    roles = torch.as_tensor(
        np.asarray(inputs["role"], dtype=object) == "PlanningAgent",
        device=log_prob.device,
    ).unsqueeze(-1)
    planner_mask = loss_mask * roles
    old = inputs["old_log_probs"]
    if on_policy and not actor.config.get("use_rollout_log_probs", False):
        old = log_prob.detach()
    loss_fn = get_policy_loss_fn(actor.config.policy_loss.get("loss_mode", "vanilla"))
    loss, *_ = loss_fn(
        old_log_prob=old, log_prob=log_prob, advantages=inputs["advantages"],
        response_mask=planner_mask, loss_agg_mode="token-mean", config=actor.config,
        rollout_is_weights=inputs.get("rollout_is_weights"),
    )
    return loss * (planner_mask.sum() / minibatch_mass)


def parameter_gradient_norm(module):
    total = torch.zeros((), dtype=torch.float64, device=next(module.parameters()).device)
    for param in module.parameters():
        if param.grad is not None:
            total += torch.linalg.vector_norm(param.grad.detach().float()).double().square()
    value = float(total.sqrt().item())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite shadow gradient norm")
    return value


def matched_planner_scale(actor, micro_batches, minibatch_mass, temperature, on_policy,
                          target_weight=0.02, epsilon=1e-12):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("Planner clipped control is validated for one rank only")
    device = next(actor.actor_module.parameters()).device
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    norms = {}
    try:
        for objective in ("psc", "planner"):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            actor.actor_optimizer.zero_grad(set_to_none=True)
            for batch in micro_batches:
                batch = batch.to(device)
                inputs = {**batch.batch, **batch.non_tensor_batch}
                mask = inputs["response_mask"] * inputs["actor_dp_valid_rows"].unsqueeze(-1)
                coef = inputs["planner_outcome_contrast_coefficients"]
                if not bool(coef.abs().sum().item()) and objective == "psc":
                    continue
                _, lp = actor._forward_micro_batch(inputs, temperature=temperature, calculate_entropy=False)
                if objective == "psc":
                    loss = (lp * coef.to(lp.device)).sum() * target_weight
                else:
                    loss = planner_ppo_share(actor, inputs, lp, mask, minibatch_mass, on_policy)
                loss.backward()
            norms[objective] = parameter_gradient_norm(actor.actor_module)
    finally:
        actor.actor_optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    target, base = norms["psc"], norms["planner"]
    scale = target / (base + epsilon) if target > 0 and base > 0 else 0.0
    if not math.isfinite(scale) or scale < 0:
        raise FloatingPointError("invalid detached Planner reweight scale")
    matched = scale * base
    return scale, {
        "aux/planner_control/enabled": 1.0,
        "aux/planner_control/clipped_ppo_reweight": 1.0,
        "aux/planner_control/scale_c": scale,
        "aux/planner_control/planner_alpha": 1.0 + scale,
        "aux/planner_control/planner_base_grad_norm": base,
        "aux/planner_control/target_grad_norm": target,
        "aux/planner_control/matched_grad_norm": matched,
        "aux/planner_control/matching_relative_error": abs(matched-target)/target if target > 0 else 0.0,
        "aux/planner_control/zero_base_positive_target": float(base == 0 and target > 0),
        "aux/planner_control/psc_loss_applied": 0.0,
    }
