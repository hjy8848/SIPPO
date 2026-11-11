"""CPU checks: clipped gradients, role isolation, norm budget and optimizer isolation."""
import json
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.actor.planner_clipped_control import matched_planner_scale, planner_ppo_share


torch.set_num_threads(2)
config = OmegaConf.create({
    "policy_loss": {"loss_mode": "vanilla"}, "clip_ratio": .2,
    "clip_ratio_low": .2, "clip_ratio_high": .2, "clip_ratio_c": 3.,
    "use_rollout_log_probs": False,
})
roles = np.asarray(["PlanningAgent", "PlanningAgent", "AnswerGenerationAgent", "Summarizer"])
lp = torch.tensor([[.4, .05], [-.4, -.05], [.05, -.05], [-.05, .05]], requires_grad=True)
adv = torch.tensor([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]])
mask = torch.ones_like(lp)
inputs = {"role": roles, "old_log_probs": torch.zeros_like(lp), "advantages": adv}
actor = SimpleNamespace(config=config)
fn = get_policy_loss_fn("vanilla")
base = fn(old_log_prob=inputs["old_log_probs"], log_prob=lp, advantages=adv,
          response_mask=mask, config=config)[0]
share = planner_ppo_share(actor, inputs, lp, mask, float(mask.sum()), False)
g_base = torch.autograd.grad(base, lp, retain_graph=True)[0]
g_planner = torch.autograd.grad(share, lp, retain_graph=True)[0]
g_r = torch.autograd.grad(base + .7 * share, lp)[0]
expected = g_base.clone()
expected[:2] *= 1.7
torch.testing.assert_close(g_r, expected)
torch.testing.assert_close(g_planner[2:], torch.zeros_like(g_planner[2:]))
assert g_planner[0, 0] == 0 and g_planner[1, 0] == 0
assert g_planner[:2, 1].abs().sum() > 0


class ToyActor:
    def __init__(self):
        self.config = config
        self.actor_module = torch.nn.Linear(3, 2)
        self.actor_optimizer = torch.optim.AdamW(self.actor_module.parameters(), lr=1e-3)

    def _forward_micro_batch(self, inputs, **kwargs):
        return None, torch.nn.functional.logsigmoid(self.actor_module(inputs["features"]))


torch.manual_seed(42)
toy = ToyActor()
features = torch.randn(4, 3)
with torch.no_grad():
    old = toy._forward_micro_batch({"features": features})[1].clone()
tensors = {"features": features, "response_mask": mask, "actor_dp_valid_rows": torch.ones(4),
           "advantages": adv, "old_log_probs": old,
           "planner_outcome_contrast_coefficients": torch.tensor([[-.2, -.2], [.2, .2], [0., 0.], [0., 0.]])}
batch = DataProto(batch=TensorDict(tensors, batch_size=[4]), non_tensor_batch={"role": roles})
micro = batch.split(2)
before = {k: v.detach().clone() for k, v in toy.actor_module.state_dict().items()}
rng = torch.get_rng_state().clone()
scale, metrics = matched_planner_scale(toy, micro, 8., .8, False)
assert scale > 0
assert metrics["aux/planner_control/matching_relative_error"] < 1e-8
assert torch.equal(rng, torch.get_rng_state())
assert len(toy.actor_optimizer.state) == 0
assert all(p.grad is None for p in toy.actor_module.parameters())
for k, v in before.items():
    torch.testing.assert_close(v, toy.actor_module.state_dict()[k], rtol=0, atol=0)

toy.actor_optimizer.zero_grad()
for part in micro:
    inp = {**part.batch, **part.non_tensor_batch}
    _, prob = toy._forward_micro_batch(inp)
    (scale * planner_ppo_share(toy, inp, prob, inp["response_mask"], 8., False)).backward()
from verl.workers.actor.planner_clipped_control import parameter_gradient_norm
actual = parameter_gradient_norm(toy.actor_module)
assert abs(actual-metrics["aux/planner_control/target_grad_norm"]) / actual < 1e-5
print(json.dumps({"passed": True, "clipping_preserved": True, "other_roles_unchanged": True,
                  "shadow_optimizer_steps": 0, "actual_matched_gradient_norm": actual,
                  "metrics": metrics}, indent=2))
