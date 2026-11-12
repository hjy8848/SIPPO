"""SIPPO entry point (verl 0.9.1 base).  python -m sippo.main <hydra overrides>

Same Hydra config as ``verl.trainer.main_ppo`` (0.9.1), with the SIPPO actor worker and the SIPPO trainer plugged in.
"""

from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.device import auto_set_device
from verl.utils.config import validate_config


@ray.remote
class SippoTaskRunner(BaseTaskRunner):
    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from sippo.worker import SippoActorRolloutRefWorker

        if need_reference_policy(config):
            raise NotImplementedError("SIPPO W/P runs use no reference policy (no KL)")
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(SippoActorRolloutRefWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        return SippoActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        from pprint import pprint

        print(f"SippoTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        if need_critic(config):
            raise NotImplementedError("workflow_rloo trains without a critic")
        self.add_reward_model_resource_pool(config)
        validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))

        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.workers.config import HFModelConfig

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer, processor = model_config.tokenizer, model_config.processor
        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True,
                                          max_samples=config.data.get("train_max_samples", -1))
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False,
                                        max_samples=config.data.get("val_max_samples", -1))
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from sippo.trainer import SippoRayPPOTrainer

        trainer = SippoRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    from sippo.seed import apply_seed_to_config

    seed = apply_seed_to_config(config)
    print(f"[sippo] training seed: {seed} (data order, vLLM sampling, process RNG)" if seed is not None
          else "[sippo] no trainer.seed given: data.seed/rollout.seed keep their config values")
    run_ppo(config, task_runner_class=SippoTaskRunner)


if __name__ == "__main__":
    main()
