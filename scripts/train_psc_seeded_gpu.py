"""Seeded entry point for the existing GPU trainer, without changing its losses."""
import os
import random

import hydra
import numpy as np
import ray
import torch

from verl.trainer.main_ppo import TaskRunner, run_ppo


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SeededTaskRunner(TaskRunner):
    def run(self, config):
        seed_process(int(os.environ["JADE_TRAIN_SEED"]))
        return super().run(config)


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    seed = int(os.environ["JADE_TRAIN_SEED"])
    assert config.data.seed == seed
    assert config.actor_rollout_ref.rollout.seed == seed
    seed_process(seed)
    print(f"GPU_SEED_CONTRACT data={seed} rollout={seed} driver/taskrunner={seed}", flush=True)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SeededTaskRunner))


if __name__ == "__main__":
    main()
