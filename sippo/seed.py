"""One training seed -> every random source that matters for a SIPPO W/P run.

``+trainer.seed=S`` is the only knob. It fans out to

  data.seed                          order in which training questions are drawn (RandomSampler generator)
  actor_rollout_ref.rollout.seed     vLLM engine sampling seed (engine seed = rollout.seed + replica_rank)
  SIPPO_SEED (env, all Ray procs)    python ``random`` / NumPy / torch (+ torch_npu) in the driver and in workers

What is NOT affected (and why): the policy initialisation (always the SFT checkpoint), the PSC pair construction
(deterministic given the batch; ``planner_outcome_contrast.shuffle_seed`` only matters for the shuffled control),
and uid strings (random UUIDs used purely as identifiers). vLLM request scheduling is asynchronous, so two runs with
the same seed are statistically equivalent, not bit-identical; the seed makes runs distinct and reproducible in
distribution, and it is logged.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED_ENV = "SIPPO_SEED"


def apply_seed_to_config(config) -> int | None:
    """Fan ``trainer.seed`` out into the config. Returns the seed (or None when not requested)."""
    from omegaconf import OmegaConf, open_dict

    seed = OmegaConf.select(config, "trainer.seed", default=None)
    if seed is None:
        return None
    seed = int(seed)
    with open_dict(config):
        config.data.seed = seed
        config.actor_rollout_ref.rollout.seed = seed
        ray_init = config.ray_kwargs.get("ray_init", None)
        if ray_init is None:
            config.ray_kwargs.ray_init = {}
        if config.ray_kwargs.ray_init.get("runtime_env", None) is None:
            config.ray_kwargs.ray_init.runtime_env = {}
        if config.ray_kwargs.ray_init.runtime_env.get("env_vars", None) is None:
            config.ray_kwargs.ray_init.runtime_env.env_vars = {}
        config.ray_kwargs.ray_init.runtime_env.env_vars[SEED_ENV] = str(seed)
    seed_this_process(seed)
    return seed


def seed_this_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        try:
            import torch_npu  # noqa: F401

            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.manual_seed_all(seed)
        except Exception:
            pass
    except Exception:
        pass
    os.environ[SEED_ENV] = str(seed)


def seed_process_from_env() -> int | None:
    """Called inside Ray workers: pick the seed up from the environment the driver propagated."""
    value = os.environ.get(SEED_ENV)
    if value is None:
        return None
    seed_this_process(int(value))
    return int(value)
