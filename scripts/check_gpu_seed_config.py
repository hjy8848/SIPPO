"""Construct the actual rollout dataclass before allocating a training GPU."""
import sys

from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.rollout import RolloutConfig


config = OmegaConf.load(sys.argv[1])
rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
assert isinstance(rollout, RolloutConfig)
assert rollout.seed == 42
assert config.data.seed == 42
assert config.trainer.n_gpus_per_node == 1
assert config.qa_manager.workflow_rollout_n == 4
assert not config.qa_manager.contract_preserving_dag_v2
assert config.qa_manager.planner_prompt_max_nodes == 0
print("ROLLOUT_CONFIG_CONSTRUCTION_OK seed=42 gpu=1 n=4 legacy_contract")
