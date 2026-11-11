import unittest
import inspect
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl.trainer.main_ppo import model_based_node_rewards_enabled
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            "<conclusion>": [20, 21],
            "Paris</conclusion>": [30, 31, 32],
        }
        return mapping[str(text)]


class _FakeAgentManager:
    @staticmethod
    def trans_messages_to_ids(messages):
        del messages
        return {
            "input_ids": torch.tensor([10, 11], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        }


class _FullPromptAgentManager:
    @staticmethod
    def trans_messages_to_ids(messages):
        del messages
        return {
            "input_ids": torch.tensor([10, 11, 12, 13], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        }


class AttributionRuntimeTest(unittest.TestCase):
    def test_hcapo_builds_only_factual_and_subgraph_counterfactual_probes(self):
        source = inspect.getsource(RayPPOTrainer._score_node_reward_specs)
        hcapo_probe_block = source.split("for spec in selected_hcapo:", 1)[1].split(
            "progress = self.global_steps", 1
        )[0]

        self.assertIn('"hcapo_factual"', hcapo_probe_block)
        self.assertIn('"hcapo_counterfactual"', hcapo_probe_block)
        self.assertNotIn('"attribution_counterfactual"', hcapo_probe_block)

    def test_open_tag_is_prompt_and_not_part_of_scored_response(self):
        trainer = SimpleNamespace(
            config=OmegaConf.create({"data": {"max_response_length": 5}}),
            tokenizer=_FakeTokenizer(),
        )
        probe = RayPPOTrainer._build_node_reward_probe(
            trainer,
            _FakeAgentManager(),
            messages=[],
            response_text="Paris</conclusion>",
            response_prefix="<conclusion>",
        )

        self.assertEqual(probe.batch["prompts"].tolist(), [[10, 11, 20, 21]])
        self.assertEqual(probe.batch["responses"].tolist(), [[30, 31, 32, 0, 0]])
        self.assertEqual(probe.batch["response_mask"].tolist(), [[1, 1, 1, 0, 0]])
        self.assertEqual(
            probe.batch["input_ids"].tolist(), [[10, 11, 20, 21, 30, 31, 32, 0, 0]]
        )

    def test_open_tag_stays_inside_fixed_full_prompt_budget(self):
        trainer = SimpleNamespace(
            config=OmegaConf.create(
                {"data": {"max_prompt_length": 4, "max_response_length": 5}}
            ),
            tokenizer=_FakeTokenizer(),
        )
        probe = RayPPOTrainer._build_node_reward_probe(
            trainer,
            _FullPromptAgentManager(),
            messages=[],
            response_text="Paris</conclusion>",
            response_prefix="<conclusion>",
        )

        self.assertEqual(probe.batch["prompts"].shape, (1, 4))
        self.assertEqual(probe.batch["prompts"].tolist(), [[12, 13, 20, 21]])
        self.assertEqual(probe.batch["input_ids"].shape, (1, 9))

    def test_attribution_requests_reference_without_enabling_kl(self):
        config = OmegaConf.create(
            {
                "algorithm": {"use_kl_in_reward": False},
                "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
                "qa_manager": {
                    "node_reward": {
                        "enabled": True,
                        "search_attribution_enabled": True,
                        "summary_attribution_enabled": True,
                    }
                },
            }
        )
        self.assertTrue(model_based_node_rewards_enabled(config))
        self.assertFalse(config.algorithm.use_kl_in_reward)
        self.assertFalse(config.actor_rollout_ref.actor.use_kl_loss)


if __name__ == "__main__":
    unittest.main()
