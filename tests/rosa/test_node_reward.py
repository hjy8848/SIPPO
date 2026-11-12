import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from qa_manager.ROSA import attribution_reward as ATTRIBUTION_REWARD


ROOT = Path(__file__).parents[2]


def _load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NODE_REWARD = _load_module("node_reward_impl", "verl/trainer/ppo/node_reward.py")
WORKFLOW_NODE_RLOO = _load_module(
    "workflow_node_rloo_impl", "verl/trainer/ppo/workflow_node_rloo.py"
)
for package_name in ("verl", "verl.trainer", "verl.trainer.ppo"):
    package = sys.modules.setdefault(package_name, types.ModuleType(package_name))
    package.__path__ = []
sys.modules["verl.trainer.ppo.workflow_node_rloo"] = WORKFLOW_NODE_RLOO
WORKFLOW_NODE_GRPO = _load_module(
    "workflow_node_grpo_impl", "verl/trainer/ppo/workflow_node_grpo.py"
)


def _estimator_kwargs(row_count):
    return {
        "outcome_exact": np.zeros(row_count, dtype=np.float32),
        "node_reward_weights": np.full(row_count, 0.2, dtype=np.float32),
        "node_reward_max_rms_ratios": np.full(row_count, 0.3, dtype=np.float32),
        "node_reward_failed_positive_scales": np.full(row_count, 0.2, dtype=np.float32),
        "node_reward_all_equal_caps": np.full(row_count, 0.1, dtype=np.float32),
    }


def test_gold_title_delta_matches_real_titles_and_rejects_numeric_ids():
    docs = [
        '[1_1]: (Title: "Hill of Tara") Hill of Tara is an ancient site.',
        '[1_2]: (Title: Other Page) unrelated text',
    ]

    delta, covered, available = NODE_REWARD.gold_title_coverage_delta(
        docs, ["Cathair Mor", "Hill of Tara"]
    )
    assert available is True
    assert delta == pytest.approx(0.5)
    assert "hill of tara" in covered

    delta, covered, available = NODE_REWARD.gold_title_coverage_delta(docs, ["814145", "7672"])
    assert available is False
    assert delta == 0.0
    assert covered == set()


def test_information_gain_and_binary_judge_are_bounded_and_directional():
    assert 0 < NODE_REWARD.information_gain_score(-1.0, -2.0) < 1
    assert -1 < NODE_REWARD.information_gain_score(-3.0, -2.0) < 0

    score, confidence = NODE_REWARD.binary_judge_score(-0.1, -3.0)
    assert score > 0.8
    assert confidence > 0.9


def test_calibrated_information_gain_uses_fixed_threshold():
    score, margin = NODE_REWARD.calibrated_information_gain_score(
        -2.0, -1.0, threshold=-1.5
    )
    assert margin == pytest.approx(-1.0)
    assert score > 0


def test_causal_validated_launcher_uses_all_nodes_and_only_audited_rewards():
    script = (
        ROOT / "exp/run_dagent_sft_rl_essential_rewards_2x_a100.sh"
    ).read_text(encoding="utf-8")
    base = (ROOT / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(encoding="utf-8")

    assert "causal_validated)" in script
    assert "NODE_REWARD_SEARCH_IG_ENABLED=true" in script
    assert "NODE_REWARD_SUBGRAPH_HCAPO_ENABLED=true" in script
    assert "NODE_REWARD_MAX_SEARCH_NODES=0" in script
    assert "NODE_REWARD_QUERY_REWRITE_ENABLED=false" in script
    assert '+qa_manager.node_reward.subgraph_hcapo_enabled=' in base
    assert '+qa_manager.node_reward.subgraph_hcapo_threshold=' in base


def test_thresholded_attribution_reward_is_positive_only_and_reports_margin():
    reward, margin = NODE_REWARD.thresholded_attribution_reward(-0.5, -1.5, threshold=0.5)
    assert margin == pytest.approx(1.0)
    assert 0.0 < reward < 1.0

    reward, margin = NODE_REWARD.thresholded_attribution_reward(-1.2, -1.5, threshold=0.5)
    assert margin == pytest.approx(0.3)
    assert reward == 0.0

    with pytest.raises(ValueError, match="temperature"):
        NODE_REWARD.thresholded_attribution_reward(-0.5, -1.5, threshold=0.5, temperature=0)


def test_attribution_prompts_and_targets_match_runtime_protocol():
    assert NODE_REWARD.thresholded_attribution_reward is ATTRIBUTION_REWARD.thresholded_attribution_reward
    assert NODE_REWARD.select_matched_counterfactual_docs is ATTRIBUTION_REWARD.select_matched_counterfactual_docs
    assert ATTRIBUTION_REWARD.SEARCH_MATCHED_ATTRIBUTION_THRESHOLD == pytest.approx(0.5653623938560486)
    assert ATTRIBUTION_REWARD.SEARCH_EMPTY_ATTRIBUTION_THRESHOLD == pytest.approx(0.6487774848937988)
    assert ATTRIBUTION_REWARD.SUMMARY_ATTRIBUTION_THRESHOLD == pytest.approx(0.8756318092346191)
    search = NODE_REWARD.build_search_attribution_messages(
        "search-system", "Who is Ada?", ["doc one", "doc two"]
    )
    assert search == [
        {"role": "system", "content": "search-system"},
        {
            "role": "user",
            "content": "**Current Sub-question**: Who is Ada?\n\n**Search Context**:\ndoc one\ndoc two",
        },
    ]
    summary = NODE_REWARD.build_summary_attribution_messages(
        "summary-system", "Where?", "Task 1\nConclusion: Paris\n"
    )
    assert summary[1]["content"].endswith("Task 1\nConclusion: Paris")
    assert NODE_REWARD.tagged_attribution_target("conclusion", "  Paris ") == (
        "<conclusion>",
        "Paris</conclusion>",
    )
    assert NODE_REWARD.tagged_attribution_target("answer", "yes") == (
        "<answer>",
        "yes</answer>",
    )


def test_verified_gold_search_requires_both_new_gold_evidence_and_grounded_conclusion():
    docs = ['[1_1]: (Title: "Walter Mischel") Walter Mischel was an Austrian-born psychologist.']

    assert NODE_REWARD.verified_search_reward(0.5, "Walter Mischel", docs) == pytest.approx(0.5)
    assert NODE_REWARD.verified_search_reward(0.5, "Albert Einstein", docs) == 0.0
    assert NODE_REWARD.verified_search_reward(0.0, "Walter Mischel", docs) == 0.0


def test_verified_reason_requires_merge_dependencies_and_gold_exact_output():
    assert NODE_REWARD.merge_reason_required("comparison") is True
    assert NODE_REWARD.merge_reason_required("bridge-comparison") is True
    assert NODE_REWARD.merge_reason_required("bridge") is False
    assert NODE_REWARD.verified_reason_reward("no", ["No"], ["1", "2"], True) == 1.0
    assert NODE_REWARD.verified_reason_reward("yes", ["No"], ["1", "2"], True) == 0.0
    assert NODE_REWARD.verified_reason_reward("no", ["No"], ["1"], True) == 0.0
    assert NODE_REWARD.verified_reason_reward("no", ["No"], ["1", "2"], False) == 0.0


def test_verified_planner_hindsight_combines_coverage_and_required_reason():
    assert NODE_REWARD.verified_planner_hindsight([0.5, 0.5], [], False) == 1.0
    assert NODE_REWARD.verified_planner_hindsight([0.5, 0.5], [1.0], True) == 1.0
    assert NODE_REWARD.verified_planner_hindsight([0.5, 0.5], [0.0], True) == 0.5
    assert (
        NODE_REWARD.verified_planner_hindsight(
            [0.5, 0.5],
            [0.0],
            True,
            reason_reward_enabled=False,
        )
        == 1.0
    )
    assert NODE_REWARD.outcome_gate(1.0, 0.0, enabled=True) == 0.0
    assert NODE_REWARD.outcome_gate(1.0, 1.0, enabled=True) == 1.0


def test_runtime_prefers_retrievable_gold_titles():
    trainer_source = (
        ROOT / "verl/trainer/ppo/ray_trainer.py"
    ).read_text(encoding="utf-8")
    assert 'extra_info.get("retrievable_golden_titles"' in trainer_source
    assert 'extra_info.get("golden_titles", [])' in trainer_source


def test_em_gold_launcher_enables_only_deterministic_gold_reward():
    script = (
        ROOT / "exp/run_dagent_mixed_rl_em_gold_3x_a6000.sh"
    ).read_text(encoding="utf-8")
    assert "FINAL_REWARD_EM_WEIGHT=1.0" in script
    assert "NODE_REWARD_GOLD_EVIDENCE_ENABLED=true" in script
    assert "NODE_REWARD_VERIFIED_REASON_ENABLED=false" in script
    assert "NODE_REWARD_SEMANTIC_CRITIC_ENABLED=false" in script
    assert "NODE_REWARD_SEARCH_ATTRIBUTION_ENABLED=false" in script
    assert "gold_enriched_wiki18_top3.parquet" in script
    assert 'ADV_ESTIMATOR="${ADV_ESTIMATOR:-workflow_node_rloo}"' in script

    grpo_script = (
        ROOT / "exp/run_dagent_2wiki_hotpot_rl_em_gold_workflow_grpo_3x_a6000.sh"
    ).read_text(encoding="utf-8")
    assert "ADV_ESTIMATOR=workflow_node_grpo" in grpo_script
    assert "rl_train_mixed_2wiki_hotpot_seed42_gold_enriched_wiki18_top3.parquet" in grpo_script
    assert "NODE_REWARD_VERIFIED_REASON_ENABLED=false" in grpo_script


def test_counterfactual_docs_are_same_dataset_and_size_matched():
    steps = [
        {"question": "target", "data_source": "hotpotqa", "docs": ["a" * 10, "b" * 10]},
        {"question": "cross", "data_source": "musique", "docs": ["x" * 20, "y" * 20]},
        {"question": "long", "data_source": "hotpotqa", "docs": ["z" * 1000]},
        {"question": "matched", "data_source": "hotpotqa", "docs": ["m" * 11, "n" * 11]},
    ]

    docs, source = NODE_REWARD.select_matched_counterfactual_docs(steps, 0)
    assert source == "same_dataset"
    assert docs == steps[3]["docs"]

    docs, source = NODE_REWARD.select_matched_counterfactual_docs(
        [{"question": "only", "data_source": "2wiki", "docs": ["real"]}], 0
    )
    assert source == "no_evidence"
    assert docs == NODE_REWARD.NO_EVIDENCE_DOCS


def test_counterfactual_docs_do_not_contain_the_target_output():
    steps = [
        {
            "question": "target",
            "data_source": "2wiki",
            "docs": ["target evidence"],
            "output": "London",
        },
        {
            "question": "leaky",
            "data_source": "2wiki",
            "docs": ["London is the capital of England."],
        },
        {
            "question": "clean",
            "data_source": "2wiki",
            "docs": ["Paris is the capital of France."],
        },
    ]

    docs, source = NODE_REWARD.select_matched_counterfactual_docs(steps, 0)
    assert source == "same_dataset"
    assert docs == steps[2]["docs"]


def test_workflow_node_rloo_uses_local_signal_when_all_outcomes_tie():
    row_count = 2
    response_mask = torch.ones((row_count, 3), dtype=torch.float32)
    kwargs = _estimator_kwargs(row_count)

    advantages, returns = WORKFLOW_NODE_RLOO.compute_workflow_node_rloo_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["q", "q"], dtype=object),
        trajectory_ids=np.array(["a", "b"], dtype=object),
        outcome_rewards=np.array([0.0, 0.0], dtype=np.float32),
        node_rewards=np.array([1.0, -1.0], dtype=np.float32),
        node_reward_types=np.array(["search_ig", "search_ig"], dtype=object),
        **kwargs,
    )

    assert torch.allclose(advantages[:, 0], torch.tensor([0.1, -0.1]), atol=1e-5)
    assert torch.allclose(advantages, returns)


def test_workflow_node_rloo_keeps_float_advantages_with_integer_response_mask():
    row_count = 2
    response_mask = torch.ones((row_count, 3), dtype=torch.long)
    kwargs = _estimator_kwargs(row_count)

    advantages, returns = WORKFLOW_NODE_RLOO.compute_workflow_node_rloo_advantage(
        token_level_rewards=torch.zeros((row_count, 3), dtype=torch.float32),
        response_mask=response_mask,
        index=np.array(["q", "q"], dtype=object),
        trajectory_ids=np.array(["a", "b"], dtype=object),
        outcome_rewards=np.array([0.0, 0.0], dtype=np.float32),
        node_rewards=np.array([1.0, -1.0], dtype=np.float32),
        node_reward_types=np.array(["search_ig", "search_ig"], dtype=object),
        **kwargs,
    )

    assert advantages.dtype == torch.float32
    assert returns.dtype == torch.float32
    assert torch.allclose(advantages[:, 0], torch.tensor([0.1, -0.1]), atol=1e-5)


def test_invalid_action_penalty_survives_direct_rl_cold_start():
    row_count = 2
    response_mask = torch.ones((row_count, 2), dtype=torch.float32)
    kwargs = _estimator_kwargs(row_count)
    kwargs["node_reward_weights"] = np.full(row_count, 0.1, dtype=np.float32)

    advantages, _ = WORKFLOW_NODE_RLOO.compute_workflow_node_rloo_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["q", "q"], dtype=object),
        trajectory_ids=np.array(["a", "b"], dtype=object),
        outcome_rewards=np.zeros(row_count, dtype=np.float32),
        node_rewards=np.full(row_count, -1.0, dtype=np.float32),
        node_reward_types=np.array(["invalid_action"] * row_count, dtype=object),
        **kwargs,
    )

    assert torch.allclose(advantages[:, 0], torch.tensor([-0.1, -0.1]), atol=1e-6)


def test_workflow_node_rloo_keeps_global_outcome_dominant():
    row_count = 4
    response_mask = torch.ones((row_count, 2), dtype=torch.float32)
    kwargs = _estimator_kwargs(row_count)

    advantages, _ = WORKFLOW_NODE_RLOO.compute_workflow_node_rloo_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["q", "q", "q", "q"], dtype=object),
        trajectory_ids=np.array(["a", "a", "b", "b"], dtype=object),
        outcome_rewards=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        node_rewards=np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float32),
        node_reward_types=np.array(["search_ig"] * row_count, dtype=object),
        **kwargs,
    )

    # Global RLOO is +/-1. Local RMS is capped to 30% of global RMS, so it
    # cannot reverse either workflow's outcome sign.
    assert torch.all(advantages[:2, 0] > 0)
    assert torch.all(advantages[2:, 0] < 0)
    assert torch.sqrt(torch.mean((advantages[:, 0] - torch.tensor([1.0, 1.0, -1.0, -1.0])) ** 2)) <= 0.3001


def test_workflow_node_grpo_normalizes_complete_workflows_not_agent_rows():
    response_mask = torch.ones((8, 2), dtype=torch.float32)
    outcomes = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    advantages, returns = WORKFLOW_NODE_GRPO.compute_workflow_node_grpo_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["q"] * 8, dtype=object),
        trajectory_ids=np.array(
            ["good_a", "good_a", "good_b", "good_b", "bad_a", "bad_a", "bad_b", "bad_b"],
            dtype=object,
        ),
        outcome_rewards=outcomes,
        outcome_exact=outcomes,
        node_rewards=np.zeros(8, dtype=np.float32),
        node_reward_types=np.array(["none"] * 8, dtype=object),
        node_reward_weights=np.zeros(8, dtype=np.float32),
        node_reward_max_rms_ratios=np.full(8, 0.3, dtype=np.float32),
        node_reward_failed_positive_scales=np.full(8, 0.2, dtype=np.float32),
        node_reward_all_equal_caps=np.full(8, 0.1, dtype=np.float32),
    )

    expected = torch.tensor(
        [0.8660, 0.8660, 0.8660, 0.8660, -0.8660, -0.8660, -0.8660, -0.8660]
    )
    assert torch.allclose(advantages[:, 0], expected, atol=1e-3)
    assert torch.allclose(advantages, returns)


def test_workflow_node_rloo_rejects_inconsistent_outcomes_within_workflow():
    kwargs = _estimator_kwargs(3)
    with pytest.raises(ValueError, match="inconsistent outcome rewards"):
        WORKFLOW_NODE_RLOO.compute_workflow_node_rloo_advantage(
            token_level_rewards=torch.zeros((3, 2)),
            response_mask=torch.ones((3, 2)),
            index=np.array(["q", "q", "q"], dtype=object),
            trajectory_ids=np.array(["a", "a", "b"], dtype=object),
            outcome_rewards=np.array([1.0, 0.5, 0.0], dtype=np.float32),
            node_rewards=np.zeros(3, dtype=np.float32),
            node_reward_types=np.array(["none"] * 3, dtype=object),
            **kwargs,
        )


def test_node_reward_launcher_enables_hierarchical_training():
    script = (ROOT / "run_rosa_train_node_reward.sh").read_text(encoding="utf-8")
    assert "export RESULT_REWARD_ONLY=false" in script
    assert "export NODE_REWARD_ENABLED=true" in script
    assert "export ADV_ESTIMATOR=workflow_node_rloo" in script
    assert "export USE_KL_LOSS=true" in script
    assert "NODE_REWARD_SEARCH_IG_ENABLED" in script
    assert "NODE_REWARD_INVALID_ACTION_WEIGHT" in script
    assert "NODE_REWARD_SEMANTIC_CRITIC_ENABLED" in script
    assert "NODE_REWARD_GOLD_TITLE_WEIGHT" not in script

    base_script = (ROOT / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(encoding="utf-8")
    assert "rl_train_2wiki_no_sft_seed42.parquet" in base_script
    assert "validation_2wiki_heldout.parquet" in base_script
    assert '+qa_manager.node_reward.enabled="${NODE_REWARD_ENABLED}"' in base_script
    assert '+qa_manager.node_reward.local_advantage_weight="${NODE_REWARD_LOCAL_WEIGHT}"' in base_script
    assert '+qa_manager.node_reward.invalid_action_weight="${NODE_REWARD_INVALID_ACTION_WEIGHT}"' in base_script
    assert '+qa_manager.node_reward.summary_penalty_enabled="${NODE_REWARD_SUMMARY_PENALTY_ENABLED}"' in base_script
    assert '+qa_manager.node_reward.gold_evidence_enabled="${NODE_REWARD_GOLD_EVIDENCE_ENABLED}"' in base_script
    assert '+qa_manager.node_reward.verified_reason_enabled="${NODE_REWARD_VERIFIED_REASON_ENABLED}"' in base_script
    assert '+qa_manager.node_reward.search_attribution_enabled="${NODE_REWARD_SEARCH_ATTRIBUTION_ENABLED}"' in base_script
    assert '+qa_manager.node_reward.summary_attribution_enabled="${NODE_REWARD_SUMMARY_ATTRIBUTION_ENABLED}"' in base_script
    assert '+qa_manager.node_reward.search_attribution_threshold="${NODE_REWARD_SEARCH_ATTRIBUTION_THRESHOLD}"' in base_script
    assert '+qa_manager.node_reward.summary_attribution_threshold="${NODE_REWARD_SUMMARY_ATTRIBUTION_THRESHOLD}"' in base_script
    assert "0.6487774848937988" in base_script
    assert 'trainer.logger="${TRAINER_LOGGER}"' in base_script
    assert 'trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"' in base_script
    assert "gold_title_weight" not in base_script

    validated_script = (
        ROOT / "exp/run_dagent_mixed_rl_validated_rewards_2x_a100.sh"
    ).read_text(encoding="utf-8")
    assert "gold_enriched.parquet" in validated_script
    assert "DAGENT_REWARD_MODE=validated" in validated_script
    assert "NODE_REWARD_QUERY_REWRITE_ENABLED=false" in (
        ROOT / "exp/run_dagent_sft_rl_essential_rewards_2x_a100.sh"
    ).read_text(encoding="utf-8")
    assert '"query_gold_delta"' not in (
        ROOT / "verl/trainer/ppo/ray_trainer.py"
    ).read_text(encoding="utf-8")
    assert "TRAINER_LOGGER='[\"console\"]'" in (
        ROOT / "exp/run_dagent_verified_reward_smoke_1gpu.sh"
    ).read_text(encoding="utf-8")

    audited_script = (
        ROOT / "exp/run_dagent_mixed_rl_audited_attribution_2x_a100.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_REWARD_MODE=audited_attribution" in audited_script
    assert "USE_KL_LOSS=false" in audited_script
    shared_script = (
        ROOT / "exp/run_dagent_sft_rl_essential_rewards_2x_a100.sh"
    ).read_text(encoding="utf-8")
    assert "NODE_REWARD_SEARCH_ATTRIBUTION_ENABLED=true" in shared_script
    assert "NODE_REWARD_SUMMARY_ATTRIBUTION_ENABLED=true" in shared_script
    assert "NODE_REWARD_SEMANTIC_CRITIC_ENABLED=false" in shared_script

    attribution_only_script = (
        ROOT / "exp/run_dagent_mixed_rl_attribution_only_2x_a100.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_REWARD_MODE=attribution_only" in attribution_only_script
    assert "USE_KL_LOSS=false" in attribution_only_script
    assert "NODE_REWARD_GOLD_EVIDENCE_ENABLED=false" in shared_script


def test_blackwell_summary_attribution_launcher_has_exactly_one_local_reward():
    script = (
        ROOT / "exp/run_dagent_mixed_rl_em_summary_attribution_2x_blackwell.sh"
    ).read_text(encoding="utf-8")
    assert "rosa-blackwell-cu128" in script
    assert 'export FINAL_REWARD_EM_WEIGHT=1.0' in script
    assert 'export NODE_REWARD_SUMMARY_ATTRIBUTION_ENABLED=true' in script
    assert 'export NODE_REWARD_SEARCH_ATTRIBUTION_ENABLED=false' in script
    assert 'export NODE_REWARD_GOLD_EVIDENCE_ENABLED=false' in script
    assert 'export NODE_REWARD_VERIFIED_REASON_ENABLED=false' in script
    assert 'export NODE_REWARD_SEMANTIC_CRITIC_ENABLED=false' in script
    assert 'export NODE_REWARD_SEARCH_IG_ENABLED=false' in script
    assert 'export NODE_REWARD_QUERY_REWRITE_ENABLED=false' in script
    assert 'export NODE_REWARD_INVALID_ACTION_WEIGHT=0.0' in script
    assert 'export TOTAL_TRAINING_STEPS=1' in script
    assert 'export PPO_MAX_TOKEN_LEN_PER_GPU=8192' in script
    assert 'export ROLLOUT_GPU_MEMORY_UTILIZATION=0.20' in script
    assert 'export SAVE_FREQ=-1' in script
    assert 'Configuration validated; training was not started.' in script


def test_essential_step650_eval_matches_outcome_only_protocol():
    script = (
        ROOT
        / "exp/run_dagent_mixed_essential_step650_multibench_2000_3090_fast.sh"
    ).read_text(encoding="utf-8")

    assert 'CHECKPOINT_STEP="${CHECKPOINT_STEP:-650}"' in script
    assert "DAGENT-Mixed-3B-SFT-RL-OutcomeEM-EssentialRewards-2xA100" in script
    assert 'NUM_SAMPLES="${NUM_SAMPLES:-2000}"' in script
    assert 'CONCURRENCY="${CONCURRENCY:-64}"' in script
    assert 'MEM_UTIL="${MEM_UTIL:-0.92}"' in script
    assert 'VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"' in script
    assert 'VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"' in script
    assert 'exec bash "${SCRIPT_DIR}/run_dagent_mixed_outcome_multibench_500.sh"' in script


def test_48gb_smoke_has_one_step_and_one_local_reward():
    script = (
        ROOT
        / "exp/run_dagent_mixed_rl_em_summary_attribution_smoke_2x48gb.sh"
    ).read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES="${DAGENT_CUDA_VISIBLE_DEVICES}"' in script
    assert '"RTX A6000"' in script
    assert '"RTX 4090"' in script
    assert 'export FINAL_REWARD_EM_WEIGHT=1.0' in script
    assert 'export NODE_REWARD_SUMMARY_ATTRIBUTION_ENABLED=true' in script
    assert 'export NODE_REWARD_SEARCH_ATTRIBUTION_ENABLED=false' in script
    assert 'export NODE_REWARD_GOLD_EVIDENCE_ENABLED=false' in script
    assert 'export NODE_REWARD_VERIFIED_REASON_ENABLED=false' in script
    assert 'export NODE_REWARD_SEMANTIC_CRITIC_ENABLED=false' in script
    assert 'export NODE_REWARD_INVALID_ACTION_WEIGHT=0.0' in script
    assert 'export TOTAL_TRAINING_STEPS=1' in script
    assert 'export TEST_FREQ=-1' in script
    assert 'export SAVE_FREQ=-1' in script
    assert 'export CLEAN_STALE_PROCESSES=false' in script


def test_six_a6000_causal_launcher_scales_verified_per_gpu_load():
    script = (
        ROOT
        / "exp/run_dagent_2wiki_hotpot_rl_causal_validated_6x_a6000.sh"
    ).read_text(encoding="utf-8")

    assert 'DAGENT_CUDA_VISIBLE_DEVICES:-1,2,3,5,6,7' in script
    assert 'export NUM_GPUS=6' in script
    assert 'export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"' in script
    assert 'export WORKFLOW_ROLLOUT_N="${WORKFLOW_ROLLOUT_N:-4}"' in script
    assert 'export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-96}"' in script
    assert 'export NODE_REWARD_SEARCH_IG_ENABLED=true' in script
    assert 'export NODE_REWARD_SUBGRAPH_HCAPO_ENABLED=true' in script
    assert 'export NODE_REWARD_QUERY_REWRITE_ENABLED=false' in script
    assert 'export NODE_REWARD_GOLD_EVIDENCE_ENABLED=false' in script
    assert 'export FINAL_REWARD_EM_WEIGHT=1.0' in script
    assert 'export CLEAN_STALE_PROCESSES=false' in script


def test_a6000_summary_attribution_full_launcher_matches_outcome_batch():
    script = (
        ROOT
        / "exp/run_dagent_mixed_rl_em_summary_attribution_2x_a6000.sh"
    ).read_text(encoding="utf-8")

    assert 'DAGENT_CUDA_VISIBLE_DEVICES' in script
    assert '"RTX A6000"' in script
    assert 'export FINAL_REWARD_EM_WEIGHT=1.0' in script
    assert 'export NODE_REWARD_SUMMARY_ATTRIBUTION_ENABLED=true' in script
    assert 'export NODE_REWARD_SEARCH_ATTRIBUTION_ENABLED=false' in script
    assert 'export NODE_REWARD_GOLD_EVIDENCE_ENABLED=false' in script
    assert 'export NODE_REWARD_VERIFIED_REASON_ENABLED=false' in script
    assert 'export NODE_REWARD_SEMANTIC_CRITIC_ENABLED=false' in script
    assert 'export NODE_REWARD_INVALID_ACTION_WEIGHT=0.0' in script
    assert 'export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"' in script
    assert 'export WORKFLOW_ROLLOUT_N="${WORKFLOW_ROLLOUT_N:-4}"' in script
    assert 'export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"' in script
    assert 'export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"' in script
    assert 'export TEST_FREQ="${TEST_FREQ:-50}"' in script
    assert 'export SAVE_FREQ="${SAVE_FREQ:-50}"' in script
    assert 'export TRAINER_LOGGER="${TRAINER_LOGGER:-[\\"console\\",\\"wandb\\"]}"' in script
    assert 'export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/dg_${USER}_a6}"' in script
    assert 'unset PYTORCH_CUDA_ALLOC_CONF' in script
    assert 'expandable_segments:True' not in script
    assert 'export CLEAN_STALE_PROCESSES=false' in script
