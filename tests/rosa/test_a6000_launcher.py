from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_a6000_launcher_has_safe_direct_rl_defaults():
    script = (ROOT / "run_rosa_train_4x_a6000_direct_rl.sh").read_text(encoding="utf-8")

    assert "export NUM_GPUS=4" in script
    assert 'CUDA_VISIBLE_DEVICES:-0,1,2,3' in script
    assert 'Qwen2.5-3B-Instruct' in script
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60' in script
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55' in script
    assert 'ROLLOUT_MAX_NUM_BATCHED_TOKENS:-49152' in script
    assert 'WORKFLOW_CONCURRENCY:-48' in script
    assert 'RETRIEVAL_CONCURRENCY:-48' in script
    assert 'TEST_FREQ:-200' in script
    assert 'SAVE_FREQ:-40' in script
    assert 'WORKFLOW_ROLLOUT_N:-4' in script
    assert 'REWARD_MODE:-outcome' in script
    assert "run_rosa_train_accuracy_first.sh" in script
    assert "run_rosa_train_node_reward.sh" in script
    assert "Retriever preflight failed" in script
    assert 'RESUME_MODE:-disable' in script


def test_six_a6000_launcher_scales_batches_by_data_parallelism():
    script = (ROOT / "run_rosa_train_6x_a6000_direct_rl.sh").read_text(encoding="utf-8")

    assert "export NUM_GPUS=6" in script
    assert 'CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5' in script
    assert 'TRAIN_BATCH_SIZE:-24' in script
    assert 'PPO_MINI_BATCH_SIZE:-96' in script
    assert 'TRAIN_BATCH_SIZE:-12' in script
    assert 'PPO_MINI_BATCH_SIZE:-48' in script
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30' in script
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION:-0.28' in script
    assert 'ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384' in script
    assert 'WORKFLOW_ROLLOUT_N:-4' in script
    assert 'REWARD_MODE:-outcome' in script
    assert "run_rosa_train_accuracy_first.sh" in script
    assert "run_rosa_train_node_reward.sh" in script
    assert "Retriever preflight failed" in script


def test_base_launcher_offloads_reference_policy_by_default():
    script = (ROOT / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(encoding="utf-8")

    assert 'REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-true}"' in script
    assert 'REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"' in script
    assert 'actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}"' in script
    assert 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"' in script
