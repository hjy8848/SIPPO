from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "exp" / "run_dagent_qwen3_17b_planner_poc_formal90.sh"


def test_qwen3_poc_core_freezes_scientific_contract():
    script = CORE.read_text(encoding="utf-8")
    required = [
        "DAGENT-Qwen3-1.7B-UnifiedOrigSFT-NoThink-step560",
        "MODEL_ENABLE_THINKING=false",
        "ADV_ESTIMATOR=workflow_rloo",
        "RESULT_REWARD_ONLY=true",
        "PLANNER_CCW_MODE=off",
        'PLANNER_POC_MODE="${PLANNER_POC_MODE:-real}"',
        'PLANNER_POC_MINIBATCH_MODE="${PLANNER_POC_MINIBATCH_MODE:-question_preserving}"',
        "ACTOR_LOSS_AGG_MODE=token-mean",
        'PLANNER_POC_LOSS_WEIGHT="${PLANNER_POC_LOSS_WEIGHT:-0.02}"',
        "PLANNER_POC_GRADIENT_AUDIT=false",
        'PLANNER_POC_GROUPED_WEIGHT_AUDITED="${PLANNER_POC_GROUPED_WEIGHT_AUDITED:-false}"',
        "Refusing grouped POC long run before a <=2-step gradient-scale audit.",
        'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"',
        'WORKFLOW_ROLLOUT_N="${WORKFLOW_ROLLOUT_N:-4}"',
        'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-96}"',
        'ACTOR_LR="${ACTOR_LR:-5e-7}"',
        "TOTAL_TRAINING_STEPS=1",
        "TOTAL_TRAINING_STEPS=\"${TOTAL_TRAINING_STEPS:-90}\"",
        "SAVE_FREQ=\"${SAVE_FREQ:-10}\"",
        "TEST_FREQ=\"${TEST_FREQ:-90}\"",
        "CLEAN_STALE_PROCESSES=false",
    ]
    for value in required:
        assert value in script
    assert "ray stop" not in script
    assert "pkill" not in script


def test_qwen3_poc_profiles_are_fast_and_hardware_specific():
    script = CORE.read_text(encoding="utf-8")
    for profile in (
        "blackwell1",
        "a6000-1",
        "a100-1",
        "a10080-1",
        "a800-1",
        "a100-2",
        "ada4090-2",
        "ampere-2",
        "ampere-4",
    ):
        assert profile in script
    assert "rosa-blackwell-cu128" in script
    assert "rosa-vllm010" in script
    assert "MODEL_ATTN_IMPLEMENTATION=flash_attention_2" in script
    assert "import flash_attn" in script
    assert "ROLLOUT_ENFORCE_EAGER=true" in script
    assert "ENABLE_GRADIENT_CHECKPOINTING=true" in script
    assert "ACTOR_CALCULATE_OLD_LOG_PROB_ENTROPY=false" in script
    assert "PPO_MAX_TOKEN_LEN_PER_GPU=24576" in script
    assert "ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768" in script


def test_blackwell_96gb_profile_uses_verified_fast_settings():
    script = CORE.read_text(encoding="utf-8")
    blackwell = script.split("blackwell1)", 1)[1].split("a6000-1)", 1)[0]
    high_memory = blackwell.split("if (( total_mib >= 90000 )); then", 1)[1].split(
        "else", 1
    )[0]
    for value in (
        "ENABLE_GRADIENT_CHECKPOINTING=false",
        "ROLLOUT_ENFORCE_EAGER=false",
        "PPO_MICRO_BATCH_SIZE_PER_GPU=8",
        "PPO_MAX_TOKEN_LEN_PER_GPU=16384",
        "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8",
        "REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4",
        "ROLLOUT_GPU_MEMORY_UTILIZATION=0.22",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768",
        "ROLLOUT_MAX_NUM_SEQS=64",
        "WORKFLOW_CONCURRENCY=64",
        "RETRIEVAL_CONCURRENCY=48",
    ):
        assert value in high_memory
    for value in (
        "MODEL_ATTN_IMPLEMENTATION=flash_attention_2",
        "ACTOR_USE_REMOVE_PADDING=true",
        "CRITIC_USE_REMOVE_PADDING=true",
        "ACTOR_TRIM_PADDED_SEQUENCES=false",
    ):
        assert value in blackwell


def test_a100_40gb_pair_profile_uses_fast_memory_safe_settings():
    script = CORE.read_text(encoding="utf-8")
    profile = script.split("a100-2)", 1)[1].split("ampere-2|ampere-4)", 1)[0]
    for value in (
        "A100-PCIE-40GB",
        "ENABLE_GRADIENT_CHECKPOINTING=true",
        "ROLLOUT_ENFORCE_EAGER=false",
        "PPO_MICRO_BATCH_SIZE_PER_GPU=1",
        "PPO_MAX_TOKEN_LEN_PER_GPU=8192",
        "ROLLOUT_GPU_MEMORY_UTILIZATION=0.20",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384",
        "WORKFLOW_CONCURRENCY=32",
        "RETRIEVAL_CONCURRENCY=24",
    ):
        assert value in profile


def test_a100_40gb_single_profile_is_registered_for_mechanism_smokes():
    script = CORE.read_text(encoding="utf-8")
    profile = script.split("a100-1)", 1)[1].split("ampere-2|ampere-4", 1)[0]
    for value in (
        "A100-PCIE-40GB",
        "ACTOR_OPTIMIZER_OFFLOAD=true",
        "ENABLE_GRADIENT_CHECKPOINTING=true",
        "PPO_MICRO_BATCH_SIZE_PER_GPU=1",
        "PPO_MAX_TOKEN_LEN_PER_GPU=8192",
        "ROLLOUT_GPU_MEMORY_UTILIZATION=0.20",
        "WORKFLOW_CONCURRENCY=16",
    ):
        assert value in profile


def test_ada4090_pair_profile_uses_fast_p2p_safe_settings():
    script = CORE.read_text(encoding="utf-8")
    profile = script.split("ada4090-2)", 1)[1].split("ampere-2|ampere-4", 1)[0]
    for value in (
        "GeForce RTX 4090",
        "DAGENT_NCCL_P2P_DISABLE=1",
        "ENABLE_GRADIENT_CHECKPOINTING=false",
        "PPO_MICRO_BATCH_SIZE_PER_GPU=2",
        "PPO_MAX_TOKEN_LEN_PER_GPU=16384",
        "ROLLOUT_GPU_MEMORY_UTILIZATION=0.30",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS=24576",
        "WORKFLOW_CONCURRENCY=32",
        "RETRIEVAL_CONCURRENCY=24",
    ):
        assert value in profile


def test_a100_80gb_single_profile_uses_fast_settings():
    script = CORE.read_text(encoding="utf-8")
    profile = script.split("a10080-1)", 1)[1].split("a800-1)", 1)[0]
    for value in (
        "A100 80GB PCIe",
        "ENABLE_GRADIENT_CHECKPOINTING=false",
        "PPO_MICRO_BATCH_SIZE_PER_GPU=4",
        "PPO_MAX_TOKEN_LEN_PER_GPU=16384",
        "ROLLOUT_GPU_MEMORY_UTILIZATION=0.22",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768",
        "WORKFLOW_CONCURRENCY=48",
        "RETRIEVAL_CONCURRENCY=32",
    ):
        assert value in profile


def test_a800_single_profile_uses_fast_settings():
    script = CORE.read_text(encoding="utf-8")
    profile = script.split("a800-1)", 1)[1].split("ada4090-2)", 1)[0]
    for value in (
        "A800-SXM4-80GB",
        "DAGENT_MINIMUM_FREE_MIB=79000",
        "ENABLE_GRADIENT_CHECKPOINTING=false",
        "PPO_MICRO_BATCH_SIZE_PER_GPU=4",
        "PPO_MAX_TOKEN_LEN_PER_GPU=16384",
        "ROLLOUT_GPU_MEMORY_UTILIZATION=0.22",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768",
        "WORKFLOW_CONCURRENCY=48",
        "RETRIEVAL_CONCURRENCY=32",
    ):
        assert value in profile


def test_blackwell_speed_smoke_is_one_step_and_disables_recomputation():
    script = CORE.read_text(encoding="utf-8")
    smoke = script.split('speed_smoke="${QWEN3_POC_SPEED_SMOKE:-false}"', 1)[1]
    for value in (
        "ENABLE_GRADIENT_CHECKPOINTING=false",
        "ROLLOUT_ENFORCE_EAGER=false",
        'PPO_MAX_TOKEN_LEN_PER_GPU="${QWEN3_POC_SMOKE_TOKEN_BUDGET:-32768}"',
        "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8",
        "ROLLOUT_MAX_NUM_SEQS=64",
        "WORKFLOW_CONCURRENCY=64",
        "RETRIEVAL_CONCURRENCY=48",
        "TOTAL_TRAINING_STEPS=1",
        "SAVE_FREQ=-1",
    ):
        assert value in smoke


def test_topology_wrappers_delegate_to_frozen_core():
    wrappers = {
        "run_dagent_qwen3_17b_planner_poc_1x_blackwell.sh": "blackwell1",
        "run_dagent_qwen3_17b_planner_poc_1x_a6000.sh": "a6000-1",
        "run_dagent_qwen3_17b_planner_poc_2x_ampere.sh": "ampere-2",
        "run_dagent_qwen3_17b_planner_poc_4x_ampere.sh": "ampere-4",
    }
    for filename, profile in wrappers.items():
        text = (ROOT / "exp" / filename).read_text(encoding="utf-8")
        assert f"QWEN3_POC_PROFILE={profile}" in text
        assert "run_dagent_qwen3_17b_planner_poc_formal90.sh" in text


def test_base_launcher_forwards_no_thinking_and_checks_compute_pids():
    base = (ROOT / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(encoding="utf-8")
    elastic = (ROOT / "exp" / "run_dagent_outcome_grpo_elastic_gpu.sh").read_text(encoding="utf-8")
    assert "MODEL_ENABLE_THINKING" in base
    assert "+data.apply_chat_template_kwargs.enable_thinking=" in base
    assert "--query-compute-apps=pid" in elastic
    assert "already has compute PID(s)" in elastic

    unified = (
        ROOT / "exp" / "run_dagent_unified_sft_pure_em_old1500_reason578_zju57_gpu23.sh"
    ).read_text(encoding="utf-8")
    for key in (
        "PPO_MICRO_BATCH_SIZE_PER_GPU",
        "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU",
        "REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU",
        "PPO_MAX_TOKEN_LEN_PER_GPU",
    ):
        assert f'export {key}="${{{key}:-' in unified


def test_sft_to_poc_handoff_requires_complete_checkpoint_and_empty_gpu():
    handoff = (ROOT / "exp" / "handoff_qwen3_17b_sft_to_poc_zju53_gpu5.sh").read_text(
        encoding="utf-8"
    )
    for value in (
        "step:560 - train/loss:",
        "Final validation metrics:",
        "tokenizer_config.json",
        "*.safetensors",
        "--query-compute-apps=pid",
        "run_dagent_qwen3_17b_planner_poc_1x_blackwell.sh",
    ):
        assert value in handoff
    assert "ray stop" not in handoff
    assert "pkill" not in handoff


def test_mixed15k_wrapper_freezes_classic_full_run_contract():
    script = (
        ROOT / "exp" / "run_dagent_qwen3_17b_planner_poc_mixed15k_formal468.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "rl_train_mixed_2wiki40_hotpot30_musique30_seed42.parquet",
        "validation_mixed_2wiki_hotpot_musique_seed42.parquet",
        "TRAIN_BATCH_SIZE=32",
        "WORKFLOW_ROLLOUT_N=4",
        "PPO_MINI_BATCH_SIZE=128",
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-468}"',
        'TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"',
        'SAVE_FREQ="${SAVE_FREQ:-50}"',
        'TEST_FREQ="${TEST_FREQ:-468}"',
        'PLANNER_POC_MODE="${PLANNER_POC_MODE:-real}"',
        'PLANNER_POC_MINIBATCH_MODE="${PLANNER_POC_MINIBATCH_MODE:-question_preserving}"',
        "ACTOR_LOSS_AGG_MODE=token-mean",
        'PLANNER_POC_LOSS_WEIGHT="${PLANNER_POC_LOSS_WEIGHT:-0.02}"',
        "PLANNER_CCW_MODE=off",
        "MODEL_ENABLE_THINKING=false",
    ):
        assert value in script


def test_grouped_poc_audit_is_two_steps_on_original4081_sft():
    script = (
        ROOT / "exp" / "run_dagent_qwen3_17b_grouped_poc_audit2_zju12_gpu3.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "DAGENT-Qwen3-1.7B-Original4081SFT-NoThink-step510-a100-1-zju-12-gpu3-r1-fa2",
        "DAGENT_CUDA_VISIBLE_DEVICES=3",
        "QWEN3_POC_PROFILE=a100-1",
        "TOTAL_TRAINING_STEPS=2",
        "PLANNER_POC_MODE=real",
        "PLANNER_POC_MINIBATCH_MODE=question_preserving",
        "PLANNER_POC_PAIRED_SHUFFLED_AUDIT=true",
        "PLANNER_POC_GROUPED_WEIGHT_AUDITED=false",
        "SAVE_FREQ=-1",
        "TEST_FREQ=-1",
    ):
        assert value in script
    assert "pkill" not in script
    assert "ray stop" not in script


def test_zju18_grouped_long_runs_are_matched_except_for_real_poc():
    pure = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_17b_grouped_pure_em_original4081_mixed15k_formal468_zju18_gpu23.sh"
    ).read_text(encoding="utf-8")
    real = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_17b_grouped_real_poc_original4081_mixed15k_formal468_zju18_gpu45.sh"
    ).read_text(encoding="utf-8")
    shared = (
        "Original4081SFT-NoThink-step510",
        "QWEN3_POC_PROFILE=ada4090-2",
        "TOTAL_TRAINING_STEPS=468",
        "TOTAL_EPOCHS=1",
        "SAVE_FREQ=50",
        "TEST_FREQ=468",
        "PLANNER_POC_MINIBATCH_MODE=question_preserving",
        "PLANNER_POC_PAIRED_SHUFFLED_AUDIT=false",
        "MODEL_ENABLE_THINKING=false",
        "run_dagent_qwen3_17b_planner_poc_mixed15k_formal468.sh",
    )
    for value in shared:
        assert value in pure
        assert value in real
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2,3" in pure
    assert "PLANNER_POC_MODE=off" in pure
    assert "DAGENT_CUDA_VISIBLE_DEVICES=4,5" in real
    assert "PLANNER_POC_MODE=real" in real
    assert "PLANNER_POC_LOSS_WEIGHT=0.02" in real
    assert "PLANNER_POC_GROUPED_WEIGHT_AUDITED=true" in real
    for script in (pure, real):
        assert "pkill" not in script
        assert "ray stop" not in script


def test_zju12_grouped_pure_em_long_run_uses_audited_framework():
    script = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_17b_grouped_pure_em_original4081_mixed15k_formal468_zju12_gpu3.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "Original4081SFT-NoThink-step510",
        "DAGENT_CUDA_VISIBLE_DEVICES=3",
        "QWEN3_POC_PROFILE=a100-1",
        "TOTAL_TRAINING_STEPS=468",
        "TOTAL_EPOCHS=1",
        "SAVE_FREQ=50",
        "TEST_FREQ=468",
        "PLANNER_POC_MODE=off",
        "PLANNER_POC_MINIBATCH_MODE=question_preserving",
        "PLANNER_POC_PAIRED_SHUFFLED_AUDIT=false",
        "MODEL_ENABLE_THINKING=false",
        "run_dagent_qwen3_17b_planner_poc_mixed15k_formal468.sh",
    ):
        assert value in script
    assert "pkill" not in script
    assert "ray stop" not in script


def test_zju12_real_poc_resume300_skips_only_empty_auxiliary_signal():
    script = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_17b_grouped_real_poc_unlimited_original4081_mixed15k_formal468_zju12_gpu13_r3_resume300.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "DAGENT_CUDA_VISIBLE_DEVICES=1,3",
        "TOTAL_TRAINING_STEPS=468",
        "SAVE_FREQ=50",
        "TEST_FREQ=468",
        "PLANNER_POC_MODE=real",
        "PLANNER_POC_MINIBATCH_MODE=question_preserving",
        "PLANNER_POC_LOSS_WEIGHT=0.02",
        "PLANNER_POC_REQUIRE_SIGNAL=false",
        "PLANNER_POC_GROUPED_WEIGHT_AUDITED=true",
        "RESUME_MODE=resume_path",
        "global_step_300",
        "model_world_size_2_rank_0.pt",
        "model_world_size_2_rank_1.pt",
        "optim_world_size_2_rank_0.pt",
        "optim_world_size_2_rank_1.pt",
        "extra_state_world_size_2_rank_0.pt",
        "extra_state_world_size_2_rank_1.pt",
        "r3-resume300",
    ):
        assert value in script
    assert "pkill" not in script
    assert "ray stop" not in script


def test_zju18_a100_80gb_real_poc_long_run_uses_audited_framework():
    script = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_17b_grouped_real_poc_original4081_mixed15k_formal468_zju18_gpu0.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "Original4081SFT-NoThink-step510",
        "DAGENT_CUDA_VISIBLE_DEVICES=0",
        "QWEN3_POC_PROFILE=a10080-1",
        "TOTAL_TRAINING_STEPS=468",
        "SAVE_FREQ=50",
        "TEST_FREQ=468",
        "PLANNER_POC_MODE=real",
        "PLANNER_POC_MINIBATCH_MODE=question_preserving",
        "PLANNER_POC_LOSS_WEIGHT=0.02",
        "PLANNER_POC_PAIRED_SHUFFLED_AUDIT=false",
        "PLANNER_POC_GROUPED_WEIGHT_AUDITED=true",
        "MODEL_ENABLE_THINKING=false",
    ):
        assert value in script
    assert "pkill" not in script
    assert "ray stop" not in script


def test_zju57_grouped_long_runs_are_matched_single_gpu_blackwell_jobs():
    pure = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_17b_grouped_pure_em_original4081_mixed15k_formal468_zju57_gpu0.sh"
    ).read_text(encoding="utf-8")
    real = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_17b_grouped_real_poc_original4081_mixed15k_formal468_zju57_gpu2.sh"
    ).read_text(encoding="utf-8")
    shared = (
        "Original4081SFT-NoThink-step510",
        "QWEN3_POC_PROFILE=blackwell1",
        "rosa-blackwell-cu128",
        "TOTAL_TRAINING_STEPS=468",
        "TOTAL_EPOCHS=1",
        "SAVE_FREQ=50",
        "TEST_FREQ=468",
        "PLANNER_POC_MINIBATCH_MODE=question_preserving",
        "PLANNER_POC_PAIRED_SHUFFLED_AUDIT=false",
        "MODEL_ENABLE_THINKING=false",
        "run_dagent_qwen3_17b_planner_poc_mixed15k_formal468.sh",
    )
    for value in shared:
        assert value in pure
        assert value in real
    assert "DAGENT_CUDA_VISIBLE_DEVICES=0" in pure
    assert "PLANNER_POC_MODE=off" in pure
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2" in real
    assert "PLANNER_POC_MODE=real" in real
    assert "PLANNER_POC_LOSS_WEIGHT=0.02" in real
    assert "PLANNER_POC_GROUPED_WEIGHT_AUDITED=true" in real
    for script in (pure, real):
        assert "pkill" not in script
        assert "ray stop" not in script


def test_zju53_gpu2_grouped_real_poc_uses_memory_safe_blackwell_profile():
    script = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_17b_grouped_real_poc_original4081_mixed15k_formal468_zju53_gpu2.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "Original4081SFT-NoThink-step510",
        "DAGENT_CUDA_VISIBLE_DEVICES=2",
        "QWEN3_POC_PROFILE=blackwell1",
        "rosa-blackwell-cu128",
        "TOTAL_TRAINING_STEPS=468",
        "SAVE_FREQ=50",
        "TEST_FREQ=468",
        "PLANNER_POC_MODE=real",
        "PLANNER_POC_MINIBATCH_MODE=question_preserving",
        "PLANNER_POC_LOSS_WEIGHT=0.02",
        "PLANNER_POC_GROUPED_WEIGHT_AUDITED=true",
        "MODEL_ENABLE_THINKING=false",
    ):
        assert value in script
    assert "pkill" not in script
    assert "ray stop" not in script


def test_core_launcher_preserves_mixed15k_wrapper_contract():
    script = (
        ROOT
        / "exp"
        / "run_dagent_unified_sft_pure_em_old1500_reason578_zju57_gpu23.sh"
    ).read_text(encoding="utf-8")
    for value in (
        'TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DEFAULT_TRAIN_DATA_PATH}}"',
        'TEST_DATA_PATH="${TEST_DATA_PATH:-${DEFAULT_TEST_DATA_PATH}}"',
        'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"',
        'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-96}"',
    ):
        assert value in script


def test_curve_to_mixed15k_handoff_requires_all_evaluations_and_empty_gpu():
    script = (
        ROOT / "exp" / "handoff_qwen3_poc_curve_to_mixed15k.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "global_step_90/data.pt",
        "formal_test_summary.json",
        "metadata_joined",
        "pipeline_error_rate",
        "for step in range(10, 91, 10)",
        "--query-compute-apps=gpu_uuid,pid",
        "gpu_is_free",
        "run_dagent_qwen3_17b_planner_poc_mixed15k_formal468.sh",
    ):
        assert value in script
    assert "pkill" not in script
    assert "ray stop" not in script


def test_poc_to_shuffled_handoff_freezes_matched_control_and_gpu_ownership():
    script = (
        ROOT / "exp" / "handoff_qwen3_17b_poc_to_shuffled100_zju57_gpu0.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "global_step_468",
        "training/global_step:468",
        "Final validation metrics:",
        "EXPECTED_SOURCE_SHA",
        "awk '{print $1}' | sha256sum",
        "gpu0_is_free",
        "--query-compute-apps=gpu_uuid,pid",
        "PLANNER_POC_MODE=shuffled",
        "PLANNER_POC_LOSS_WEIGHT=0.02",
        "TOTAL_TRAINING_STEPS=100",
        "SAVE_FREQ=50",
        "TEST_FREQ=100",
        "run_dagent_qwen3_17b_planner_poc_mixed15k_formal468.sh",
    ):
        assert value in script
    assert "pkill" not in script
    assert "ray stop" not in script
