from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_common_launcher_exposes_all_planner_ccw_controls():
    text = (ROOT / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(encoding="utf-8")
    required = {
        "algorithm.planner_ccw.mode",
        "algorithm.planner_ccw.probe_samples",
        "algorithm.planner_ccw.replay_samples_per_plan",
        "algorithm.planner_ccw.replacement_attempts",
        "algorithm.planner_ccw.probe_concurrency",
        "algorithm.planner_ccw.seed",
        "algorithm.planner_ccw.noise_z",
        "algorithm.planner_ccw.minimum_effect",
        "algorithm.planner_ccw.audit_only",
        "algorithm.planner_ccw.alpha",
        "algorithm.planner_ccw.minimum_weight",
        "algorithm.planner_ccw.maximum_weight",
    }
    assert all(value in text for value in required)
    assert "Planner CCW requires pure workflow-RLOO exact-match training" in text
    assert 'PLANNER_CCW_NOISE_Z="${PLANNER_CCW_NOISE_Z:-1.0}"' in text


def test_all_planner_ccw_wrappers_use_paired_soft_shrink_default():
    wrappers = (ROOT / "exp").glob("run_dagent_planner_ccw_*.sh")
    configured = []
    for wrapper in wrappers:
        text = wrapper.read_text(encoding="utf-8")
        if "PLANNER_CCW_NOISE_Z" not in text:
            continue
        configured.append(wrapper.name)
        assert 'PLANNER_CCW_NOISE_Z="${PLANNER_CCW_NOISE_Z:-1.0}"' in text
    assert configured


def test_planner_ccw_wrapper_defaults_to_the_audited_step20_checkpoint():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju18_gpu34.sh").read_text(
        encoding="utf-8"
    )
    assert "PlannerCCW-${MODE}" in text
    assert "r3_step20" in text
    assert 'PLANNER_CCW_PROBE_SAMPLES:-1' in text


def test_zju12_wrapper_excludes_gpu_zero_and_uses_three_way_fsdp():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju12_gpu123.sh").read_text(
        encoding="utf-8"
    )
    assert 'DAGENT_CUDA_VISIBLE_DEVICES:-1,2,3' in text
    assert "DAGENT_SUPPORTED_GPU_PATTERN='A100'" in text
    assert 'gpu_label="${DAGENT_CUDA_VISIBLE_DEVICES//,/}"' in text
    assert 'ACTOR_OPTIMIZER_OFFLOAD:-false' in text
    assert '/dev/shm/pccw-' in text


def test_zju46_wrapper_uses_only_free_a6000_gpu4_with_offload():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju46_gpu4.sh").read_text(
        encoding="utf-8"
    )
    assert "DAGENT_CUDA_VISIBLE_DEVICES=4" in text
    assert "ACTOR_PARAM_OFFLOAD=true" in text
    assert "ACTOR_OPTIMIZER_OFFLOAD=true" in text
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION:-0.08' in text


def test_zju56_wrapper_uses_two_unshared_3090s_with_deterministic_offloaded_suffixes():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju56_gpu01.sh").read_text(
        encoding="utf-8"
    )
    assert 'DAGENT_CUDA_VISIBLE_DEVICES:-0,1' in text
    assert "DAGENT_SUPPORTED_GPU_PATTERN='RTX 3090'" in text
    assert "DAGENT_ALLOW_SHARED_GPU=false" in text
    assert "ACTOR_PARAM_OFFLOAD=true" in text
    assert "ACTOR_OPTIMIZER_OFFLOAD=true" in text
    assert "MODEL_ATTN_IMPLEMENTATION=sdpa" in text
    assert "ACTOR_USE_REMOVE_PADDING=false" in text
    assert 'ROLLOUT_ENFORCE_EAGER:-true' in text
    assert 'PLANNER_CCW_PROBE_CONCURRENCY:-1' in text


def test_zju53_wrapper_is_single_gpu_noise_aware_audit_without_optimizer_update():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju53_gpu5.sh").read_text(
        encoding="utf-8"
    )
    assert "DAGENT_CUDA_VISIBLE_DEVICES=5" in text
    assert "MODEL_ATTN_IMPLEMENTATION=sdpa" in text
    assert "ACTOR_USE_REMOVE_PADDING=false" in text
    assert "PLANNER_CCW_REPLAY_SAMPLES_PER_PLAN" in text
    assert "PLANNER_CCW_PROBE_CONCURRENCY=1" in text
    assert "PLANNER_CCW_AUDIT_ONLY=true" in text
    assert "TOTAL_TRAINING_STEPS=1" in text


def test_zju12_audit_wrapper_uses_only_the_free_pcie_gpu3():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju12_gpu3_audit.sh").read_text(
        encoding="utf-8"
    )
    assert "DAGENT_CUDA_VISIBLE_DEVICES=3" in text
    assert "A100-PCIE-40GB" in text
    assert "ACTOR_PARAM_OFFLOAD=true" in text
    assert "ACTOR_OPTIMIZER_OFFLOAD=true" in text
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20' in text
    assert "PLANNER_CCW_PROBE_CONCURRENCY=1" in text
    assert "PLANNER_CCW_AUDIT_ONLY=true" in text
    assert "TOTAL_TRAINING_STEPS=1" in text


def test_zju46_audit_wrapper_uses_only_free_gpu3_with_noise_aware_config():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju46_gpu3_audit.sh").read_text(
        encoding="utf-8"
    )
    assert "DAGENT_CUDA_VISIBLE_DEVICES=3" in text
    assert "DAGENT_SUPPORTED_GPU_PATTERN='RTX A6000'" in text
    assert "DAGENT_MINIMUM_FREE_MIB=47000" in text
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20' in text
    assert "PLANNER_CCW_REPLAY_SAMPLES_PER_PLAN" in text
    assert "PLANNER_CCW_PROBE_CONCURRENCY=1" in text
    assert 'PLANNER_CCW_AUDIT_ONLY:-true' in text


def test_zju46_two_gpu_update_requires_two_empty_a6000s_and_enables_optimizer():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju46_gpu23_update.sh").read_text(
        encoding="utf-8"
    )
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2,3" in text
    assert "DAGENT_SUPPORTED_GPU_PATTERN='RTX A6000'" in text
    assert "DAGENT_ALLOW_SHARED_GPU=false" in text
    assert "DAGENT_MINIMUM_FREE_MIB=47000" in text
    assert "ACTOR_PARAM_OFFLOAD=false" in text
    assert "ACTOR_OPTIMIZER_OFFLOAD=false" in text
    assert "PLANNER_CCW_PROBE_CONCURRENCY=1" in text
    assert "PLANNER_CCW_AUDIT_ONLY=false" in text


def test_zju18_two_gpu_update_records_explicit_shared_a100_approval():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju18_gpu01_update.sh").read_text(
        encoding="utf-8"
    )
    assert "DAGENT_CUDA_VISIBLE_DEVICES=0,1" in text
    assert "DAGENT_SUPPORTED_GPU_PATTERN='A100 80GB PCIe'" in text
    assert "DAGENT_ALLOW_SHARED_GPU=true" in text
    assert "DAGENT_MINIMUM_FREE_MIB=65536" in text
    assert "ACTOR_PARAM_OFFLOAD=false" in text
    assert "ACTOR_OPTIMIZER_OFFLOAD=false" in text
    assert "PLANNER_CCW_PROBE_CONCURRENCY=1" in text
    assert "PLANNER_CCW_AUDIT_ONLY=false" in text


def test_common_unified_launcher_preserves_explicit_shared_gpu_approval():
    text = (
        ROOT / "exp" / "run_dagent_unified_sft_pure_em_old1500_reason578_zju57_gpu23.sh"
    ).read_text(encoding="utf-8")
    assert 'DAGENT_ALLOW_SHARED_GPU="${DAGENT_ALLOW_SHARED_GPU:-false}"' in text


def test_zju18_formal_ccw_is_a_matched_90_step_run_from_unified_sft():
    text = (ROOT / "exp" / "run_dagent_planner_ccw_zju18_gpu01_formal90.sh").read_text(
        encoding="utf-8"
    )
    assert "DAGENT_CUDA_VISIBLE_DEVICES=0,1" in text
    assert "DAGENT_ALLOW_SHARED_GPU=true" in text
    assert 'DAGENT_MINIMUM_FREE_MIB="${DAGENT_MINIMUM_FREE_MIB:-45000}"' in text
    assert "DAGENT-UnifiedOrigSFT-Causal114-MatchedExposure-fromBase" in text
    assert "r3_step20" not in text
    assert "TOTAL_TRAINING_STEPS=90" in text
    assert "TOTAL_EPOCHS=2" in text
    assert "SAVE_FREQ=10" in text
    assert "TEST_FREQ=90" in text
    assert 'PLANNER_CCW_PROBE_CONCURRENCY="${PLANNER_CCW_PROBE_CONCURRENCY:-2}"' in text
    assert "PLANNER_CCW_AUDIT_ONLY=false" in text

    common = (ROOT / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(encoding="utf-8")
    assert 'TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"' in common
    assert 'trainer.total_epochs="${TOTAL_EPOCHS}"' in common


def test_zju47_mixed15k_formal_ccw_uses_old_sft_and_two_empty_a6000s():
    text = (
        ROOT
        / "exp"
        / "run_dagent_planner_ccw_mixed15k_old_sft_zju47_gpu56_formal468.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=5,6" in text
    assert "DAGENT_SUPPORTED_GPU_PATTERN='RTX A6000'" in text
    assert "DAGENT_ALLOW_SHARED_GPU=false" in text
    assert "DAGENT_MINIMUM_FREE_MIB=47000" in text
    assert "rl_train_mixed_2wiki40_hotpot30_musique30_seed42.parquet" in text
    assert "Qwen2.5-3B-DeepSeek800-1xA6000-bf16/global_step_510" in text
    assert "TRAIN_BATCH_SIZE=32" in text
    assert "WORKFLOW_ROLLOUT_N=4" in text
    assert "PPO_MINI_BATCH_SIZE=128" in text
    assert "TOTAL_TRAINING_STEPS=468" in text
    assert "TOTAL_EPOCHS=1" in text
    assert 'PLANNER_CCW_PROBE_CONCURRENCY="${PLANNER_CCW_PROBE_CONCURRENCY:-2}"' in text
    assert "PLANNER_CCW_AUDIT_ONLY=false" in text
    assert "CLEAN_STALE_PROCESSES=false" in text


def test_planner_ccw_randomizes_only_the_replacement_plan_and_greedily_replays_suffixes():
    trainer = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
        encoding="utf-8"
    )
    worker = (
        ROOT / "verl" / "experimental" / "agent_loop" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "downstream_sampling_temperature=None" in trainer
    assert "planning_data_ij = run_inference(planning_agent_dict_, 'PlanningAgent')" in trainer
    assert trainer.count("temperature=downstream_sampling_temperature") == 5
    assert trainer.count("downstream_sampling_temperature=0.0") == 4
    assert "repeated_factual_metrics" in trainer
    assert "compare_identity_replay" in trainer
    assert "if replay_index == 0:" in trainer
    assert trainer.index("repeated_factual_metrics") < trainer.index(
        "replayed_alternative_metrics"
    )
    assert "ThreadPoolExecutor(max_workers=probe_concurrency)" in trainer
    assert "executor.submit(run_probe, row) for row in probe_candidates" in trainer
    assert "for future in as_completed(futures):" in trainer
    assert "if probe_concurrency not in {1, 2}:" in trainer
    assert "requires probe_concurrency in {1, 2}" in trainer
    assert "effective_sampling_temperature" in trainer
    assert "downstream sampling temperature was not applied" in trainer
    assert 'record_metrics["action_trace"] = action_trace' in trainer
    assert 'sampling_params["temperature"] = float(sampling_temperature)' in worker
    assert 'sampling_params["top_p"] = 1.0' in worker
    assert 'output.extra_fields["effective_sampling_temperature"]' in worker
