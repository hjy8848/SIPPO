from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def read(name: str) -> str:
    return (EXP / name).read_text(encoding="utf-8")


def test_a800_matched_core_freezes_runtime_and_training_contract():
    script = read("run_dagent_qwen3_4b_matched_a800_repl_r21.sh")
    for value in (
        "38lmdgu9bdc9m-0",
        "NVIDIA A800-SXM4-80GB",
        "DAGENT-Qwen3-4B-UnifiedSFT-NoThink-step560-zju53-gpu6-r1",
        "flash-attn-2.8.3-torch2.7-cu12-a800-r1",
        "gcc12-a800-repl-r1",
        "python311-dev-a800-r1",
        "MODEL_ATTN_IMPLEMENTATION=flash_attention_2",
        "ACTOR_USE_REMOVE_PADDING=true",
        "CRITIC_USE_REMOVE_PADDING=true",
        "TRAIN_BATCH_SIZE",
        "PLANNER_PROMPT_MAX_NODES=5",
        "MAX_NODES_PER_TURN=0",
        "ROLLOUT_GPU_MEMORY_UTILIZATION=0.22",
        "ACTOR_OPTIMIZER_OFFLOAD=true",
    ):
        assert value in script
    for forbidden in ("pkill", "ray stop", "rm -", "SIGKILL"):
        assert forbidden not in script


def test_a800_smoke_and_formal_wrappers_are_matched_and_disjoint():
    pure_smoke = read("run_dagent_qwen3_4b_pureem_smoke2_a800_repl_gpu01_fa2_r21.sh")
    psc_smoke = read("run_dagent_qwen3_4b_psc_smoke2_a800_repl_gpu45_fa2_r21.sh")
    pure_full = read("run_dagent_qwen3_4b_pureem_full468_a800_repl_gpu01_fa2_r21.sh")
    psc_full = read("run_dagent_qwen3_4b_psc_full468_a800_repl_gpu45_fa2_r21.sh")

    assert "A800_GPU_IDS=0,1" in pure_smoke
    assert "A800_GPU_IDS=4,5" in psc_smoke
    assert "PLANNER_POC_MODE=off" in pure_smoke
    assert "PLANNER_POC_MODE=real" in psc_smoke
    assert "TOTAL_TRAINING_STEPS=2" in pure_smoke
    assert "TOTAL_TRAINING_STEPS=2" in psc_smoke
    assert "TOTAL_TRAINING_STEPS=468" in pure_full
    assert "TOTAL_TRAINING_STEPS=468" in psc_full
    assert "SAVE_FREQ=50" in pure_full
    assert "SAVE_FREQ=50" in psc_full

    for script in (pure_smoke, psc_smoke, pure_full, psc_full):
        assert "run_dagent_qwen3_4b_matched_a800_repl_r21.sh" in script
        for forbidden in ("pkill", "ray stop", "rm -", "SIGKILL"):
            assert forbidden not in script


def test_a800_psc_recovery_launcher_allows_zero_pair_em_updates():
    core = read("run_dagent_qwen3_4b_matched_a800_repl_r22.sh")
    recovery = read("run_dagent_qwen3_4b_psc_full468_a800_repl_gpu45_fa2_r22.sh")

    assert '${PLANNER_POC_REQUIRE_SIGNAL:?' in core
    assert "PLANNER_POC_REQUIRE_SIGNAL=false" in recovery
    assert "PLANNER_POC_MODE=real" in recovery
    assert "step468-a800-repl-gpu45-fa2-r22" in recovery
    assert "q34psc-468-g45-r22" in recovery
    assert "run_dagent_qwen3_4b_matched_a800_repl_r22.sh" in recovery


def test_a800_runtime_profile_uses_only_the_fresh_replacement_environment():
    profiles = (ROOT / "scripts" / "lab_runtime_profiles.json").read_text(
        encoding="utf-8"
    )
    assert '"dagent-qwen3-a800-replacement-rl"' in profiles
    assert '"38lmdgu9bdc9m-0"' in profiles
    assert "rosa-vllm010-a800-repl-r1" in profiles
    assert "flash-attn-2.8.3-torch2.7-cu12-a800-r1" in profiles


def test_a800_handoff_hard_gates_both_smokes_before_formal_launch():
    script = read("handoff_qwen3_4b_matched_a800_repl_r21.sh")
    assert script.count("validate_dagent_training_smoke.py") == 3
    assert "--expected-poc-mode off" in script
    assert "--expected-poc-mode real" in script
    assert "expected 399 tensors" in script
    assert "retriever did not return exactly top-3" in script
    assert "dagent-qwen3-a800-replacement-rl" in script
    assert "NCCL all_reduce/barrier succeeded" in script
    assert "nohup setsid bash" in script
    assert script.index("Both exact-config smoke2 gates passed") < script.index(
        'nohup setsid bash "${PURE_FULL_LAUNCHER}"'
    )
    for forbidden in ("pkill", "ray stop", "rm -", "SIGKILL"):
        assert forbidden not in script
