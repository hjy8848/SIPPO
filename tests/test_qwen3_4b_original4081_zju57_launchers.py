from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def read(name: str) -> str:
    return (EXP / name).read_text(encoding="utf-8")


def test_zju57_core_freezes_the_single_gpu_original4081_pureem_contract():
    script = read("run_dagent_qwen3_4b_original4081_matched_zju57_gpu0_r1.sh")
    for value in (
        '"zju-57"',
        "DAGENT_CUDA_VISIBLE_DEVICES=0",
        "GPU-1a61c7d2-ff37-3b2a-d14b-02fea9d1254b",
        "Original4081SFT-NoThink-step510-A100-zju12-gpu23-r1",
        "TRAIN_BATCH_SIZE=32",
        "WORKFLOW_ROLLOUT_N=4",
        "PPO_MINI_BATCH_SIZE=128",
        "PLANNER_POC_MODE=off",
        "FINAL_REWARD_F1_TIEBREAK_ENABLED=false",
        "CP_DAGENT_ENABLED=true",
        "PLANNER_PROMPT_MAX_NODES=5",
        "MAX_NODES_PER_TURN=0",
        "MODEL_ATTN_IMPLEMENTATION=flash_attention_2",
        "ACTOR_USE_REMOVE_PADDING=true",
        "CRITIC_USE_REMOVE_PADDING=true",
        "ACTOR_PARAM_OFFLOAD=true",
        "ACTOR_OPTIMIZER_OFFLOAD=true",
        "TOTAL_EPOCHS=1",
        "TEST_FREQ=-1",
    ):
        assert value in script
    for forbidden in ("pkill", "ray stop", "rm -", "SIGKILL"):
        assert forbidden not in script


def test_zju57_smoke_and_formal_wrappers_are_disjoint_and_matched():
    smoke = read("run_dagent_qwen3_4b_original4081_pureem_smoke2_zju57_gpu0_r1.sh")
    formal = read("run_dagent_qwen3_4b_original4081_pureem_full468_zju57_gpu0_r1.sh")

    assert "TOTAL_TRAINING_STEPS=2" in smoke
    assert "q34o-pe-smk-g0-r1" in smoke
    assert "TOTAL_TRAINING_STEPS=468" in formal
    assert "SAVE_FREQ=50" in formal
    assert "q34o-pe-468-g0-r1" in formal
    assert "smoke2" not in formal
    for script in (smoke, formal):
        assert "run_dagent_qwen3_4b_original4081_matched_zju57_gpu0_r1.sh" in script


def test_zju57_handoff_waits_then_requires_fa2_smoke_before_formal():
    core = read("run_dagent_qwen3_4b_original4081_matched_zju57_gpu0_r1.sh")
    handoff = read("handoff_dagent_qwen3_4b_original4081_pureem_zju57_gpu0_r1.sh")
    wait = read("wait_dagent_qwen3_4b_original4081_pureem_zju57_gpu0_r1.sh")

    for value in (
        "validate_dagent_training_smoke.py",
        "preflight_lab_runtime.py",
        "--phase claim",
        "--gpu 0=GPU-1a61c7d2-ff37-3b2a-d14b-02fea9d1254b",
        "--expected-step 2",
        "--expected-workflows 128",
        "--expected-poc-mode off",
    ):
        assert value in handoff
    assert "check_flash_attn_cuda.py" in core
    assert "MODEL_ATTN_IMPLEMENTATION=flash_attention_2" in core
    assert handoff.index("validate_dagent_training_smoke.py") < handoff.index(
        "starting the matched 468-step"
    )
    for value in (
        "WAIT_LAUNCH_AUTHORIZED=true",
        "WAIT_GPU_INDEX=0",
        "GPU-1a61c7d2-ff37-3b2a-d14b-02fea9d1254b",
        "wait_for_exclusive_gpu_then_exec_v2.sh",
    ):
        assert value in wait


def test_zju57_r2_formal_requires_the_preserved_r1_smoke_gate():
    formal = read("run_dagent_qwen3_4b_original4081_pureem_full468_zju57_gpu0_r2.sh")
    launch = read("launch_dagent_qwen3_4b_original4081_pureem_formal_zju57_gpu0_r2.sh")

    assert "TOTAL_TRAINING_STEPS=468" in formal
    assert "q34o-pe-468-g0-r2" in formal
    for value in (
        "validate_dagent_training_smoke.py",
        "--expected-step 2",
        "--expected-workflows 128",
        "--expected-poc-mode off",
        "preflight_lab_runtime.py",
        "--phase claim",
        "step468-zju57-gpu0-r2",
    ):
        assert value in launch
    assert launch.index("validate_dagent_training_smoke.py") < launch.index(
        "preflight_lab_runtime.py"
    )
