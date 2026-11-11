import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def read(name: str) -> str:
    return (EXP / name).read_text(encoding="utf-8")


def test_memory_safe_r2_preserves_scientific_contract():
    script = read("run_dagent_qwen3_4b_original4081_legacy_matched_zju47_gpu56_r2.sh")
    required = (
        "DAGENT-Qwen3-4B-Original4081SFT-NoThink-step510-A100-zju12-gpu23-r1",
        "rl_train_mixed_2wiki40_hotpot30_musique30_seed42.parquet",
        "CP_DAGENT_ENABLED=false",
        "ADV_ESTIMATOR=workflow_rloo",
        "FINAL_REWARD_EM_WEIGHT=1.0",
        "FINAL_REWARD_F1_TIEBREAK_ENABLED=false",
        "TRAIN_BATCH_SIZE=32",
        "WORKFLOW_ROLLOUT_N=4",
        "PPO_MINI_BATCH_SIZE=64",
        "ACTOR_LR=5e-7",
        "ROLLOUT_TEMPERATURE=0.8",
        "ROLLOUT_TOP_P=0.95",
        "RETRIEVAL_TOPK=3",
    )
    for value in required:
        assert value in script


def test_memory_safe_r2_is_the_only_execution_change():
    script = read("run_dagent_qwen3_4b_original4081_legacy_matched_zju47_gpu56_r2.sh")
    for value in (
        "ENABLE_GRADIENT_CHECKPOINTING=true",
        "ACTOR_PARAM_OFFLOAD=true",
        "ACTOR_OPTIMIZER_OFFLOAD=true",
        "PPO_MICRO_BATCH_SIZE_PER_GPU=1",
        "PPO_MAX_TOKEN_LEN_PER_GPU=6144",
        "ROLLOUT_GPU_MEMORY_UTILIZATION=0.20",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288",
        "ROLLOUT_MAX_NUM_SEQS=12",
    ):
        assert value in script
    for forbidden in ("F1_TIEBREAK_ENABLED=true", "PLANNER_POC_MODE=on", "CP_DAGENT_ENABLED=true"):
        assert forbidden not in script


def test_r2_smoke_and_formal_use_disjoint_artifacts():
    smoke = read("run_dagent_qwen3_4b_original4081_legacy_matched_smoke2_zju47_gpu56_r2.sh")
    formal = read("run_dagent_qwen3_4b_original4081_legacy_matched_full468_zju47_gpu56_r2.sh")
    assert "TOTAL_TRAINING_STEPS=2" in smoke
    assert "q34leg-smk-g56-r2" in smoke
    assert "TOTAL_TRAINING_STEPS=468" in formal
    assert "q34leg-468-g56-r2" in formal
    assert "smoke2" not in formal.split("EXPERIMENT_NAME=", 1)[1]


def test_zju47_dual_a6000_runtime_profile_is_frozen():
    profiles = json.loads(
        (ROOT / "scripts" / "lab_runtime_profiles.json").read_text(encoding="utf-8")
    )["profiles"]
    profile = profiles["dagent-qwen3-zju47-rl-dual"]
    assert profile["status"] == "ready"
    assert profile["allowed_hosts"] == ["zju-47"]
    assert profile["gpu_name_pattern"] == "NVIDIA RTX A6000"
    assert "flash_attn" in profile["python_imports"]


def test_r3_corrects_the_global_minibatch_contract_without_other_scientific_changes():
    r2 = read("run_dagent_qwen3_4b_original4081_legacy_matched_zju47_gpu56_r2.sh")
    r3 = read("run_dagent_qwen3_4b_original4081_legacy_matched_zju47_gpu56_r3.sh")
    assert "PPO_MINI_BATCH_SIZE=64" in r2
    assert "PPO_MINI_BATCH_SIZE=128" in r3
    for value in (
        "TRAIN_BATCH_SIZE=32",
        "WORKFLOW_ROLLOUT_N=4",
        "ACTOR_LR=5e-7",
        "ADV_ESTIMATOR=workflow_rloo",
        "FINAL_REWARD_F1_TIEBREAK_ENABLED=false",
        "PLANNER_POC_MODE=off",
        "CP_DAGENT_ENABLED=false",
    ):
        assert value in r2
        assert value in r3


def test_r3_smoke_and_formal_use_fresh_artifacts():
    smoke = read("run_dagent_qwen3_4b_original4081_legacy_matched_smoke2_zju47_gpu56_r3.sh")
    formal = read("run_dagent_qwen3_4b_original4081_legacy_matched_full468_zju47_gpu56_r3.sh")
    assert "smoke2-zju47-gpu56-r3" in smoke
    assert "q34leg-smk-g56-r3" in smoke
    assert "step468-zju47-gpu56-r3" in formal
    assert "q34leg-468-g56-r3" in formal
