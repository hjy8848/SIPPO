from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def read(name: str) -> str:
    return (EXP / name).read_text(encoding="utf-8")


def test_legacy_matched_core_freezes_scientific_contract():
    script = read("run_dagent_qwen3_4b_original4081_legacy_matched_zju47_gpu56_r1.sh")
    required = (
        "DAGENT-Qwen3-4B-Original4081SFT-NoThink-step510-A100-zju12-gpu23-r1",
        "rl_train_mixed_2wiki40_hotpot30_musique30_seed42.parquet",
        "CP_DAGENT_ENABLED=false",
        "PLANNER_PROMPT_MAX_NODES=0",
        "MAX_NODES_PER_TURN=0",
        "SEARCH_REWRITE_MAX_RETRIES=1",
        "FINAL_ANSWER_RETRY_LIMIT=1",
        "MODEL_ENABLE_THINKING=false",
        "ADV_ESTIMATOR=workflow_rloo",
        "FINAL_REWARD_EM_WEIGHT=1.0",
        "FINAL_REWARD_F1_TIEBREAK_ENABLED=false",
        "TRAIN_BATCH_SIZE=32",
        "WORKFLOW_ROLLOUT_N=4",
        "PPO_MINI_BATCH_SIZE=64",
        "PPO_MAX_TOKEN_LEN_PER_GPU=8192",
        "ACTOR_LR=5e-7",
        "ACTOR_LOSS_AGG_MODE=token-mean",
        "ROLLOUT_TEMPERATURE=0.8",
        "ROLLOUT_TOP_P=0.95",
        "RETRIEVAL_TOPK=3",
    )
    for value in required:
        assert value in script
    for forbidden in ("pkill", "ray stop", "rm -", "kill "):
        assert forbidden not in script


def test_legacy_smoke_and_formal_use_disjoint_frozen_artifacts():
    smoke = read("run_dagent_qwen3_4b_original4081_legacy_matched_smoke2_zju47_gpu56_r1.sh")
    formal = read("run_dagent_qwen3_4b_original4081_legacy_matched_full468_zju47_gpu56_r1.sh")
    assert "TOTAL_TRAINING_STEPS=2" in smoke
    assert "SAVE_FREQ=2" in smoke
    assert "q34leg-smk-g56-r1" in smoke
    assert "TOTAL_TRAINING_STEPS=468" in formal
    assert "SAVE_FREQ=234" in formal
    assert "TEST_FREQ=468" in formal
    assert "q34leg-468-g56-r1" in formal
    assert "smoke2" not in formal.split("EXPERIMENT_NAME=", 1)[1]
