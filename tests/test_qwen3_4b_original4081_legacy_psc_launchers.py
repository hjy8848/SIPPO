from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def read(name: str) -> str:
    return (EXP / name).read_text(encoding="utf-8")


def test_legacy_psc_freezes_the_new_4b_matched_contract():
    script = read("run_dagent_qwen3_4b_original4081_legacy_psc_zju12_gpu01_r1.sh")
    for value in (
        "DAGENT-Qwen3-4B-Original4081SFT-NoThink-step510-A100-zju12-gpu23-r1",
        "rl_train_mixed_2wiki40_hotpot30_musique30_seed42.parquet",
        "CP_DAGENT_ENABLED=false",
        "PLANNER_PROMPT_MAX_NODES=0",
        "MAX_NODES_PER_TURN=0",
        "PLANNER_POC_MODE=real",
        "PLANNER_POC_REQUIRE_SIGNAL=true",
        "PLANNER_POC_LOSS_WEIGHT=0.02",
        "ADV_ESTIMATOR=workflow_rloo",
        "FINAL_REWARD_EM_WEIGHT=1.0",
        "FINAL_REWARD_F1_TIEBREAK_ENABLED=false",
        "TRAIN_BATCH_SIZE=32",
        "WORKFLOW_ROLLOUT_N=4",
        "PPO_MINI_BATCH_SIZE=128",
        "ACTOR_LR=5e-7",
        "ROLLOUT_TEMPERATURE=0.8",
        "ROLLOUT_TOP_P=0.95",
        "RETRIEVAL_TOPK=3",
    ):
        assert value in script
    for forbidden in ("pkill", "ray stop", "rm -", "kill "):
        assert forbidden not in script


def test_legacy_psc_smoke_and_formal_use_disjoint_artifacts():
    smoke = read("run_dagent_qwen3_4b_original4081_legacy_psc_smoke2_zju12_gpu01_r1.sh")
    formal = read("run_dagent_qwen3_4b_original4081_legacy_psc_full468_zju12_gpu01_r1.sh")
    assert "TOTAL_TRAINING_STEPS=2" in smoke
    assert "q34legpsc-smk-z12-g01-r1" in smoke
    assert "TOTAL_TRAINING_STEPS=468" in formal
    assert "q34legpsc-468-z12-g01-r1" in formal
    assert "smoke2" not in formal.split("EXPERIMENT_NAME=", 1)[1]

