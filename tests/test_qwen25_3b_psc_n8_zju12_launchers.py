from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "exp" / "run_dagent_qwen25_3b_psc_n8_zju12_gpu01_r1.sh"
F1 = ROOT / "exp" / "run_dagent_qwen25_3b_psc_f1zero_n8_zju12_gpu01_r1.sh"
SMOKE = ROOT / "exp" / "run_dagent_qwen25_3b_psc_f1zero_n8_zju12_gpu01_smoke2_r1.sh"
F1_R2 = ROOT / "exp" / "run_dagent_qwen25_3b_psc_f1zero_n8_zju12_gpu01_r2.sh"
SMOKE_R2 = (
    ROOT / "exp" / "run_dagent_qwen25_3b_psc_f1zero_n8_zju12_gpu01_smoke2_r2.sh"
)
F1_R3 = ROOT / "exp" / "run_dagent_qwen25_3b_psc_f1zero_n8_zju12_gpu01_r3.sh"
SMOKE_R3 = (
    ROOT / "exp" / "run_dagent_qwen25_3b_psc_f1zero_n8_zju12_gpu01_smoke2_r3.sh"
)


def test_zju12_n8_contract_matches_the_a800_scientific_contract():
    text = BASE.read_text(encoding="utf-8")
    for value in (
        '"$(hostname)" == "zju-12"',
        "GPU-a7337bf3-2665-ef32-d66e-098046f76234",
        "GPU-00c7981b-83e5-0fa3-a01d-dcce5179a950",
        "Qwen2.5-3B-DeepSeek800-1xA6000-bf16/global_step_510/huggingface",
        "WORKFLOW_ROLLOUT_N=8",
        "PPO_MINI_BATCH_SIZE=256",
        "NUM_GPUS=2",
        "PLANNER_POC_MODE=real",
        "PLANNER_POC_LOSS_WEIGHT=0.02",
        'FINAL_REWARD_F1_TIEBREAK_ENABLED="${FINAL_REWARD_F1_TIEBREAK_ENABLED:-false}"',
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-468}"',
        'SAVE_FREQ="${SAVE_FREQ:-50}"',
        "MAX_PROMPT_LENGTH=4096",
        "ROLLOUT_TEMPERATURE=0.8",
        "ROLLOUT_TOP_P=0.95",
        "RETRIEVAL_TOPK=3",
        "MAX_NODES_PER_TURN=0",
        "PLANNER_PROMPT_MAX_NODES=5",
        "MODEL_ENABLE_THINKING=false",
    ):
        assert value in text
    assert 'exec bash "${JADE_ROOT}/run_rosa_train_2x_rtx_pro_6000.sh"' in text


def test_zju12_f1_wrapper_only_enables_the_lexicographic_tie_break():
    text = F1.read_text(encoding="utf-8")
    assert "run_dagent_qwen25_3b_psc_n8_zju12_gpu01_r1.sh" in text
    assert "FINAL_REWARD_F1_TIEBREAK_ENABLED=true" in text
    assert "FINAL_REWARD_F1_TIEBREAK_WEIGHT=0.1" in text
    assert "F1ZeroTieBreak" in text
    assert "q25f1n8-z12-g01-r1" in text


def test_zju12_f1_smoke_is_two_steps_and_uses_fresh_artifacts():
    text = SMOKE.read_text(encoding="utf-8")
    assert "TOTAL_TRAINING_STEPS=2" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text
    assert "FINAL_REWARD_F1_TIEBREAK_ENABLED=true" in text
    assert "smoke2" in text
    assert "q25f1n8s-z12-g01-r1" in text


def test_zju12_r2_memory_repair_preserves_the_scientific_contract():
    for path in (F1_R2, SMOKE_R2):
        text = path.read_text(encoding="utf-8")
        assert "FINAL_REWARD_F1_TIEBREAK_ENABLED=true" in text
        assert "FINAL_REWARD_F1_TIEBREAK_WEIGHT=0.1" in text
        assert "ENABLE_GRADIENT_CHECKPOINTING=true" in text
        assert "ACTOR_PARAM_OFFLOAD=true" in text
        assert "ACTOR_FORCE_SINGLE_ROW_DYNAMIC_MICROBATCHES=true" in text
        assert "run_dagent_qwen25_3b_psc_n8_zju12_gpu01_r1.sh" in text
        assert "r2" in text


def test_zju12_r3_avoids_the_force_single_row_padding_assertion():
    for path in (F1_R3, SMOKE_R3):
        text = path.read_text(encoding="utf-8")
        assert "FINAL_REWARD_F1_TIEBREAK_ENABLED=true" in text
        assert "ENABLE_GRADIENT_CHECKPOINTING=true" in text
        assert "ACTOR_PARAM_OFFLOAD=true" in text
        assert "ACTOR_FORCE_SINGLE_ROW_DYNAMIC_MICROBATCHES=false" in text
        assert "PPO_MAX_TOKEN_LEN_PER_GPU=6144" in text
        assert "run_dagent_qwen25_3b_psc_n8_zju12_gpu01_r1.sh" in text
        assert "r3" in text
