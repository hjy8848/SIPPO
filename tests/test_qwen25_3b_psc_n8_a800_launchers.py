from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "exp" / "run_dagent_qwen25_3b_psc_n8_a800_repl_gpu45_r1.sh"
F1 = ROOT / "exp" / "run_dagent_qwen25_3b_psc_f1zero_n8_a800_repl_gpu45_r1.sh"
SMOKE = ROOT / "exp" / "run_dagent_qwen25_3b_psc_n8_a800_repl_gpu45_smoke2_r1.sh"
F1_SMOKE = ROOT / "exp" / "run_dagent_qwen25_3b_psc_f1zero_n8_a800_repl_gpu45_smoke2_r1.sh"


def test_a800_n8_psc_contract_is_frozen():
    text = BASE.read_text(encoding="utf-8")
    for value in (
        "38lmdgu9bdc9m-0",
        "GPU-2a478145-88e8-4713-aaf9-1db5f593f4fb",
        "GPU-05322139-8f57-32ce-692d-1aee157ba9b5",
        "Qwen2.5-3B-DeepSeek800-1xA6000-bf16/global_step_510/huggingface",
        "WORKFLOW_ROLLOUT_N=8",
        "PPO_MINI_BATCH_SIZE=256",
        "NUM_GPUS=2",
        "PLANNER_POC_MODE=real",
        "PLANNER_POC_LOSS_WEIGHT=0.02",
        'FINAL_REWARD_F1_TIEBREAK_ENABLED="${FINAL_REWARD_F1_TIEBREAK_ENABLED:-false}"',
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-468}"',
        'SAVE_FREQ="${SAVE_FREQ:-50}"',
        "RETRIEVER_URL=http://127.0.0.1:18066/retrieve",
        "MAX_PROMPT_LENGTH=4096",
        "ROLLOUT_TEMPERATURE=0.8",
        "ACTOR_CALCULATE_OLD_LOG_PROB_ENTROPY=false",
    ):
        assert value in text
    assert 'exec bash "${JADE_ROOT}/run_rosa_train_2x_rtx_pro_6000.sh"' in text
    assert "run_dagent_qwen25_3b_psc_mixed15k_smoke2_zju57.sh" not in text
    assert "gpu45-r1" in text


def test_a800_f1_n8_wrapper_only_changes_reward_and_artifact_identity():
    text = F1.read_text(encoding="utf-8")
    assert "run_dagent_qwen25_3b_psc_n8_a800_repl_gpu45_r1.sh" in text
    assert "FINAL_REWARD_F1_TIEBREAK_ENABLED=true" in text
    assert "F1ZeroTieBreak" in text
    assert "q25psc-f1zero-n8-g45-r1" in text
    assert "gpu45-r1" in text


def test_a800_n8_smokes_are_two_step_and_isolated():
    for path, enabled in ((SMOKE, "false"), (F1_SMOKE, "true")):
        text = path.read_text(encoding="utf-8")
        assert "TOTAL_TRAINING_STEPS=2" in text
        assert "SAVE_FREQ=-1" in text
        assert "TEST_FREQ=-1" in text
        assert f"FINAL_REWARD_F1_TIEBREAK_ENABLED={enabled}" in text
        assert "smoke2" in text
        assert "q25psc-" in text
