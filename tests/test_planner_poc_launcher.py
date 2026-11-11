from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "exp" / "run_dagent_planner_poc_matched_short.sh"
FORMAL_LAUNCHER = ROOT / "exp" / "run_dagent_planner_poc_zju47_gpu56_formal90.sh"
BASE = ROOT / "run_rosa_train_2x_rtx_pro_6000.sh"


def test_poc_launcher_freezes_the_matched_contract():
    text = LAUNCHER.read_text(encoding="utf-8")
    required = [
        "ADV_ESTIMATOR=workflow_rloo",
        "RESULT_REWARD_ONLY=true",
        "FINAL_REWARD_EM_WEIGHT=1.0",
        "PLANNER_CCW_MODE=off",
        "TRAIN_BATCH_SIZE=24",
        "WORKFLOW_ROLLOUT_N=4",
        'MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"',
        'MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"',
        "PPO_MINI_BATCH_SIZE=96",
        "ACTOR_FORCE_SINGLE_ROW_DYNAMIC_MICROBATCHES=true",
        "minimum_padded_sequence_length=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))",
        "PPO_MAX_TOKEN_LEN_PER_GPU < minimum_padded_sequence_length",
        "ACTOR_LR=5e-7",
        'ACTOR_CALCULATE_OLD_LOG_PROB_ENTROPY="${ACTOR_CALCULATE_OLD_LOG_PROB_ENTROPY:-false}"',
        'PLANNER_POC_GRADIENT_AUDIT="${PLANNER_POC_GRADIENT_AUDIT:-false}"',
        "export SAVE_FREQ=-1",
        "export TEST_FREQ=-1",
        "TOTAL_EPOCHS=1",
        "CLEAN_STALE_PROCESSES=false",
        "off|real|shuffled",
    ]
    for value in required:
        assert value in text
    assert "ray stop" not in text
    assert "pkill" not in text


def test_base_launcher_exposes_poc_without_weakening_pure_em_guards():
    text = BASE.read_text(encoding="utf-8")
    for key in (
        "PLANNER_POC_MODE",
        "PLANNER_POC_BETA",
        "PLANNER_POC_LOSS_WEIGHT",
        "PLANNER_POC_SHUFFLE_SEED",
        "PLANNER_POC_GRADIENT_AUDIT",
        "PLANNER_POC_CURRENT_POLICY_AUDIT_ONLY",
        "algorithm.planner_outcome_contrast.mode",
        "algorithm.planner_outcome_contrast.gradient_audit",
        "algorithm.planner_outcome_contrast.current_policy_audit_only",
        "actor_rollout_ref.actor.calculate_old_log_prob_entropy",
        "actor_rollout_ref.actor.force_single_row_dynamic_microbatches",
    ):
        assert key in text
    assert '"${PLANNER_CCW_MODE}" != "off"' in text
    assert '"${WORKFLOW_SOD_ENABLED}" != "false"' in text


def test_poc_default_mode_is_quoted_to_avoid_yaml_boolean_coercion():
    text = (ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml").read_text(
        encoding="utf-8"
    )
    section = text.split("planner_outcome_contrast:", 1)[1]
    assert 'mode: "off"' in section


def test_poc_formal_launcher_preserves_the_unified_90_step_contract():
    text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
    required = [
        'DAGENT_CUDA_VISIBLE_DEVICES="${DAGENT_CUDA_VISIBLE_DEVICES:-5,6}"',
        'DAGENT_MINIMUM_FREE_MIB="${DAGENT_MINIMUM_FREE_MIB:-47000}"',
        'MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}"',
        'ACTOR_USE_REMOVE_PADDING="${ACTOR_USE_REMOVE_PADDING:-true}"',
        'CRITIC_USE_REMOVE_PADDING="${CRITIC_USE_REMOVE_PADDING:-true}"',
        "DAGENT_ALLOW_SHARED_GPU=false",
        "ADV_ESTIMATOR=workflow_rloo",
        "RESULT_REWARD_ONLY=true",
        "PLANNER_CCW_MODE=off",
        "PLANNER_POC_MODE=real",
        "PLANNER_POC_LOSS_WEIGHT=0.02",
        "PLANNER_POC_GRADIENT_AUDIT=false",
        "TRAIN_BATCH_SIZE=24",
        "WORKFLOW_ROLLOUT_N=4",
        "PPO_MINI_BATCH_SIZE=96",
        "TOTAL_TRAINING_STEPS=90",
        "TOTAL_EPOCHS=2",
        "SAVE_FREQ=10",
        "TEST_FREQ=90",
        "CLEAN_STALE_PROCESSES=false",
        "run_dagent_unified_sft_pure_em_old1500_reason578_zju57_gpu23.sh",
    ]
    for value in required:
        assert value in text
    assert "ray stop" not in text
    assert "pkill" not in text
