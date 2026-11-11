from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EGCP = ROOT / "exp/evaluate_q25_3b_step468_egcp1000_pair_r1.sh"
SWAP = ROOT / "exp/evaluate_q25_3b_step468_planner_swap_2wiki500_r1.sh"
GRADIENT = ROOT / "exp/audit_q25_3b_oldpsc_step468_current_policy_gradients_r1.sh"


def test_egcp_pair_uses_real_3b_checkpoints_and_frozen_cap0_contract():
    text = EGCP.read_text(encoding="utf-8")
    assert "Qwen2.5-3B-PureEM-Unlimited-step468-cap0-formal-train-r1" in text
    assert "q25-oldpsc-step468-formalpilot175-20260812-r1/merged_model" in text
    assert "dagent_reason_search_evidence_first_mixed_train1000_v1" in text
    assert "export MAX_NODES_PER_TURN=0" in text
    assert "export PLANNER_PROMPT_MAX_NODES=0" in text
    assert "export CHAT_TEMPLATE_ENABLE_THINKING=false" in text
    assert "--bootstrap-samples 100000" in text
    assert "nvidia-smi --query-compute-apps=gpu_uuid,pid" in text
    assert "Refusing to overwrite evidence" in text


def test_planner_swap_wrapper_uses_3b_cap0_and_question_gold_alignment():
    text = SWAP.read_text(encoding="utf-8")
    assert "Qwen2.5-3B-PureEM-Unlimited-step468-cap0-formal-train-r1" in text
    assert "q25-oldpsc-step468-formalpilot175-20260812-r1/merged_model" in text
    assert "PLANNER_PROMPT_MAX_NODES=0" in text
    assert "MAX_NODES_PER_TURN=0" in text
    assert "CHAT_TEMPLATE_ENABLE_THINKING=false" in text
    assert "ALIGNMENT_KEY=question_gold" in text
    assert "Qwen2.5-3B-PSCplanner-Purebody" in text
    assert "Qwen2.5-3B-Pureplanner-PSCbody" in text
    assert "FINAL_PSC_RUN=" in text
    assert "FINAL_PURE_RUN=" in text


def test_current_policy_gradient_audit_is_3b_paired_and_update_free():
    text = GRADIENT.read_text(encoding="utf-8")
    assert "q25-oldpsc-step468-formalpilot175-20260812-r1/merged_model" in text
    assert "PLANNER_POC_MINIBATCH_MODE=question_preserving" in text
    assert "PLANNER_POC_PAIRED_SHUFFLED_AUDIT=true" in text
    assert "PLANNER_POC_CURRENT_POLICY_AUDIT_ONLY=true" in text
    assert "PLANNER_POC_SHUFFLE_SEED=20260813" in text
    assert "MODEL_ENABLE_THINKING=false" in text
    assert "MAX_NODES_PER_TURN=0" in text
    assert "TOTAL_TRAINING_STEPS=1" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text
    assert "nvidia-smi --query-compute-apps=gpu_uuid,pid" in text
    assert "Refusing to overwrite audit evidence" in text

    worker = (ROOT / "verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
    assert '"aux/planner_outcome_contrast/current_policy_audit_only"' in worker
    assert "scheduler_step = not (gradient_audit or current_policy_audit)" in worker
