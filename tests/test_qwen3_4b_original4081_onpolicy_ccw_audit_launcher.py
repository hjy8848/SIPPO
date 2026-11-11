from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "exp"
    / "run_qwen3_4b_original4081_sft_onpolicy_ccw_audit_a800_repl_gpu4_r1.sh"
)


def test_launcher_freezes_strict_offline_credit_contract():
    source = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "QUESTION_COUNT_PER_DATASET=8",
        "ROLLOUTS_PER_QUESTION=4",
        "SUFFIX_SAMPLES=8",
        "CONTRACT_PRESERVING_DAG_V1_REQUEST=1",
        "MAX_NODES_PER_TURN=0",
        "PLANNER_PROMPT_MAX_NODES=5",
        "CHAT_TEMPLATE_ENABLE_THINKING=false",
        "strict paired on-policy",
        "minimum_informative_cases\":30",
        "direction_ci_low_must_exceed\":0.55",
        "identity_terminal_mismatches_required\":0",
        "optimizer_updates\":0",
    ):
        assert required in source
    assert "Insufficient information" not in source
    assert "optimizer.step" not in source
    assert "run_rosa_train" not in source


def test_launcher_reexecutes_and_blocks_training_until_gate_passes():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "audit_dagent_ccw_onpolicy_replacements.py" in source
    assert "AUDIT_SERVER_PATH" in source
    assert "identity_response_terminal_em_changed_samples" in source
    assert '"one_batch_authorized": bool(report["passed"])' in source
    assert "--concurrency 1" in source
    assert "--phase claim" in source
    assert 'GPU_UUID=GPU-2a478145-88e8-4713-aaf9-1db5f593f4fb' in source
    assert '--path-absent "${ROOT}"' in source
    assert '--port-free "${PORT}"' in source
    assert '[[ ! -e "${path}" ]]' in source
