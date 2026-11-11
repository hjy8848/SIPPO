from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "exp"
    / "run_qwen3_4b_cpv2_lexicographic_f1_rollout_pilot_a800_repl_gpu5_r1.sh"
)


def test_rollout_pilot_freezes_current_policy_contract_without_training():
    source = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "questions\":96",
        "rollouts_per_question\":4",
        "workflows\":384",
        "CONTRACT_PRESERVING_DAG_V1_REQUEST=1",
        "MAX_NODES_PER_TURN=0",
        "PLANNER_PROMPT_MAX_NODES=5",
        "GEN_TEMPERATURE=0.8",
        "GEN_TOP_P=0.95",
        "TOPK=3",
        "CHAT_TEMPLATE_ENABLE_THINKING=false",
        "optimizer_updates\":0",
    ):
        assert required in source
    assert "run_rosa_train" not in source
    assert "optimizer.step" not in source


def test_rollout_pilot_is_collision_resistant_and_keeps_evidence_unique():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--phase claim" in source
    assert 'GPU_UUID=GPU-05322139-8f57-32ce-692d-1aee157ba9b5' in source
    assert '--path-absent "${ROOT}"' in source
    assert '--port-free "${PORT}"' in source
    assert '[[ ! -e "${path}" ]]' in source
    assert "RESUME_RESULTS=true" not in source
