from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sft_formal_launcher_freezes_prompt_and_runtime_limits():
    script = (
        ROOT
        / "exp/evaluate_dagent_qwen3_17b_sft_step510_prompt5_runtime0_formal_zju54_gpu1_r1.sh"
    ).read_text(encoding="utf-8")

    assert "export MAX_NODES_PER_TURN=0" in script
    assert "export PLANNER_PROMPT_MAX_NODES=5" in script
    assert "Use at most 5 nodes." in script
    assert 'protocol.get("planner_prompt_max_nodes") != 5' in script
    assert 'protocol.get("runtime_max_nodes_per_turn") != 0' in script


def test_sft_formal_launcher_preserves_old_evidence_and_checks_gpu_identity():
    script = (
        ROOT
        / "exp/evaluate_dagent_qwen3_17b_sft_step510_prompt5_runtime0_formal_zju54_gpu1_r1.sh"
    ).read_text(encoding="utf-8")

    assert "Refusing to overwrite existing evidence" in script
    assert "GPU-84973b7c-709f-7d9c-d4b6-a86d53d44f94" in script
    assert "assert_gpu_free" in script
    assert "kill " not in script
