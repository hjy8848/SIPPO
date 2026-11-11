from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERIC = ROOT / "exp" / "evaluate_qwen3_17b_grouped_realpoc_vs_pureem_step300_zju54_gpu1_r1.sh"
LAUNCHER = ROOT / "exp" / "evaluate_qwen25_3b_pureem_unlimited_step468_cap0_formal_train1000_zju53_gpu6_r1.sh"


def test_generic_evaluator_allows_host_gpu_and_environment_overrides():
    script = GENERIC.read_text(encoding="utf-8")
    assert 'EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-1}"' in script
    assert 'ROSA_PYTHON_ENV="${ROSA_PYTHON_ENV:-' in script
    assert 'EXPECTED_HOST="${EXPECTED_HOST:-zju-54}"' in script
    assert 'EXPECTED_GPU_UUID="${EXPECTED_GPU_UUID:-GPU-' in script
    assert 'PLANNER_PROMPT_MAX_NODES="${PLANNER_PROMPT_MAX_NODES:-${MAX_NODES_PER_TURN}}"' in script


def test_qwen25_pureem_eval_contract_is_frozen():
    script = LAUNCHER.read_text(encoding="utf-8")
    expected = (
        "STEP=468",
        "ALLOW_NON_MULTIPLE_OF_10_STEP=true",
        "EVAL_SIDE=pure",
        "EVAL_SCOPE=formal_train",
        "REUSE_MERGED_MODEL=false",
        "MAX_NODES_PER_TURN=0",
        "PLANNER_PROMPT_MAX_NODES=0",
        "EVAL_CUDA_VISIBLE_DEVICES=6",
        "EXPECTED_HOST=zju-53",
        "EXPECTED_GPU_UUID=GPU-0400dad8-3492-7afc-612e-1c89040cb688",
        "EXPECTED_WORLD_SIZE=1",
        "EXPECTED_MODEL_SHARD_SIZE=13588626251",
        "EXPECTED_OPTIM_SHARD_SIZE=24687668767",
    )
    for item in expected:
        assert item in script
    assert "Qwen2.5-3B-GroupedPureEM" in script
    assert "smoke16" not in script
