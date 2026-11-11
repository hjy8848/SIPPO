from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "exp" / "evaluate_shoppingbench_grounded_sft_r5_qwen25_3b_zju47_gpu4_r1.sh"


def test_r5_grounded_launcher_freezes_new_data_gpu_and_no_gold_contract():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'export GPU_INDEX=4' in source
    assert 'export GPU_UUID="GPU-f68525c2-8e8a-a678-e083-f973d9c93468"' in source
    assert 'export SAMPLE_COUNT=16' in source
    assert 'export EVAL_CONCURRENCY=16' in source
    assert 'export ARMS="contract"' in source
    assert 'export SHOPPING_SYSTEM_PLAN_COMPILER_V1=1' in source
    assert '--expected-per-task 16' in source
    assert '--min-total 8' in source
    assert '--min-per-task 1' in source
    assert '"gold_runtime_access": False' in source
    assert '"external_api_calls": 0' in source
    assert '"plan_override": False' in source
    assert 'shoppingbench-cp-sft-grounding-pilot-r5-val48-zju47-cpu-r1/manifest.json' in source
    assert 'shoppingbench-cp-sft-spec-retrieval-r5-zju53-cpu-r1/report.json' in source
    assert 'gpu6' not in source
