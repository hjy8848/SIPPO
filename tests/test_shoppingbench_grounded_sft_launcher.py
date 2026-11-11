from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "exp" / "evaluate_shoppingbench_grounded_sft_pilot_qwen25_3b_zju47_gpu6_r1.sh"


def test_grounded_sft_launcher_freezes_validation_gate_and_no_gold_contract():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'export GPU_INDEX=6' in source
    assert 'export GPU_UUID="GPU-e94570b1-a559-c455-6250-7a6b8a6a8722"' in source
    assert 'export SAMPLE_COUNT=16' in source
    assert 'export ARMS="contract"' in source
    assert 'export SHOPPING_SYSTEM_PLAN_COMPILER_V1=1' in source
    assert '--expected-per-task 16' in source
    assert '--min-total 8' in source
    assert '--min-per-task 1' in source
    assert '"gold_runtime_access": False' in source
    assert '"external_api_calls": 0' in source
    assert '"plan_override": False' in source
    assert 'shoppingbench-cp-sft-grounding-pilot-val48-r1/manifest.json' in source
    assert 'shoppingbench-cp-sft-spec-retrieval-audit-zju47-cpu-r2/report.json' in source
