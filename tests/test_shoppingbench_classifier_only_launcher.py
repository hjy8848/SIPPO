from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "exp" / "evaluate_shoppingbench_classifier_only_r5_qwen25_3b_zju47_gpu3_r1.sh"
QUEUE = ROOT / "exp" / "queue_shoppingbench_classifier_only_r5_zju47_gpu3_r1.sh"


def test_classifier_only_launcher_freezes_online_causal_contract():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'export GPU_INDEX=3' in source
    assert 'export GPU_UUID="GPU-92e299e4-69dc-adf1-85e7-8bfc54898c2d"' in source
    assert 'export SAMPLE_COUNT=16' in source
    assert 'export EVAL_CONCURRENCY=16' in source
    assert 'export ARMS="contract"' in source
    assert 'export SHOPPING_SYSTEM_PLAN_COMPILER_V1=1' in source
    assert '"changed_variable": "deterministic request classifier"' in source
    assert '"gold_runtime_access": False' in source
    assert '"external_api_calls": 0' in source
    assert '"training_authorized": False' in source
    assert "PLANNER_OUTPUT_OVERRIDE_MANIFEST" in source
    assert "2220fd04d596eaacf703cdd7307bcf4a228a8ad7d54d9f73022663f4b0e53049" in source
    assert '"planner_output_frozen": True' in source
    assert '"planner_model_calls": 0' in source


def test_classifier_only_queue_pins_uuid_and_unique_artifact():
    source = QUEUE.read_text(encoding="utf-8")
    assert "WAIT_LAUNCH_AUTHORIZED=true" in source
    assert "WAIT_GPU_INDEX=3" in source
    assert 'WAIT_GPU_UUID="GPU-92e299e4-69dc-adf1-85e7-8bfc54898c2d"' in source
    assert "wait_for_exclusive_gpu_then_exec_v2.sh" in source
    assert "ShoppingBench-ClassifierOnly-Qwen2.5-3B-Instruct-r5-zju47-gpu3-val48-r1" in source
