from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "exp" / "evaluate_shoppingbench_classifier_only_r5_a800_repl_gpu1_r1.sh"


def test_a800_classifier_launcher_freezes_identity_and_resources():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'GPU_INDEX=1' in source
    assert 'GPU_UUID="GPU-8bad8428-3b61-78b2-e6b6-51dd2be7472d"' in source
    assert 'PLANNER_OUTPUTS=' in source
    assert '2220fd04d596eaacf703cdd7307bcf4a228a8ad7d54d9f73022663f4b0e53049' in source
    assert '--planner-output-override-manifest "${PLANNER_OUTPUTS}"' in source
    assert 'GEN_TEMPERATURE=0' in source
    assert 'TOPK=5' in source
    assert 'CHAT_TEMPLATE_ENABLE_THINKING=0' in source
    assert 'compare_shoppingbench_classifier_only_online.py' in source
    assert 'nvidia-smi --query-compute-apps=gpu_uuid,pid' in source
    assert 'external' not in source.lower()
