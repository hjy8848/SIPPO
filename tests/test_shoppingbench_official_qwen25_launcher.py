from pathlib import Path


SCRIPT = Path("exp/evaluate_shoppingbench_official_qwen25_3b_full750_zju57_gpu0_r1.sh")


def test_launcher_freezes_official_workflow_and_local_model():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "2f4d132500d4962d7ee3143e103b93499c82aca0" in text
    assert "Qwen2.5-3B-Instruct" in text
    assert 'run_rollout.py" "${config}"' in text
    assert '"external_shopping_api_calls": 0' in text
    assert '"max_react_steps": 30' in text
    assert '"tool_result_page_size": 10' in text
    assert "src/search_engine/server.py" in text
    assert 'PORT="${SEARCH_PORT}"' in text


def test_launcher_is_collision_resistant():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "GPU-1a61c7d2-ff37-3b2a-d14b-02fea9d1254b" in text
    assert "query-compute-apps=pid,process_name,used_memory" in text
    assert "Refusing to overwrite" in text
    assert "Refusing existing cache" in text
    assert "Matching run exists" in text
    assert 'sport = :${port}' in text
    assert text.index("Official search server is unhealthy") < text.index("GPU UUID mismatch")
    assert 'CACHE_TAG="${CACHE_TAG:-$(printf' in text
    assert 'MODEL_PORT="${MODEL_PORT:-12431}"' in text
    assert 'SEARCH_PORT="${SEARCH_PORT:-5631}"' in text
    assert 'PREFLIGHT_COUNT="${PREFLIGHT_COUNT:-1}"' in text
    assert "preflight_gate.json" in text
    assert 'config_formal_${task}.json' in text
    assert "find_product_calls" in text
