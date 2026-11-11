from pathlib import Path


SCRIPT = Path(
    "exp/evaluate_shoppingbench_official_vs_asc_qwen25_3b_shop16_zju54_gpu1_r1.sh"
)


def test_shop16_comparison_freezes_contract_and_preserves_strict_scoring() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SAMPLE_COUNT=16" in source
    assert "GPU_UUID=GPU-84973b7c-709f-7d9c-d4b6-a86d53d44f94" in source
    assert "temperature\": 0" in source
    assert '"gold_runtime_access": False' in source
    assert '"external_shopping_api_calls": 0' in source
    assert "official_strict_product_ids" in source
    assert "official_diagnostic_recovered_product_ids" in source
    assert "never promoted to strict success" in source
    assert "Refusing to overwrite" in source
    assert "kill " not in source


def test_shop16_comparison_uses_query_only_runtime_and_real_lucene_probe() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'json.dumps({"query": row["question"]}' in source
    assert "--profile shoppingbench-official" in source
    assert "probe_adaptive_shop_rag.py" in source
    assert "run_rollout.py" in source
    assert "reused_exact_contract_service" in source
