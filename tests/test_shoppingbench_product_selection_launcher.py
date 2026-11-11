from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_product_selection_launcher_freezes_isolated_online_contract():
    text = (
        ROOT / "exp/evaluate_shoppingbench_product_selection_fallback_zju47_gpu2_r1.sh"
    ).read_text(encoding="utf-8")

    assert "ShoppingBench-ProductSelectionFallback" in text
    assert 'GPU_INDEX=2' in text
    assert 'GPU_UUID="GPU-beefff4e-6ee0-c6d3-e551-d03f97adf855"' in text
    assert 'RETRIEVER_PORT=5672' in text
    assert 'ROSA_PORT=8212' in text
    assert "rosa_server_product_selection_fallback.py" in text
    assert "compare_shoppingbench_product_selection_fallback.py" in text
    assert '"preregistered_trigger":"product-0011 only"' in text
    assert '"gold_runtime_access":False' in text
    assert "--limit 16 --concurrency 16" in text
    assert "--arm contract" in text
    assert "nvidia-smi --query-compute-apps=gpu_uuid,pid" in text
    assert "Refusing to overwrite" in text
