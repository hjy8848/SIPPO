import json

from scripts.audit_shoppingbench_product_conservative_selection import audit


def test_conservative_selection_changes_only_unique_non_satisfier(tmp_path):
    attribution = {
        "rows": [
            {"sample_id": "unique", "visible_query": "Basket; it should have color family silver"},
            {"sample_id": "ambiguous", "visible_query": "Part; it should have brand pitsbike"},
            {"sample_id": "success", "visible_query": "Desk; it should have color family black"},
        ]
    }
    contexts = {
        "unique": """[1_1]: (Title: Product ID: 1) Product ID: 1
SKU Options: {"1": {"color_family": "carbon"}}
[1_2]: (Title: Product ID: 2) Product ID: 2
SKU Options: {"1": {"color_family": "silver"}}
""",
        "ambiguous": """[1_1]: (Title: Product ID: 3) Product ID: 3
Attributes: {"brand": ["pitsbike"]}
[1_2]: (Title: Product ID: 4) Product ID: 4
Attributes: {"brand": ["pitsbike"]}
""",
        "success": """[1_1]: (Title: Product ID: 5) Product ID: 5
SKU Options: {"1": {"color_family": "black"}}
[1_2]: (Title: Product ID: 6) Product ID: 6
SKU Options: {"1": {"color_family": "white"}}
""",
    }
    results = []
    for sample_id, predicted, expected in (
        ("unique", "1", "2"),
        ("ambiguous", "3", "4"),
        ("success", "5", "5"),
    ):
        results.append(
            {
                "sample_id": sample_id,
                "predicted_product_ids": [predicted],
                "expected_product_ids": [expected],
                "raw_trace": {
                    "nodes_execution": [{"type": "search", "search_context": contexts[sample_id]}]
                },
            }
        )
    attribution_path = tmp_path / "attribution.json"
    results_path = tmp_path / "results.jsonl"
    attribution_path.write_text(json.dumps(attribution), encoding="utf-8")
    results_path.write_text("\n".join(json.dumps(row) for row in results), encoding="utf-8")

    report = audit(attribution_path, results_path)

    by_id = {row["sample_id"]: row for row in report["rows"]}
    assert by_id["unique"]["candidate_ids_without_gold"] == ["2"]
    assert by_id["ambiguous"]["candidate_ids_without_gold"] == ["3"]
    assert by_id["success"]["candidate_ids_without_gold"] == ["5"]
    assert report["changed_rows"] == 1
    assert report["baseline_exact"] == 1
    assert report["candidate_exact"] == 2
    assert report["improved_rows"] == 1
    assert report["regressed_rows"] == 0
    assert report["gold_used_by_decision"] is False
