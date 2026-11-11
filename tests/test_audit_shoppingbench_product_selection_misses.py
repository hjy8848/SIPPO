import json

from scripts.audit_shoppingbench_product_selection_misses import audit


def test_selection_miss_audit_separates_unique_and_ambiguous_constraints(tmp_path):
    attribution = {
        "rows": [
            {
                "sample_id": "unique",
                "category": "selection_miss",
                "visible_query": "Basket; it should have color family silver",
                "predicted_ids": ["1"],
                "gold_ids": ["2"],
            },
            {
                "sample_id": "ambiguous",
                "category": "selection_miss",
                "visible_query": "Part; it should have brand pitsbike",
                "predicted_ids": ["3"],
                "gold_ids": ["4"],
            },
        ]
    }
    unique_context = """[1_1]: (Title: Product ID: 1) Product ID: 1
Shop ID: 10
Title: Carbon basket
SKU Options: {"1": {"color_family": "carbon"}}
[1_2]: (Title: Product ID: 2) Product ID: 2
Shop ID: 20
Title: Silver basket
SKU Options: {"1": {"color_family": "silver"}}
"""
    ambiguous_context = """[1_1]: (Title: Product ID: 3) Product ID: 3
Shop ID: 30
Title: Pitsbike part
Attributes: {"brand": ["pitsbike"]}
[1_2]: (Title: Product ID: 4) Product ID: 4
Shop ID: 40
Title: Another Pitsbike part
Attributes: {"brand": ["pitsbike"]}
"""
    results = [
        {
            "sample_id": "unique",
            "raw_trace": {
                "nodes_execution": [{"type": "search", "search_context": unique_context}],
                "sft_events": [],
                "final_answer": "<product_ids>1</product_ids>",
            },
        },
        {
            "sample_id": "ambiguous",
            "raw_trace": {
                "nodes_execution": [{"type": "search", "search_context": ambiguous_context}],
                "sft_events": [],
                "final_answer": "<product_ids>3</product_ids>",
            },
        },
    ]
    attribution_path = tmp_path / "attribution.json"
    results_path = tmp_path / "results.jsonl"
    attribution_path.write_text(json.dumps(attribution), encoding="utf-8")
    results_path.write_text("\n".join(json.dumps(row) for row in results), encoding="utf-8")

    report = audit(attribution_path, results_path)

    by_id = {row["sample_id"]: row for row in report["rows"]}
    assert by_id["unique"]["diagnosis"] == "unique_structured_match_missed"
    assert by_id["unique"]["satisfying_ids_without_gold"] == ["2"]
    assert by_id["ambiguous"]["diagnosis"] == "multiple_visible_satisfiers"
    assert by_id["ambiguous"]["satisfying_ids_without_gold"] == ["3", "4"]
    assert report["gold_used_to_classify_candidates"] is False
    assert report["model_calls"] == 0
