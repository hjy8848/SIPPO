from scripts.replay_shoppingbench_artifact_runtime import aggregate, replay_row


def test_replay_uses_retrieved_artifact_for_sink():
    row = {
        "sample_id": "product-0000",
        "query": "Find Product A",
        "expected_product_ids": ["11"],
        "format_error": "old invalid",
        "score": {
            "exact_id_fraction": 0.0,
            "exact_sequence_success": False,
            "first_id_exact": False,
        },
        "raw_trace": {
            "protocol_profile": "shoppingbench",
            "plan_override_used": True,
            "answer_sink_id": "1",
            "contract_protocol": {
                "plan_nodes": [{"id": "1", "type": "search"}]
            },
            "final_answer": "Product ID: 999",
            "nodes_execution": [
                {
                    "id": "1",
                    "type": "search",
                    "executed_query": "Product A",
                    "conclusion": "Product ID: 999\nProduct ID: 11",
                    "search_context": (
                        "[1_1]: (Title: Product ID: 11) Product ID: 11\n"
                        "Shop ID: 7\nTitle: Product A\nPrice: 10\nServices: COD"
                    ),
                }
            ],
        },
    }
    replayed = replay_row(row)
    assert replayed["replayed_product_ids"] == ["11"]
    assert replayed["replayed_score"]["exact_id_fraction"] == 1.0


def test_aggregate_labels_replay_as_offline_not_scientific_claim():
    row = {
        "old_format_valid": False,
        "replayed_format_valid": True,
        "old_score": {
            "exact_id_fraction": 0.0,
            "exact_sequence_success": False,
            "first_id_exact": False,
        },
        "replayed_score": {
            "exact_id_fraction": 1.0,
            "exact_sequence_success": True,
            "first_id_exact": True,
        },
    }
    report = aggregate([row])
    assert report["gold_used_by_runtime"] is False
    assert report["scientific_improvement_claim"] is False
    assert report["replayed_format_valid"] == 1
