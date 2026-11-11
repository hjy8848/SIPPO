import json

from scripts.analyze_shoppingbench_voucher_compiler_failures import analyze


def _row(sample_id, plan, queries, contexts, predicted):
    nodes = [
        {
            "type": "search",
            "executed_query": query,
            "search_context": context,
            "rewrite_queries": [],
        }
        for query, context in zip(queries, contexts)
    ]
    return {
        "sample_id": sample_id,
        "query": "voucher question",
        "expected_product_ids": ["1", "2"],
        "predicted_product_ids": predicted,
        "score": {},
        "format_error": None,
        "contract_error": None,
        "pipeline_error": None,
        "request_error": None,
        "raw_trace": {
            "raw_plan_output": "raw-plan",
            "plan_output": plan,
            "nodes_execution": nodes,
        },
    }


def test_failure_slice_attributes_retrieval_gain(tmp_path):
    baseline_row = _row("voucher-0000", "old", ["q1"], ["Product ID: 1"], [])
    candidate_row = _row(
        "voucher-0000",
        "new",
        ["q1", "q2"],
        ["Product ID: 1", "Product ID: 2"],
        ["1", "2"],
    )
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text(json.dumps(baseline_row) + "\n")
    candidate.write_text(json.dumps(candidate_row) + "\n")

    report = analyze(baseline, candidate, tmp_path / "report", 1)

    assert report["baseline"]["corresponding_recall"] == 0.5
    assert report["candidate"]["corresponding_recall"] == 1.0
    assert report["category_transitions"] == {"retrieval_miss->success": 1}
    assert report["candidate_remaining_bottleneck"] == "success"
    assert report["training_authorized"] is False
