import json

from scripts.compare_shoppingbench_voucher_compiler_runs import compare


def _row(sample_id, raw, plan, query, predicted, fraction):
    return {
        "sample_id": sample_id,
        "query": "voucher question",
        "expected_product_ids": ["1", "2"],
        "predicted_product_ids": predicted,
        "score": {
            "exact_id_fraction": fraction,
            "exact_sequence_success": fraction == 1,
            "first_id_exact": bool(fraction),
        },
        "format_error": None,
        "contract_error": None,
        "pipeline_error": None,
        "request_error": None,
        "search_nodes": len(query),
        "retrieval_calls": len(query),
        "raw_trace": {
            "raw_plan_output": raw,
            "plan_output": plan,
            "final_answer": ",".join(predicted),
            "nodes_execution": [
                {"type": "search", "executed_query": item, "rewrite_queries": []}
                for item in query
            ],
        },
    }


def test_identity_control_and_affected_gain_are_separated(tmp_path):
    raw_control = "<plan>one</plan>"
    raw_affected = "<plan>two</plan>"
    baseline_rows = [
        _row("voucher-0000", raw_control, "compiled-one", ["q1"], ["1"], 0.5),
        _row("voucher-0001", raw_affected, "compiled-old", ["q2"], [], 0.0),
    ]
    candidate_rows = [
        _row("voucher-0000", raw_control, "compiled-one", ["q1"], ["1"], 0.5),
        _row("voucher-0001", raw_affected, "compiled-new", ["q2", "q3"], ["1"], 0.5),
    ]
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    overrides = tmp_path / "overrides.json"
    baseline.write_text("".join(json.dumps(row) + "\n" for row in baseline_rows))
    candidate.write_text("".join(json.dumps(row) + "\n" for row in candidate_rows))
    overrides.write_text(json.dumps({"voucher-0000": raw_control, "voucher-0001": raw_affected}))

    report = compare(baseline, candidate, overrides, tmp_path / "report", 2)

    assert report["identity_controls"]["rows"] == 1
    assert report["identity_controls"]["passed"] is True
    assert report["compiler_affected"] == {
        "rows": 1,
        "sample_ids": ["voucher-0001"],
        "mean_delta_exact_id_fraction": 0.5,
        "wins": 1,
        "losses": 0,
        "ties": 0,
    }
    assert report["runtime_causal_gate_passed"] is True
    assert report["training_authorized"] is False
