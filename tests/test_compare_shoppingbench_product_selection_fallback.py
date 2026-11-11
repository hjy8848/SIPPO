import json

from scripts.compare_shoppingbench_product_selection_fallback import compare


def _write_rows(path, *, candidate=False, extra_trigger=None):
    rows = []
    for index in range(16):
        sample_id = f"product-{index:04d}"
        trigger = sample_id == "product-0011" or sample_id == extra_trigger
        old_id = str(1000 + index)
        new_id = "3081527776" if sample_id == "product-0011" else old_id
        predicted = [new_id if candidate and trigger else old_id]
        expected = [new_id if sample_id == "product-0011" else old_id]
        decision = {
            "applied": bool(candidate and trigger),
            "baseline_id": old_id,
            "selected_id": predicted[0],
            "gold_used": False,
        }
        trace = {
            "raw_plan_output": f"<plan><search>query {index}</search></plan>",
            "planner_output_override_used": True,
            "contract_protocol": {
                "plan_compiler": {
                    "product_query_fallback": {"applied": False, "gold_used": False}
                }
            },
        }
        if candidate:
            trace["product_selection_fallback"] = {
                "enabled": True,
                "gold_used": False,
                "decisions": [decision],
            }
        rows.append(
            {
                "sample_id": sample_id,
                "query": f"question {index}",
                "predicted_product_ids": predicted,
                "expected_product_ids": expected,
                "score": {
                    "exact_sequence_success": predicted == expected,
                    "exact_id_fraction": float(predicted == expected),
                },
                "format_error": None,
                "contract_error": None,
                "pipeline_error": None,
                "request_error": None,
                "raw_trace": trace,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_comparator_requires_one_preregistered_improvement(tmp_path):
    baseline, candidate = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    _write_rows(baseline)
    _write_rows(candidate, candidate=True)

    report = compare(baseline, candidate, tmp_path / "report")

    assert report["passed"] is True
    assert report["baseline"]["exact_sequences"] == 15
    assert report["candidate"]["exact_sequences"] == 16
    assert report["activation_pass"] is True


def test_comparator_rejects_an_extra_trigger(tmp_path):
    baseline, candidate = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    _write_rows(baseline)
    _write_rows(candidate, candidate=True, extra_trigger="product-0003")

    report = compare(baseline, candidate, tmp_path / "report")

    assert report["passed"] is False
    assert report["activation_pass"] is False
    assert report["nontrigger_prediction_identity_pass"] is True
