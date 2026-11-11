import json

from scripts.compare_shoppingbench_product_query_fallback import compare


def write_rows(path, *, candidate=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(16):
        fallback = index < 4
        raw_plan = (
            "<plan><search>PLAIN_QUERY_OR_JSON_FILTER</search></plan>"
            if fallback
            else f"<plan><search>model query {index}</search></plan>"
        )
        compiled_query = (
            f"Item {index}; it should have color black.?"
            if fallback
            else f"model query {index}"
        )
        failed = fallback and not candidate
        trace = {
            "raw_plan_output": raw_plan,
            "planner_output_override_used": True,
            "plan_override_used": False,
            "contract_protocol": {
                "plan_compiler": {
                    "enabled": True,
                    "applied": True,
                    "request_kind": "product",
                    "queries": [compiled_query if candidate else "PLAIN_QUERY_OR_JSON_FILTER"],
                    "gold_used": False,
                    **(
                        {
                            "product_query_fallback": {
                                "applied": fallback,
                                "gold_used": False,
                            }
                        }
                        if candidate
                        else {}
                    ),
                }
            },
            "nodes_execution": [
                {
                    "type": "search",
                    "executed_query": (
                        compiled_query if candidate else "PLAIN_QUERY_OR_JSON_FILTER"
                    ),
                }
            ],
        }
        rows.append({
            "sample_id": f"product-{index:04d}",
            "query": f"Find it. Product details: Item {index}; it should have color black.",
            "format_error": "missing" if failed else None,
            "contract_error": None,
            "pipeline_error": None,
            "request_error": None,
            "score": {
                "exact_id_fraction": 0.0 if failed else 1.0,
                "exact_sequence_success": not failed,
            },
            "raw_trace": trace,
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_product_query_fallback_comparator_passes_strict_identity(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_rows(baseline)
    write_rows(candidate, candidate=True)

    report = compare(baseline, candidate, tmp_path / "report")

    assert report["passed"] is True
    assert report["activation_pass"] is True
    assert report["baseline"]["failures"] == 4
    assert report["candidate"]["failures"] == 0
    assert sum(row["fallback_expected"] for row in report["paired"]) == 4


def test_product_query_fallback_comparator_replays_runtime_url_cleaning(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_rows(baseline)
    write_rows(candidate, candidate=True)

    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    rows[0]["query"] = (
        "Find it. Product details: Item 0; it should have color black + white."
    )
    compiler = rows[0]["raw_trace"]["contract_protocol"]["plan_compiler"]
    compiler["queries"] = ["Item 0; it should have color black white.?"]
    rows[0]["raw_trace"]["nodes_execution"][0]["executed_query"] = compiler[
        "queries"
    ][0]
    candidate.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    baseline_rows = [json.loads(line) for line in baseline.read_text().splitlines()]
    baseline_rows[0]["query"] = rows[0]["query"]
    baseline.write_text(
        "".join(json.dumps(row) + "\n" for row in baseline_rows), encoding="utf-8"
    )

    report = compare(baseline, candidate, tmp_path / "report")

    assert report["passed"] is True
    assert report["paired"][0]["expected_query"] == "Item 0; it should have color black white.?"
