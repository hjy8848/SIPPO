import json

from scripts.analyze_shoppingbench_grounded_pilot import analyze


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_slices_route_and_search_count_failures(tmp_path):
    runtime = tmp_path / "runtime"
    run = tmp_path / "run"
    templates = {
        "product": "Find an item.",
        "shop": "Find one store for all requested items.",
        "voucher": "Find products. Purchase terms: budget 100; voucher discount 10.",
    }
    for task, query in templates.items():
        source_rows = []
        result_rows = []
        count = 1 if task == "product" else 2
        for position in range(16):
            rewards = [{"product_id": str(position * 10 + item + 1)} for item in range(count)]
            source_rows.append({"query": query, "reward": rewards[0] if task == "product" else rewards})
            searches = [
                {"type": "search", "search_context": f"Product ID: {rewards[0]['product_id']}"}
            ]
            result_rows.append({
                "sample_id": f"{task}-{position:04d}", "position": position, "query": query,
                "expected_product_ids": [row["product_id"] for row in rewards],
                "score": {"exact_sequence_success": task == "product"},
                "format_error": None, "contract_error": None, "pipeline_error": None, "request_error": None,
                "raw_trace": {"nodes_execution": searches, "sft_events": [], "contract_protocol": {
                    "plan_compiler": {"applied": True, "request_kind": task, "queries": ["q"]}
                }},
            })
        write_jsonl(runtime / "source" / "data" / f"synthesize_{task}_test.jsonl", source_rows)
        write_jsonl(run / "grounding_pilot" / "contract" / task / "results.jsonl", result_rows)

    report = analyze(run, runtime, tmp_path / "output")
    assert report["rows"] == 48
    assert report["by_task"]["product"]["search_count_sufficient"] == 16
    assert report["by_task"]["shop"]["search_count_sufficient"] == 0
    assert report["by_task"]["voucher"]["search_count_sufficient"] == 0
    assert report["defect_overlap"]["route_pass__topology_fail"] == 32
    assert report["training_authorized"] is False
