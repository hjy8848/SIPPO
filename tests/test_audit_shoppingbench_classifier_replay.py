import json

from scripts.audit_shoppingbench_classifier_replay import replay


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_replay_changes_only_route_and_exposes_missing_atomic_queries(tmp_path):
    run = tmp_path / "run"
    fixtures = {
        "product": ("Find a matching product.", "<plan><search>item</search></plan>", "product"),
        "shop": (
            "Find one store for all requested items.",
            "<plan><search>combined items</search></plan>",
            "product",
        ),
        "voucher": (
            "Purchase terms: budget 100 with a voucher discount.",
            "<plan><search>item</search></plan>",
            "voucher",
        ),
    }
    for task, (query, raw_plan, old_kind) in fixtures.items():
        rows = []
        for position in range(16):
            rows.append({
                "sample_id": f"{task}-{position:04d}",
                "query": query,
                "raw_trace": {
                    "raw_plan_output": raw_plan,
                    "contract_protocol": {"plan_compiler": {"request_kind": old_kind}},
                },
            })
        write_jsonl(run / "grounding_pilot" / "contract" / task / "results.jsonl", rows)

    report = replay(run, tmp_path / "output")
    assert report["by_task"]["shop"]["route_corrected"] == 16
    assert report["by_task"]["shop"]["compile_succeeded"] == 0
    assert report["by_task"]["product"]["compile_succeeded"] == 16
    assert report["by_task"]["voucher"]["compile_succeeded"] == 16
    assert report["overall"]["route_regressed"] == 0
    assert report["model_calls"] == 0
    assert report["retrieval_executed"] is False
