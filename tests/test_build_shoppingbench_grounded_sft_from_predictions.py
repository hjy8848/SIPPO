import json

from scripts.build_shoppingbench_grounded_sft_from_predictions import (
    build,
    validate_prediction,
)


def prediction(task="product", product_ids=None):
    product_ids = product_ids or ["100"]
    searches = []
    events = [
        {
            "agent_name": "PlanningAgent",
            "messages": [{"role": "system", "content": "plan"}, {"role": "user", "content": "question"}],
            "response": "<invalid-model-plan>",
            "metadata": {},
        }
    ]
    for index, product_id in enumerate(product_ids, start=1):
        searches.append(
            {
                "id": str(index),
                "type": "search",
                "search_context": f"[1_1]: Product ID: {product_id}\nTitle: Product {product_id}",
            }
        )
        events.append(
            {
                "agent_name": "AnswerGenerationAgent",
                "messages": [{"role": "system", "content": "search"}, {"role": "user", "content": "context"}],
                "response": f"<conclusion>Product ID: {product_id}</conclusion>",
                "metadata": {"node_id": str(index)},
            }
        )
    if task != "product":
        reason_id = str(len(product_ids) + 1)
        searches.append({"id": reason_id, "type": "reason"})
        events.append(
            {
                "agent_name": "ReasonerAgent",
                "messages": [{"role": "system", "content": "reason"}, {"role": "user", "content": "artifacts"}],
                "response": "<conclusion>" + " ".join(product_ids) + "</conclusion>",
                "metadata": {"node_id": reason_id},
            }
        )
    sink = len(searches)
    plan = '<plan sink="#%d#" answer_type="product_ids">\n' % sink
    plan += "\n".join(f'<search id="{i}" subject="catalog" relation="catalog_candidates">q{i}</search>' for i in range(1, len(product_ids) + 1))
    if task != "product":
        plan += f'\n<reason id="{sink}" op="join_shop" inputs="#1#" return="product_ids">select</reason>'
    plan += "\n</plan>"
    return {
        "sample_id": f"{task}-0000",
        "arm": "contract",
        "query": "visible query",
        "expected_product_ids": product_ids,
        "predicted_product_ids": product_ids,
        "score": {"exact_sequence_success": True},
        "format_error": None,
        "contract_error": None,
        "pipeline_error": None,
        "request_error": None,
        "raw_trace": {
            "protocol_profile": "shoppingbench",
            "ablation_config": {
                "contract_preserving_dag_v1": True,
                "shopping_system_plan_compiler_v1": True,
            },
            "plan_override_used": False,
            "plan_output": plan,
            "contract_protocol": {
                "validation_errors": [],
                "plan_compiler": {
                    "applied": True,
                    "gold_used": False,
                    "request_kind": task,
                },
            },
            "nodes_execution": searches,
            "sft_events": events,
        },
    }


def write_inputs(root, rows_by_task):
    for task, rows in rows_by_task.items():
        path = root / task
        path.mkdir(parents=True)
        (path / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_accepts_grounded_exact_trace_and_replaces_invalid_planner_target(tmp_path):
    rows = {"product": [prediction()], "shop": [prediction("shop", ["100", "200"])], "voucher": [prediction("voucher")]}
    input_root = tmp_path / "predictions"
    write_inputs(input_root, rows)

    report, passed = build(input_root, tmp_path / "output", expected_per_task=1, min_total=3, min_per_task=1)

    assert passed is True
    assert report["accepted_by_task"] == {"product": 1, "shop": 1, "voucher": 1}
    events = [json.loads(line) for line in (tmp_path / "output" / "sft_events.jsonl").read_text().splitlines()]
    planner = next(row for row in events if row["agent_name"] == "PlanningAgent")
    assert planner["messages"][-1]["content"].startswith('<plan sink="#')
    assert planner["target_source"] == "system_compiled_model_search_semantics"
    assert "invalid-model-plan" not in json.dumps(planner)


def test_rejects_gold_missing_from_corresponding_search_or_runtime_gold():
    row = prediction()
    row["raw_trace"]["nodes_execution"][0]["search_context"] = "Product ID: 999"
    assert "gold product 100 missing from corresponding Search" in validate_prediction(row, "product")

    row = prediction()
    row["raw_trace"]["contract_protocol"]["plan_compiler"]["gold_used"] = True
    assert "system compiler did not prove no-gold application" in validate_prediction(row, "product")


def test_deterministic_reason_needs_no_fabricated_sft_event():
    row = prediction("shop", ["100", "200"])
    reason = row["raw_trace"]["nodes_execution"][-1]
    reason["reason_execution_method"] = "deterministic_shop_join"
    row["raw_trace"]["sft_events"] = [
        event
        for event in row["raw_trace"]["sft_events"]
        if event["agent_name"] != "ReasonerAgent"
    ]

    assert "at least one Reason lacks an SFT event" not in validate_prediction(row, "shop")

    reason["reason_execution_method"] = ""
    assert "at least one Reason lacks an SFT event" in validate_prediction(row, "shop")
