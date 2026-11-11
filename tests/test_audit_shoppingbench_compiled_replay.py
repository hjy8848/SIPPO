from scripts.audit_shoppingbench_compiled_replay import aggregate, audit_row


def test_retrieval_audit_separates_corresponding_and_any_node_hits():
    row = audit_row(
        {
            "sample_id": "shop-0000",
            "expected_product_ids": ["11", "22"],
            "predicted_product_ids": [],
            "raw_trace": {
                "nodes_execution": [
                    {"type": "search", "executed_query": "first", "search_context": "Product ID: 22"},
                    {"type": "search", "executed_query": "second", "search_context": "Product ID: 11"},
                ]
            },
        }
    )
    assert row["gold_in_corresponding_node"] == [False, False]
    assert row["gold_in_any_node"] == [True, True]
    assert row["failure_category"] == "cross_node_mismatch"
    report = aggregate([row])
    assert report["overall"]["corresponding_node_recall"] == 0.0
    assert report["overall"]["any_node_recall"] == 1.0
    assert report["overall"]["category_counts"]["cross_node_mismatch"] == 1
    assert report["gold_used_by_runtime"] is False


def make_row(
    *,
    expected=("11",),
    predicted=(),
    contexts=("Product ID: 11",),
    **overrides,
):
    row = {
        "sample_id": "product-0000",
        "expected_product_ids": list(expected),
        "predicted_product_ids": list(predicted),
        "raw_trace": {
            "nodes_execution": [
                {"type": "search", "search_context": context}
                for context in contexts
            ]
        },
    }
    row.update(overrides)
    return row


def test_failure_categories_are_mutually_exclusive_and_cover_all_stages():
    rows = [
        audit_row(make_row(pipeline_error="planner failed")),
        audit_row(make_row(predicted=("11",))),
        audit_row(make_row(contexts=("Product ID: 99",))),
        audit_row(
            make_row(
                expected=("11", "22"),
                contexts=("Product ID: 22", "Product ID: 11"),
            )
        ),
        audit_row(make_row(predicted=("99",))),
    ]
    assert [row["failure_category"] for row in rows] == [
        "execution_failure",
        "success",
        "retrieval_miss",
        "cross_node_mismatch",
        "selection_miss",
    ]
    report = aggregate(rows)
    assert report["overall"]["category_counts"] == {
        "execution_failure": 1,
        "success": 1,
        "retrieval_miss": 1,
        "cross_node_mismatch": 1,
        "selection_miss": 1,
    }
    assert sum(report["overall"]["category_counts"].values()) == 5


def test_execution_failure_takes_precedence_over_accidentally_exact_output():
    row = audit_row(make_row(predicted=("11",), request_error="timeout"))
    assert row["failure_category"] == "execution_failure"
    assert row["execution_errors"] == {"request_error": "timeout"}
