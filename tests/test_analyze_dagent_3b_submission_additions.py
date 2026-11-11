from scripts.analyze_dagent_3b_submission_additions import (
    arm_metric_report,
    compute_row,
    graph_depth,
    parse_plan_nodes,
)


def test_arm_metric_report_has_micro_and_macro_metrics() -> None:
    arm = {
        dataset: [{"em": float(index % 2), "f1": index / 10.0} for index in range(size)]
        for dataset, size in {
            "2wikimultihopqa": 2,
            "hotpotqa": 2,
            "musique": 2,
            "bamboogle": 2,
        }.items()
    }
    report = arm_metric_report(arm)
    assert report["micro"]["samples"] == 8
    assert report["micro"]["em"] == 0.5
    assert report["macro"]["f1"] == 0.05


def test_plan_depth_and_compute_counts() -> None:
    row = {
        "retrieval_calls": 2,
        "error": None,
        "raw_trace": {
            "plan_output": (
                '<plan><search id="1">first</search>'
                '<search id="2">use #1#</search>'
                '<reason id="3">merge #1# and #2#</reason></plan>'
            ),
            "plan_warnings": [],
            "nodes_execution": [
                {"type": "search", "attempted_queries": ["a", "b"], "rewrite_queries": ["b"]},
                {"type": "search", "attempted_queries": ["c"], "rewrite_queries": []},
                {"type": "reason"},
            ],
            "sft_events": [
                {"agent_name": "PlanningAgent", "response": "abc", "metadata": {}},
                {"agent_name": "QueryRewriteAgent", "response": "def", "metadata": {}},
                {"agent_name": "AnswerGenerationAgent", "response": "ghi", "metadata": {"retry": 1}},
            ],
        },
    }
    nodes = parse_plan_nodes(row["raw_trace"])
    assert graph_depth(nodes) == 3
    result = compute_row(row, len)
    assert result["planned_search_nodes"] == 2
    assert result["executed_reason_nodes"] == 1
    assert result["retrieval_retry_queries"] == 1
    assert result["query_rewrites"] == 1
    assert result["answer_generation_retries"] == 1
    assert result["role_output_tokens"]["PlanningAgent"] == 3
