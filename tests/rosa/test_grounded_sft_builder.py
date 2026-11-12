from copy import deepcopy

import pandas as pd

from scripts.generate_rosa_grounded_sft import (
    event_to_sft_row,
    selected_sft_events,
    validate_event_coverage,
    validate_trace,
    write_sft_outputs,
)


QUESTION = "Who was the father of Ada Lovelace's husband?"
PLAN = (
    '<plan>\n'
    '<search id="1">Who was Ada Lovelace\'s husband?</search>\n'
    '<search id="2">Who was the father of #1#?</search>\n'
    '</plan>'
)


def event(agent_name, messages, response, **metadata):
    return {
        "agent_name": agent_name,
        "messages": messages,
        "response": response,
        "metadata": metadata,
    }


def valid_trace():
    planner_messages = [
        {"role": "system", "content": "planner"},
        {"role": "user", "content": f"Question: {QUESTION}\nGenerate the execution plan."},
    ]
    search_one_messages = [
        {"role": "system", "content": "search"},
        {"role": "user", "content": "Ada Lovelace husband context: William King-Noel"},
    ]
    search_two_messages = [
        {"role": "system", "content": "search"},
        {"role": "user", "content": "previous context"},
        {"role": "assistant", "content": "<conclusion>William King-Noel</conclusion>"},
        {"role": "user", "content": "William King-Noel father context: Peter King"},
    ]
    final_messages = [
        {"role": "system", "content": "final"},
        {"role": "user", "content": "Resolved conclusions: William King-Noel; Peter King"},
    ]
    return {
        "question": QUESTION,
        "plan_output": PLAN,
        "plan_warnings": [],
        "nodes_execution": [
            {
                "id": "1",
                "type": "search",
                "query": "Who was Ada Lovelace's husband?",
                "executed_query": "Who was Ada Lovelace's husband?",
                "search_context": "Ada Lovelace married William King-Noel in 1835.",
                "conclusion": "William King-Noel",
            },
            {
                "id": "2",
                "type": "search",
                "query": "Who was the father of #1#?",
                "executed_query": "Who was the father of William King-Noel?",
                "search_context": "William King-Noel was the son of Peter King.",
                "conclusion": "Peter King",
            },
        ],
        "final_answer": "Peter King",
        "evaluation_mode": "training_protocol",
        "verification": {"grounded_by_nodes": True, "direct_fallback_used": False},
        "ablation_config": {
            "disable_query_rewrite": False,
            "disable_final_summarizer": False,
            "linearized_execution": False,
        },
        "sft_events": [
            event("PlanningAgent", planner_messages, PLAN),
            event(
                "AnswerGenerationAgent",
                search_one_messages,
                "<conclusion>William King-Noel</conclusion>",
                node_id="1",
            ),
            event(
                "QueryRewriteAgent",
                [{"role": "system", "content": "rewrite"}, {"role": "user", "content": "draft"}],
                "Who was the father of William King-Noel?",
                node_id="2",
            ),
            event(
                "AnswerGenerationAgent",
                search_two_messages,
                "<conclusion>Peter King</conclusion>",
                node_id="2",
            ),
            event("AnswerSummarizationAgent", final_messages, "<answer>Peter King</answer>", attempt=0),
        ],
    }


def test_strict_trace_accepts_grounded_execution():
    trace = valid_trace()
    reasons, diagnostics = validate_trace(QUESTION, ["Peter King"], trace, max_nodes=5)
    events = selected_sft_events(trace)

    assert reasons == []
    assert validate_event_coverage(trace, events) == []
    assert diagnostics["search_nodes"] == 2


def test_strict_trace_rejects_unsupported_search_conclusion():
    trace = valid_trace()
    trace["nodes_execution"][1]["search_context"] = "No parent is named in this document."

    reasons, _ = validate_trace(QUESTION, ["Peter King"], trace, max_nodes=5)

    assert any("not stated in its retrieved context" in reason for reason in reasons)


def test_only_current_assistant_response_receives_loss():
    trace = valid_trace()
    search_event = trace["sft_events"][3]

    row = event_to_sft_row(search_event, "trajectory-1", QUESTION, 0)

    assistant_messages = [message for message in row["messages"] if message["role"] == "assistant"]
    assert assistant_messages[0]["loss_mask"] == 0
    assert assistant_messages[-1]["loss_mask"] == 1


def test_sft_outputs_split_by_trajectory(tmp_path):
    first = valid_trace()
    second = deepcopy(first)
    accepted = []
    for index, trace in enumerate([first, second]):
        accepted.append({
            "trajectory_id": f"trajectory-{index}",
            "source_index": index,
            "question": QUESTION,
            "selected_sft_events": selected_sft_events(trace),
        })

    stats = write_sft_outputs(
        accepted,
        tmp_path,
        seed=7,
        val_ratio=0.5,
        role_repeats={"PlanningAgent": 2},
    )
    train = pd.read_parquet(tmp_path / "sft_train.parquet")
    val = pd.read_parquet(tmp_path / "sft_val.parquet")

    assert stats["train_trajectories"] == 1
    assert stats["val_trajectories"] == 1
    assert set(train["trajectory_id"]).isdisjoint(set(val["trajectory_id"]))
    assert stats["role_counts"]["PlanningAgent"] == 4
