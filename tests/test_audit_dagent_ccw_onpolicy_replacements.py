import asyncio
import json

from scripts.audit_dagent_ccw_onpolicy_replacements import (
    _flat_replacement_statistics,
    is_nonpolicy_environment_node,
    load_replacement_trajectories,
    replace_target_node,
    sample_valid_replacement,
)


class FakeReplayer:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def sample(self, *args, **kwargs):
        return next(self.responses)


def test_valid_replacement_rejects_invalid_protocol_output():
    event = {
        "messages": [{"role": "user", "content": "reason"}],
        "response": "<conclusion>observed</conclusion>",
    }
    result = asyncio.run(
        sample_valid_replacement(
            FakeReplayer(["no xml", "<conclusion>alternative</conclusion>"]),
            event,
            seed=10,
            maximum_attempts=2,
        )
    )
    assert result["valid"] is True
    assert result["fallback_to_factual"] is False
    assert result["conclusion"] == "alternative"
    assert len(result["attempts"]) == 2


def test_invalid_replacements_fall_back_to_identity_not_missing_marker():
    event = {
        "messages": [{"role": "user", "content": "reason"}],
        "response": "<conclusion>observed</conclusion>",
    }
    result = asyncio.run(
        sample_valid_replacement(
            FakeReplayer(["no xml", "<conclusion>Insufficient information</conclusion>"]),
            event,
            seed=10,
            maximum_attempts=2,
        )
    )
    assert result["valid"] is False
    assert result["fallback_to_factual"] is True
    assert result["response"] == event["response"]
    assert result["conclusion"] == "observed"


def test_replace_target_node_does_not_mutate_factual_nodes():
    factual = [
        {
            "id": "1",
            "type": "search",
            "raw_output": "<conclusion>old</conclusion>",
            "conclusion": "old",
        },
        {
            "id": "2",
            "type": "reason",
            "raw_output": "<conclusion>final</conclusion>",
            "conclusion": "final",
        },
    ]
    counterfactual = replace_target_node(
        factual,
        "1",
        {
            "response": "<conclusion>new</conclusion>",
            "conclusion": "new",
        },
    )
    assert factual[0]["conclusion"] == "old"
    assert counterfactual[0]["conclusion"] == "new"
    assert counterfactual[1] == factual[1]


def test_identity_replacement_is_reported_as_negative_control():
    sample = {
        "replacement_valid": True,
        "fallback_to_factual": False,
        "same_response": True,
        "same_normalized_conclusion": True,
        "factual_em": 1,
        "counterfactual_em": 0,
        "replacement_attempt_count": 1,
    }
    results = [
        {
            "cases": [
                {
                    "skipped_zero_advantage": False,
                    "samples": [sample],
                }
            ]
        }
    ]
    report = _flat_replacement_statistics(results)
    assert report["identity_response_samples"] == 1
    assert report["identity_response_terminal_em_changed_samples"] == 1


def test_loader_joins_exact_final_action_event_after_retry(tmp_path):
    rows = []
    for rollout_index, reward in enumerate([1, 0, 0, 0]):
        final_response = f"<conclusion>answer-{rollout_index}</conclusion>"
        events = [
            {
                "agent_name": "AnswerGenerationAgent",
                "messages": [{"role": "user", "content": "first"}],
                "response": "<rewrite>better query</rewrite>",
                "metadata": {"node_id": "1", "retry": 0},
            },
            {
                "agent_name": "AnswerGenerationAgent",
                "messages": [{"role": "user", "content": "second"}],
                "response": final_response,
                "metadata": {"node_id": "1", "retry": 1},
            },
        ]
        rows.append(
            {
                "sample_index": "q1",
                "rollout_index": rollout_index,
                "question": "question",
                "gold_answer": ["answer-0"],
                "pred_answer": "answer-0" if reward else "wrong",
                "em": reward,
                "raw_trace": {
                    "nodes_execution": [
                        {
                            "id": "1",
                            "type": "search",
                            "query": "query",
                            "conclusion": f"answer-{rollout_index}",
                            "raw_output": final_response,
                            "search_context": "",
                        }
                    ],
                    "sft_events": events,
                },
            }
        )
    path = tmp_path / "full_logs_2wiki.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    loaded = load_replacement_trajectories([path], {"2wiki"}, 4, 4)
    assert len(loaded) == 4
    for row in loaded:
        event = row.events_by_node["1"]
        assert event["metadata"]["retry"] == 1
        assert event["response"] == row.trajectory.nodes[0]["raw_output"]


def test_loader_excludes_deterministic_reason_without_policy_event(tmp_path):
    rows = []
    for rollout_index, reward in enumerate([1, 0, 0, 0]):
        search_response = f"<conclusion>country-{rollout_index}</conclusion>"
        rows.append(
            {
                "sample_index": "q1",
                "rollout_index": rollout_index,
                "question": "same country?",
                "gold_answer": ["yes"],
                "pred_answer": "yes" if reward else "no",
                "em": reward,
                "raw_trace": {
                    "nodes_execution": [
                        {
                            "id": "1",
                            "type": "search",
                            "query": "country",
                            "conclusion": f"country-{rollout_index}",
                            "raw_output": search_response,
                            "search_context": "",
                        },
                        {
                            "id": "2",
                            "type": "reason",
                            "query": "compare",
                            "conclusion": "yes",
                            "raw_output": "<conclusion>yes</conclusion>",
                            "reason_execution_method": "deterministic_boolean",
                        },
                    ],
                    "sft_events": [
                        {
                            "agent_name": "AnswerGenerationAgent",
                            "messages": [{"role": "user", "content": "country"}],
                            "response": search_response,
                            "metadata": {"node_id": "1", "retry": 0},
                        }
                    ],
                },
            }
        )
    path = tmp_path / "full_logs_hotpotqa.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    loaded = load_replacement_trajectories([path], {"hotpotqa"}, 4, 4)

    assert len(loaded) == 4
    assert all(set(row.events_by_node) == {"1"} for row in loaded)


def test_unresolved_dependency_search_is_environment_skip_not_policy_action():
    assert is_nonpolicy_environment_node(
        {
            "type": "search",
            "conclusion": "Insufficient information",
            "executed_query": "",
            "attempted_queries": [],
            "raw_search_outputs": [],
            "handoff_dependency_error": (
                "contract query compiler could not resolve the typed subject"
            ),
        }
    )
    assert not is_nonpolicy_environment_node(
        {
            "type": "search",
            "conclusion": "Insufficient information",
            "executed_query": "query was attempted",
            "attempted_queries": ["query was attempted"],
            "handoff_dependency_error": "",
        }
    )
