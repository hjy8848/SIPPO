import asyncio
import json

from scripts.audit_dagent_ccw_agent_roles import (
    AgentRoleReplayer,
    LoggedTrajectory,
    _make_case,
    load_role_trajectories,
    parse_plan_nodes,
    plan_errors,
)


def make_trajectory(advantage=1.0):
    return LoggedTrajectory(
        trajectory_id="2wiki:q1:rollout:0",
        question_group_id="2wiki:q1",
        rollout_index=0,
        dataset="2wiki",
        question="Where was Ada Lovelace born?",
        answers=["London"],
        logged_answer="London",
        logged_em=1,
        nodes=[],
        rloo_advantage=advantage,
    )


def test_parse_and_validate_runtime_plan_contract():
    nodes = parse_plan_nodes(
        '<plan><search id="1">Ada Lovelace birthplace</search>'
        '<reason id="2">Return the birthplace from #1#</reason></plan>'
    )
    assert nodes == [
        {"type": "search", "id": "1", "task": "Ada Lovelace birthplace"},
        {"type": "reason", "id": "2", "task": "Return the birthplace from #1#"},
    ]
    assert plan_errors("Where was Ada Lovelace born?", nodes, max_nodes=5) == []


def test_plan_gate_rejects_future_reference_and_nonconsecutive_ids():
    nodes = [
        {"type": "search", "id": "1", "task": "Ada Lovelace birthplace"},
        {"type": "reason", "id": "3", "task": "Use #4#"},
    ]
    errors = plan_errors("Where was Ada Lovelace born?", nodes, max_nodes=5)
    assert any("future/current" in error for error in errors)
    assert any("not consecutive" in error for error in errors)


def test_role_case_uses_rloo_sign_and_positive_bounded_weight():
    case = _make_case(
        role="planner",
        case_id="planner-case",
        trajectory=make_trajectory(advantage=-1 / 3),
        event={"response": "<plan></plan>", "metadata": {}},
        samples=[
            {"factual_em": 1, "counterfactual_em": 0},
            {"factual_em": 1, "counterfactual_em": 0},
            {"factual_em": 1, "counterfactual_em": 0},
            {"factual_em": 1, "counterfactual_em": 0},
        ],
        metadata={},
    )
    assert case["estimate_delta"] == 1
    assert case["validation_delta"] == 1
    assert case["signed_credit"] == -1
    assert case["weight"] == 2 / 3


def test_load_role_trajectories_preserves_complete_rloo_group(tmp_path):
    rows = []
    rewards = [1, 0, 0, 0]
    for rollout_index, reward in enumerate(rewards):
        rows.append(
            {
                "sample_index": "q1",
                "rollout_index": rollout_index,
                "question": "Where was Ada Lovelace born?",
                "gold_answer": ["London"],
                "pred_answer": "London" if reward else "Paris",
                "em": reward,
                "raw_trace": {
                    "nodes_execution": [],
                    "execution_error": "invalid plan",
                    "sft_events": [
                        {
                            "agent_name": "PlanningAgent",
                            "messages": [{"role": "user", "content": "question"}],
                            "response": "<plan></plan>",
                            "metadata": {},
                        }
                    ],
                },
            }
        )
    path = tmp_path / "full_logs_2wiki.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    loaded = load_role_trajectories([path], {"2wiki"}, 4, 4)
    assert len(loaded) == 4
    advantages = {
        row.trajectory.rollout_index: row.trajectory.rloo_advantage for row in loaded
    }
    assert advantages == {0: 1.0, 1: -1 / 3, 2: -1 / 3, 3: -1 / 3}
    assert all(row.planner_event is not None for row in loaded)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def generate_sample(self, *args, **kwargs):
        return next(self.responses)


def test_summarizer_counterfactual_follows_real_retry_rule():
    replayer = AgentRoleReplayer(
        FakeClient(["<answer>London</answer>"]),
        prompt_guard=None,
        rewrite_retries=2,
        temperature=0.7,
        top_p=0.95,
    )
    event = {
        "messages": [{"role": "user", "content": "summarize"}],
        "metadata": {"attempt": 0},
    }
    result = asyncio.run(
        replayer.evaluate_summary_response(
            make_trajectory(),
            event,
            "not valid xml",
            seed=7,
        )
    )
    assert result == {"answer": "London", "em": 1, "retried": True}
