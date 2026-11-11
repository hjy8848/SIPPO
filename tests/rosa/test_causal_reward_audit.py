from __future__ import annotations

import asyncio

import pytest

from scripts.audit_dagent_causal_rewards import (
    FixedPlanReplayer,
    Intervention,
    Trajectory,
    candidate_report,
    compact_phase2_messages,
    truncate_docs_for_prompt,
)


class FakeAuditClient:
    def __init__(self) -> None:
        self.generated_systems = []
        self.generated_messages = []

    async def retrieve(self, query: str) -> list[str]:
        return [f"retrieved evidence for {query}"]

    async def generate(self, messages, max_tokens=384) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        self.generated_systems.append(system)
        self.generated_messages.append(messages)
        if "Rewrite a dependent search query" in system:
            value = user.split("# 1 = ", 1)[1].splitlines()[0]
            return f"second fact about {value}"
        if "Answer the current sub-question" in system:
            if "COUNTERFACTUAL" in user:
                return "<conclusion>changed-root</conclusion>"
            if "second fact about changed-root" in user:
                return "<conclusion>changed-child</conclusion>"
            if "second fact about factual-root" in user:
                return "<conclusion>factual-child</conclusion>"
            return "<conclusion>factual-root</conclusion>"
        if "Resolve the logic task" in system:
            if "changed-child" in user:
                return "<conclusion>changed-reason</conclusion>"
            return "<conclusion>factual-reason</conclusion>"
        if "Synthesize the resolved sub-questions" in system:
            if "changed-reason" in user:
                return "<answer>changed</answer>"
            return "<answer>factual</answer>"
        raise AssertionError(system)


def test_replacing_first_search_replays_every_later_node_and_summary():
    trajectory = Trajectory(
        trajectory_id="2wiki:1",
        dataset="2wiki",
        question="What is the result?",
        answers=["factual"],
        logged_answer="factual",
        logged_em=1,
        nodes=[
            {"id": "1", "type": "search", "task": "first fact", "conclusion": ""},
            {"id": "2", "type": "search", "task": "second fact about #1#", "conclusion": ""},
            {"id": "3", "type": "reason", "task": "derive from #2#", "conclusion": ""},
        ],
    )
    client = FakeAuditClient()
    replayer = FixedPlanReplayer(client, rewrite_retries=0)

    factual = asyncio.run(replayer.replay(trajectory))
    counterfactual = asyncio.run(
        replayer.replay(
            trajectory,
            Intervention("replace_docs", "1", ["COUNTERFACTUAL"]),
        )
    )

    assert factual["final_answer"] == "factual"
    assert counterfactual["final_answer"] == "changed"
    assert counterfactual["nodes"][1]["resolved_task"] == "second fact about changed-root"
    assert counterfactual["nodes"][2]["resolved_task"] == "derive from changed-child"
    assert len(counterfactual["nodes"]) == 3


def test_later_search_uses_training_state_and_compacts_old_documents():
    trajectory = Trajectory(
        trajectory_id="2wiki:state",
        dataset="2wiki",
        question="What is the result?",
        answers=["factual"],
        logged_answer="factual",
        logged_em=1,
        nodes=[
            {"id": "1", "type": "search", "task": "first fact", "conclusion": ""},
            {"id": "2", "type": "search", "task": "second fact", "conclusion": ""},
        ],
    )
    client = FakeAuditClient()

    asyncio.run(FixedPlanReplayer(client, rewrite_retries=0).replay(trajectory))

    search_calls = [
        messages
        for messages in client.generated_messages
        if "Answer the current sub-question" in messages[0]["content"]
    ]
    assert len(search_calls) == 2
    second_call = search_calls[1]
    assert [message["role"] for message in second_call] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "factual-root" in second_call[2]["content"]
    assert "[omitted from previous turn]" in second_call[1]["content"]
    assert "retrieved evidence for first fact" not in second_call[1]["content"]
    assert "retrieved evidence for second fact" in second_call[-1]["content"]


def test_force_query_intervention_skips_rewriter_and_replays_descendants():
    class ParaphrasingClient(FakeAuditClient):
        async def generate(self, messages, max_tokens=384) -> str:
            system = messages[0]["content"]
            user = messages[-1]["content"]
            self.generated_systems.append(system)
            self.generated_messages.append(messages)
            if "Rewrite a dependent search query" in system:
                return "paraphrased dependency query"
            if "Answer the current sub-question" in system:
                if "paraphrased dependency query" in user:
                    return "<conclusion>rewrite-result</conclusion>"
                if "second fact about factual-root" in user:
                    return "<conclusion>direct-result</conclusion>"
                return "<conclusion>factual-root</conclusion>"
            if "Resolve the logic task" in system:
                return f"<conclusion>{user}</conclusion>"
            if "Synthesize the resolved sub-questions" in system:
                return "<answer>done</answer>"
            raise AssertionError(system)

    trajectory = Trajectory(
        trajectory_id="2wiki:rewrite",
        dataset="2wiki",
        question="What is the result?",
        answers=["done"],
        logged_answer="done",
        logged_em=1,
        nodes=[
            {"id": "1", "type": "search", "task": "first fact", "conclusion": ""},
            {"id": "2", "type": "search", "task": "second fact about #1#", "conclusion": ""},
            {"id": "3", "type": "reason", "task": "derive #2#", "conclusion": ""},
        ],
    )
    client = ParaphrasingClient()
    replayer = FixedPlanReplayer(client, rewrite_retries=0)

    baseline = asyncio.run(replayer.replay(trajectory))
    direct = asyncio.run(
        replayer.replay(
            trajectory,
            Intervention(
                "force_query",
                "2",
                query="second fact about factual-root",
            ),
        )
    )

    assert baseline["nodes"][1]["executed_query"] == "paraphrased dependency query"
    assert direct["nodes"][1]["executed_query"] == "second fact about factual-root"
    assert direct["nodes"][1]["dependency_rewrite_attempted"] is False
    assert direct["nodes"][2]["resolved_task"] == "derive direct-result"


def test_prompt_helpers_match_training_document_compaction_and_fallback():
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "**Current Sub-question**: old\n\n**Search Context**:\nsecret docs",
        },
        {"role": "assistant", "content": "<conclusion>old fact</conclusion>"},
    ]

    compacted = compact_phase2_messages(messages)
    truncated = truncate_docs_for_prompt(["x" * 1300])

    assert compacted[1]["content"] == (
        "**Current Sub-question**: old\n\n"
        "**Search Context**:\n[omitted from previous turn]"
    )
    assert compacted[2] == messages[2]
    assert truncated[0].startswith("x" * 1200)
    assert truncated[0].endswith("\n[...truncated...]")


def test_empty_search_action_becomes_insufficient_instead_of_dropping_trajectory():
    class EmptyActionClient(FakeAuditClient):
        async def generate(self, messages, max_tokens=384) -> str:
            system = messages[0]["content"]
            if "Answer the current sub-question" in system:
                return ""
            if "Synthesize the resolved sub-questions" in system:
                return "<answer>unknown</answer>"
            return await super().generate(messages, max_tokens)

    trajectory = Trajectory(
        trajectory_id="hotpotqa:2",
        dataset="hotpotqa",
        question="What is the fact?",
        answers=["gold"],
        logged_answer="",
        logged_em=0,
        nodes=[
            {"id": "1", "type": "search", "task": "find fact", "conclusion": ""},
        ],
    )

    replay = asyncio.run(
        FixedPlanReplayer(EmptyActionClient(), rewrite_retries=1).replay(trajectory)
    )

    assert replay["nodes"][0]["conclusion"] == "Insufficient information"
    assert replay["em"] == 0


def _synthetic_result(split: str, dataset: str, index: int, positive: bool):
    if positive:
        score, factual, counterfactual = 1.0, 1, 0
    else:
        score, factual, counterfactual = -1.0, 0, 1
    case = {
        "case_id": f"{dataset}:{split}:{index}",
        "reward_score": score,
        "causal_delta_em": factual - counterfactual,
        "factual_em": factual,
        "counterfactual_em": counterfactual,
    }
    return {
        "trajectory_id": f"{dataset}:{split}:{index}",
        "dataset": dataset,
        "split": split,
        "search_ig_cases": [case],
    }


def test_acceptance_gate_uses_disjoint_test_causal_branch_selection():
    results = []
    for dataset in ("2wiki", "hotpotqa"):
        for split in ("calibration", "test"):
            for index in range(12):
                results.append(
                    _synthetic_result(split, dataset, index, positive=index % 2 == 0)
                )

    report = candidate_report(
        results,
        "search_ig_cases",
        seed=42,
        minimum_changed_test=20,
    )

    assert report["passed"] is True
    assert report["test"]["direction_accuracy"] == pytest.approx(1.0)
    assert report["test"]["selection_em_gain_over_random"] == pytest.approx(0.5)
    assert report["test_by_dataset"]["2wiki"]["changed_pairs"] == 12
    assert report["test_by_dataset"]["hotpotqa"]["changed_pairs"] == 12


def test_acceptance_gate_rejects_reward_without_causal_outcome_changes():
    rows = []
    for index in range(24):
        rows.append(
            {
                "trajectory_id": f"2wiki:test:{index}",
                "dataset": "2wiki",
                "split": "test" if index % 2 else "calibration",
                "search_ig_cases": [
                    {
                        "case_id": f"case:{index}",
                        "reward_score": float(index),
                        "causal_delta_em": 0,
                        "factual_em": 1,
                        "counterfactual_em": 1,
                    }
                ],
            }
        )

    report = candidate_report(
        rows,
        "search_ig_cases",
        seed=42,
        minimum_changed_test=1,
    )

    assert report["passed"] is False
    assert report["test"]["changed_pairs"] == 0
