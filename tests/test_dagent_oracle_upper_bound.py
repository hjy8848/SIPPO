import asyncio
import json

from scripts.audit_dagent_ccw_paired_suffixes import LoggedTrajectory
from scripts.audit_dagent_oracle_upper_bound import (
    OracleCase,
    audit_case,
    canonical_reason_id,
    descendants_only,
    load_cases,
    oracle_reason_nodes,
    summarize,
    summarize_canonical_increment,
    validate_canonical_plan,
)


def canonical_nodes():
    return [
        {"id": "1", "type": "search", "task": "A date"},
        {"id": "2", "type": "search", "task": "B date"},
        {"id": "3", "type": "reason", "task": "Choose A or B from #1# and #2#"},
        {"id": "4", "type": "search", "task": "#3# birthplace"},
    ]


def metadata():
    return {
        "trajectory_id": "q1",
        "question": "Where was the older candidate born?",
        "answer": "Paris",
        "gold_answers": ["Paris"],
        "winner": "A",
        "source": {"source_dataset": "reason_search"},
        "plan_output": (
            '<plan><search id="1">A date</search><search id="2">B date</search>'
            '<reason id="3">Choose A or B from #1# and #2#</reason>'
            '<search id="4">#3# birthplace</search></plan>'
        ),
        "nodes_execution": [
            {"id": "1", "type": "search", "task": "A date", "conclusion": "1900"},
            {"id": "2", "type": "search", "task": "B date", "conclusion": "1950"},
            {
                "id": "3",
                "type": "reason",
                "task": "Choose A or B from #1# and #2#",
                "conclusion": "A",
            },
            {
                "id": "4",
                "type": "search",
                "task": "#3# birthplace",
                "conclusion": "Paris",
            },
        ],
    }


def model_nodes():
    return [
        {
            "id": "1",
            "type": "search",
            "task": "A date",
            "conclusion": "1900",
            "executed_query": "A date",
            "docs": [],
            "raw_output": "<conclusion>1900</conclusion>",
        },
        {
            "id": "2",
            "type": "search",
            "task": "B date",
            "conclusion": "1950",
            "executed_query": "B date",
            "docs": [],
            "raw_output": "<conclusion>1950</conclusion>",
        },
        {
            "id": "3",
            "type": "reason",
            "task": "Choose A or B from #1# and #2#",
            "conclusion": "B",
            "executed_query": "",
            "docs": [],
            "raw_output": "<conclusion>B</conclusion>",
        },
        {
            "id": "4",
            "type": "search",
            "task": "#3# birthplace",
            "conclusion": "Rome",
            "executed_query": "B birthplace",
            "docs": [],
            "raw_output": "<conclusion>Rome</conclusion>",
        },
    ]


def trajectory():
    return LoggedTrajectory(
        trajectory_id="q1",
        question_group_id="q1",
        rollout_index=0,
        dataset="reason_search",
        question="Where was the older candidate born?",
        answers=["Paris"],
        logged_answer="Rome",
        logged_em=0,
        nodes=model_nodes(),
    )


class FakeReplayer:
    def __init__(self):
        self.calls = []

    async def execute_nodes(
        self, trajectory, nodes, seed, regenerate_ids=None, initial_query_overrides=None
    ):
        self.calls.append((nodes, seed, set(regenerate_ids or set())))
        reason = next(node for node in nodes if node["type"] == "reason")
        answer = "Paris" if reason.get("conclusion") == "A" else "Rome"
        return {
            "answer": answer,
            "em": int(answer == "Paris"),
            "nodes": [
                {
                    **node,
                    "resolved_task": node.get("task", ""),
                    "regenerated": str(node["id"]) in set(regenerate_ids or set()),
                }
                for node in nodes
            ],
        }


def test_reason_oracle_preserves_upstream_and_replays_only_descendants():
    factual = model_nodes()
    oracle = oracle_reason_nodes(factual, "3", "A")
    assert factual[2]["conclusion"] == "B"
    assert oracle[2]["conclusion"] == "A"
    assert oracle[:2] == factual[:2]
    assert descendants_only(factual, "3") == {"4"}


def test_canonical_reason_must_select_winner_and_have_descendant():
    assert canonical_reason_id(metadata(), canonical_nodes()) == "3"
    validate_canonical_plan(metadata(), canonical_nodes(), "q1")


def test_loader_strictly_joins_eval_and_metadata(tmp_path):
    meta_path = tmp_path / "accepted_trajectories.jsonl"
    meta_path.write_text(json.dumps(metadata()) + "\n", encoding="utf-8")
    row = {
        "sample_index": "q1",
        "question": "Where was the older candidate born?",
        "gold_answer": ["Paris"],
        "pred_answer": "Rome",
        "em": 0,
        "raw_trace": {
            "nodes_execution": [
                {
                    "id": node["id"],
                    "type": node["type"],
                    "query": node["task"],
                    "conclusion": node["conclusion"],
                    "executed_query": node["executed_query"],
                    "search_context": "",
                    "raw_output": node["raw_output"],
                }
                for node in model_nodes()
            ]
        },
    }
    eval_path = tmp_path / "full_logs_reason.json"
    eval_path.write_text(json.dumps([row]), encoding="utf-8")
    cases, joins = load_cases([eval_path], meta_path, maximum_cases=0, only_logged_errors=False)
    assert len(cases) == 1
    assert cases[0].model_reason_id == "3"
    assert joins["selected_rows"] == 1


def test_loader_keeps_empty_model_plan_for_plan_oracle(tmp_path):
    meta_path = tmp_path / "accepted_trajectories.jsonl"
    meta_path.write_text(json.dumps(metadata()) + "\n", encoding="utf-8")
    eval_path = tmp_path / "full_logs_reason.json"
    eval_path.write_text(
        json.dumps(
            [
                {
                    "sample_index": "q1",
                    "question": "Where was the older candidate born?",
                    "gold_answer": ["Paris"],
                    "pred_answer": "",
                    "em": 0,
                    "raw_trace": {"nodes_execution": []},
                }
            ]
        ),
        encoding="utf-8",
    )
    cases, joins = load_cases([eval_path], meta_path, maximum_cases=0, only_logged_errors=False)
    assert len(cases) == 1
    assert cases[0].trajectory.nodes == []
    assert cases[0].model_reason_id is None
    assert joins["model_empty_plan_rows"] == 1


def test_reason_audit_has_exact_identity_and_positive_terminal_delta():
    case = OracleCase(
        trajectory=trajectory(),
        metadata=metadata(),
        canonical_nodes=canonical_nodes(),
        canonical_reason_id="3",
        model_reason_id="3",
    )
    replayer = FakeReplayer()
    result = asyncio.run(audit_case(case, replayer, samples=2, modes={"reason"}, seed=7))
    samples = result["modes"]["reason"]["samples"]
    assert all(sample["identity_match"] for sample in samples)
    assert all(sample["factual_em"] == 0 for sample in samples)
    assert all(sample["oracle_em"] == 1 for sample in samples)
    assert all(call[2] == {"4"} for call in replayer.calls)
    assert len(replayer.calls) == 8


def test_plan_and_plan_reason_share_seed_and_isolate_reason_increment():
    case = OracleCase(
        trajectory=trajectory(),
        metadata=metadata(),
        canonical_nodes=canonical_nodes(),
        canonical_reason_id="3",
        model_reason_id="3",
    )
    result = asyncio.run(
        audit_case(
            case,
            FakeReplayer(),
            samples=2,
            modes={"plan", "plan_reason"},
            seed=7,
        )
    )
    plan_seeds = [sample["seed"] for sample in result["modes"]["plan"]["samples"]]
    oracle_seeds = [
        sample["seed"] for sample in result["modes"]["plan_reason"]["samples"]
    ]
    assert plan_seeds == oracle_seeds
    report = summarize_canonical_increment([result])
    assert report["factual_control_mismatches"] == 0
    assert report["reason_increment"] == 1


def test_report_fails_identity_mismatch_even_with_oracle_gain():
    result = {
        "modes": {
            "reason": {
                "eligible": True,
                "intervention_changed": True,
                "samples": [
                    {
                        "factual_em": 0,
                        "oracle_em": 1,
                        "factual_identity_match": False,
                        "oracle_identity_match": True,
                        "identity_match": False,
                    }
                ],
            }
        }
    }
    report = summarize(
        [result],
        {
            "evaluation_rows": 1,
            "metadata_rows": 1,
            "joined_eligible_rows": 1,
            "selected_rows": 1,
            "skipped_pipeline_errors": 0,
            "model_empty_plan_rows": 0,
        },
        {"reason"},
        minimum_delta=0.03,
    )
    assert report["hard_gates"]["identity_controls_exact"] is False
    assert report["passed"] is False
