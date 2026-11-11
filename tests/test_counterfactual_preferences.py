import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dagent_counterfactual_preferences.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_preferences", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def event(agent_name, messages, response, node_id=None):
    metadata = {} if node_id is None else {"node_id": str(node_id)}
    return {"agent_name": agent_name, "messages": messages, "response": response, "metadata": metadata}


def accepted_record(index, *, anchor=False, shared_parent=None):
    candidate_a = f"Alpha {index}"
    candidate_b = f"Beta {index}"
    winner = candidate_a
    loser = candidate_b
    answer = f"Job A {index}"
    counterfactual_answer = f"Job B {index}"
    question = f"What was the occupation of the older candidate, {candidate_a} or {candidate_b}?"
    reasoning = {
        "mode": "anchor_distance" if anchor else "direct",
        "operation": "closer_to_anchor" if anchor else "earlier",
        "anchor_year": 1920 if anchor else None,
        "requires_additional_retrieval": False,
    }
    reason_task = (
        f"Candidate {candidate_a} has value 1900 and candidate {candidate_b} has value 1980. "
        "Convert both dates of birth to years, compute each absolute distance from 1920, "
        "and select the candidate with the smaller distance. Return the selected candidate."
        if anchor
        else f"Candidate {candidate_a} has value 1900 and candidate {candidate_b} has value 1980. "
        "Compare the two dates of birth and select the candidate with the earlier value. "
        "Return the selected candidate."
    )
    plan = (
        "<plan>\n"
        f"<search id=\"1\">{candidate_a} date of birth</search>\n"
        f"<search id=\"2\">{candidate_b} date of birth</search>\n"
        f"<reason id=\"3\">{reason_task}</reason>\n"
        "<search id=\"4\">#3# occupation</search>\n"
        "</plan>"
    )
    planning_messages = [
        {"role": "system", "content": "Generate a DAG execution plan."},
        {"role": "user", "content": f"Question: {question}\nGenerate the execution plan."},
    ]
    reason_messages = [
        {"role": "system", "content": "Resolve the logic task using only supplied facts."},
        {"role": "user", "content": f"**Current Logic Task**: {reason_task}"},
    ]
    query_messages = [
        {"role": "system", "content": "Rewrite a dependent query."},
        {
            "role": "user",
            "content": (
                f"Original Goal: {question}\nDraft: #3# occupation\n"
                f"# 3 = {winner}\nRewritten query:"
            ),
        },
    ]
    parent_a = shared_parent or f"parent-a-{index}"
    parent_b = f"parent-b-{index}"
    return {
        "trajectory_id": f"trajectory-{index}",
        "topology": "search-search-reason-search",
        "question": question,
        "answer": answer,
        "gold_answers": [answer],
        "winner": winner,
        "counterfactual_entity": loser,
        "counterfactual_answer": counterfactual_answer,
        "source": {"parent_trajectory_ids": [parent_a, parent_b]},
        "blueprint": {"candidate_a": candidate_a, "candidate_b": candidate_b},
        "downstream": {"relation": "occupation"},
        "comparison_evidence": {
            "candidate_a": {"answer": "1900", "comparison_key": 19000101.0},
            "candidate_b": {"answer": "1980", "comparison_key": 19800101.0},
        },
        "plan_output": plan,
        "nodes_execution": [
            {"id": "1", "type": "search", "executed_query": f"{candidate_a} date of birth"},
            {"id": "2", "type": "search", "executed_query": f"{candidate_b} date of birth"},
            {"id": "3", "type": "reason", "conclusion": winner},
            {"id": "4", "type": "search", "executed_query": f"{winner} occupation"},
        ],
        "verification": {
            "evidence_first_graph_join": True,
            "reason_is_required_by_search_4": True,
            "counterfactual_retrievable": True,
            "counterfactual_answer_differs": True,
            "direct_question_does_not_retrieve_answer": True,
            "final_answer_grounded_in_search_4": True,
        },
        "counterfactual": {
            "executed_query": f"{loser} occupation",
            "answer": counterfactual_answer,
            "documents": [{"title": loser, "contents": counterfactual_answer}],
        },
        "reasoning": reasoning,
        "evidence_join": {"left_card_id": f"card-a-{index}", "right_card_id": f"card-b-{index}"},
        "sft_events": [
            event("PlanningAgent", planning_messages, plan),
            event("ReasonerAgent", reason_messages, f"<conclusion>{winner}</conclusion>", 3),
            event("QueryRewriteAgent", query_messages, f"{winner} occupation", 4),
        ],
    }


def failed_rollout(record, *, em=0.0):
    rejected_plan = (
        "<plan>\n"
        f"<search id=\"1\">Who is older, {record['blueprint']['candidate_a']} or "
        f"{record['blueprint']['candidate_b']}?</search>\n"
        "<search id=\"2\">What is #1# occupation?</search>\n"
        "</plan>"
    )
    messages = [
        {"role": "system", "content": "Generate a DAG execution plan under the current policy."},
        {"role": "user", "content": f"Question: {record['question']}\nGenerate the execution plan."},
    ]
    return {
        "sample_index": record["trajectory_id"],
        "question": record["question"],
        "gold_answer": [record["answer"]],
        "pred_answer": record["counterfactual_answer"],
        "em": em,
        "search_steps": 2,
        "pipeline_error": None,
        "error": None,
        "raw_trace": {
            "plan_warnings": [],
            "nodes_execution": [{"type": "search"}, {"type": "search"}],
            "sft_events": [event("PlanningAgent", messages, rejected_plan)],
        },
    }


def test_direct_reason_pair_and_condition_flip_reverse_the_label():
    record = accepted_record(1)
    factual = MODULE.reason_pair(record)
    flipped = MODULE.flipped_reason_pair(record)
    assert factual["chosen"] == "<conclusion>Alpha 1</conclusion>"
    assert factual["rejected"] == "<conclusion>Beta 1</conclusion>"
    assert flipped["chosen"] == factual["rejected"]
    assert flipped["rejected"] == factual["chosen"]
    assert "later value" in flipped["prompt"][1]["content"]
    assert flipped["verification"]["label_flipped_from_factual_pair"] is True


def test_anchor_reason_flip_moves_anchor_to_counterfactual_winner():
    record = accepted_record(2, anchor=True)
    flipped = MODULE.flipped_reason_pair(record)
    assert "distance from 1980" in flipped["prompt"][1]["content"]
    assert flipped["chosen"] == "<conclusion>Beta 2</conclusion>"
    assert flipped["chosen_outcome"]["counterfactual_task_em"] == 1


def test_query_pair_uses_preverified_counterfactual_query():
    record = accepted_record(3)
    pair = MODULE.query_rewrite_pair(record)
    assert pair["chosen"] == "Alpha 3 occupation"
    assert pair["rejected"] == "Beta 3 occupation"
    assert pair["chosen_outcome"]["final_em"] == 1
    assert pair["rejected_outcome"]["final_em"] == 0


def test_planning_pair_uses_logged_failed_policy_action_under_one_prompt(tmp_path):
    record = accepted_record(4)
    path = tmp_path / "rollouts.json"
    path.write_text(json.dumps([failed_rollout(record)]), encoding="utf-8")
    indexed, stats = MODULE.index_failed_planner_rollouts(path)
    assert stats["eligible_failed_rollout"] == 1
    pair = MODULE.planning_pair(record, indexed[record["trajectory_id"]])
    assert pair["chosen"] == record["plan_output"]
    assert pair["rejected_outcome"]["topology"] == "search-search"
    assert pair["verification"]["rejected_execution_verified"] is True


def test_successful_policy_rollout_is_not_used_as_a_negative(tmp_path):
    record = accepted_record(5)
    path = tmp_path / "rollouts.json"
    path.write_text(json.dumps([failed_rollout(record, em=1.0)]), encoding="utf-8")
    indexed, stats = MODULE.index_failed_planner_rollouts(path)
    assert indexed == {}
    assert stats["successful_rollout"] == 1


def test_end_to_end_build_has_zero_parent_entity_and_card_overlap(tmp_path):
    records = [accepted_record(index, anchor=index % 2 == 0) for index in range(1, 7)]
    leaking_record = accepted_record(8)
    leaking_record["answer"] = "occupation"
    records.append(leaking_record)
    accepted_path = tmp_path / "accepted.jsonl"
    MODULE.write_jsonl(accepted_path, records)
    rollout_path = tmp_path / "rollouts.json"
    rollout_path.write_text(json.dumps([failed_rollout(record) for record in records]), encoding="utf-8")
    output_dir = tmp_path / "preferences"
    args = argparse.Namespace(
        accepted_jsonl=accepted_path,
        policy_rollout_json=rollout_path,
        output_dir=output_dir,
        split_ratios=(0.5, 0.25, 0.25),
        write_parquet=True,
    )
    report = MODULE.build(args)
    assert report["preference_pairs"] == 24
    assert report["source_records"] == 7
    assert report["retained_source_records"] == 6
    assert report["excluded_source_records_with_prompt_answer_leak"] == 1
    assert report["condition_flip_pairs"] == 6
    assert report["planner_join"] == {
        "constructed": 7,
        "excluded_with_prompt_answer_leak": 1,
        "retained": 6,
    }
    assert report["pair_audit"] == {
        "invalid_action_count": 0,
        "invalid_action_examples": [],
        "invalid_outcome_direction_count": 0,
        "invalid_outcome_direction_examples": [],
        "prompt_answer_leak_count": 0,
        "prompt_answer_leak_examples": [],
    }
    for details in report["split_overlap_audit"].values():
        assert set(details["overlap_counts"].values()) == {0}
    all_rows = MODULE.read_jsonl(output_dir / "preferences_all.jsonl")
    assert {row["agent_name"] for row in all_rows} == {
        "PlanningAgent", "ReasonerAgent", "QueryRewriteAgent"
    }
    assert all(row["verification"]["same_prompt"] for row in all_rows)
    import pandas as pd

    parquet_rows = sum(len(pd.read_parquet(output_dir / f"preferences_{split}.parquet")) for split in MODULE.SPLITS)
    assert parquet_rows == len(all_rows)


def test_pair_audit_detects_answer_leak_and_invalid_action():
    record = accepted_record(7)
    pair = MODULE.reason_pair(record)
    pair["prompt"][0]["content"] += " The final answer is Job A 7."
    pair["rejected"] = "Beta 7"
    audit = MODULE.audit_pairs([pair], {record["trajectory_id"]: record})
    assert audit["prompt_answer_leak_count"] == 1
    assert audit["invalid_action_count"] == 1


def test_canonical_prompt_gate_rejects_legacy_system_prompts(tmp_path):
    record = accepted_record(9)
    accepted_path = tmp_path / "accepted.jsonl"
    MODULE.write_jsonl(accepted_path, [record])
    args = argparse.Namespace(
        accepted_jsonl=accepted_path,
        policy_rollout_json=None,
        output_dir=tmp_path / "preferences",
        split_ratios=(0.5, 0.25, 0.25),
        write_parquet=False,
        require_canonical_prompts=True,
    )
    try:
        MODULE.build(args)
    except ValueError as exc:
        assert "canonical prompt audit failed" in str(exc)
        return
    raise AssertionError("legacy prompts passed the canonical prompt gate")
