import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_dagent_hindsight_opsd_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hindsight_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def planner_event(plan):
    return {
        "agent_name": "PlanningAgent",
        "messages": [
            {"role": "system", "content": "Plan carefully."},
            {"role": "user", "content": "Question: q"},
        ],
        "response": plan,
    }


def test_builder_uses_only_plans_and_outcome_labels(tmp_path):
    module = load_module()
    rows = [
        {
            "sample_index": "q1",
            "rollout_index": 0,
            "question": "q",
            "gold_answer": ["SECRET_GOLD"],
            "pred_answer": "SECRET_FINAL",
            "em": 1.0,
            "raw_trace": {
                "sft_events": [planner_event('<plan><search id="1">good query</search></plan>')],
                "nodes_execution": [{"search_context": "SECRET_DOCUMENT"}],
            },
        },
        {
            "sample_index": "q1",
            "rollout_index": 1,
            "question": "q",
            "gold_answer": ["SECRET_GOLD"],
            "pred_answer": "wrong",
            "em": 0.0,
            "raw_trace": {
                "sft_events": [planner_event('<plan><search id="1">bad query</search></plan>')],
                "nodes_execution": [{"search_context": "SECRET_DOCUMENT"}],
            },
        },
    ]
    log = tmp_path / "rollouts.json"
    log.write_text(json.dumps(rows), encoding="utf-8")
    rejected = module.Counter()
    rollouts = module.read_rollouts([str(log)], "standard", "candidate", rejected)

    class Args:
        seed = 42
        max_incorrect_per_group = 2
        max_hindsight_rollouts = 6
        max_hindsight_chars = 6000

    examples = module.build_examples(rollouts, Args(), rejected)
    assert len(examples) == 1
    payload = json.dumps(examples[0]["teacher_messages"])
    assert "good query" in payload
    assert "Correct" in payload
    assert "SECRET_GOLD" not in payload
    assert "SECRET_FINAL" not in payload
    assert "SECRET_DOCUMENT" not in payload
    assert "bad query" not in payload
    control = json.dumps(examples[0]["control_teacher_messages"])
    assert "Incorrect" in control


def test_reason_examples_require_successful_reason_then_search_reference(tmp_path):
    module = load_module()
    good = (
        '<plan><search id="1">a</search><search id="2">b</search>'
        '<reason id="3">compare #1# and #2#</reason>'
        '<search id="4">find detail about #3#</search></plan>'
    )
    bad = '<plan><search id="1">direct</search></plan>'
    rows = [
        {"sample_index": "q1", "rollout_index": 0, "question": "q", "em": 1, "raw_trace": {"sft_events": [planner_event(good)]}},
        {"sample_index": "q1", "rollout_index": 1, "question": "q", "em": 0, "raw_trace": {"sft_events": [planner_event(bad)]}},
    ]
    log = tmp_path / "reason.json"
    log.write_text(json.dumps(rows), encoding="utf-8")
    rejected = module.Counter()
    rollouts = module.read_rollouts([str(log)], "reason", "candidate", rejected)

    class Args:
        seed = 42
        max_incorrect_per_group = 2
        max_hindsight_rollouts = 6
        max_hindsight_chars = 6000

    examples = module.build_examples(rollouts, Args(), rejected)
    assert len(examples) == 1
    assert examples[0]["objective"] == "planner"
    assert examples[0]["successful_reference_target_topology"] is True
    assert examples[0]["focal_target_topology"] is False


def test_verified_trajectory_can_supply_reason_reference(tmp_path):
    module = load_module()
    good = (
        '<plan><search id="1">a</search><search id="2">b</search>'
        '<reason id="3">compare #1# and #2#</reason>'
        '<search id="4">find detail about #3#</search></plan>'
    )
    accepted = {
        "trajectory_id": "q1",
        "question": "q",
        "gold_answers": ["SECRET_GOLD"],
        "nodes_execution": [{"search_context": "SECRET_DOCUMENT"}],
        "sft_events": [planner_event(good)],
    }
    trajectory_file = tmp_path / "accepted.jsonl"
    trajectory_file.write_text(json.dumps(accepted) + "\n", encoding="utf-8")
    bad_rows = [
        {
            "sample_index": "q1",
            "rollout_index": 0,
            "question": "q",
            "em": 0,
            "raw_trace": {"sft_events": [planner_event('<plan><search id="1">direct</search></plan>')]},
        }
    ]
    rollout_file = tmp_path / "reason.json"
    rollout_file.write_text(json.dumps(bad_rows), encoding="utf-8")
    rejected = module.Counter()
    candidates = module.read_rollouts([str(rollout_file)], "reason", "candidate", rejected)
    references = module.read_verified_reason_trajectories([str(trajectory_file)], rejected)

    class Args:
        seed = 42
        max_incorrect_per_group = 2
        max_hindsight_rollouts = 6
        max_hindsight_chars = 6000

    examples = module.build_examples(candidates + references, Args(), rejected)
    assert len(examples) == 1
    assert examples[0]["positive_reference_response"] == good
    teacher = json.dumps(examples[0]["teacher_messages"])
    assert "SECRET_GOLD" not in teacher
    assert "SECRET_DOCUMENT" not in teacher


def test_verified_parquet_row_exposes_only_plan_reference():
    module = load_module()
    plan = (
        '<plan><search id="1">a</search><search id="2">b</search>'
        '<reason id="3">compare #1# and #2#</reason>'
        '<search id="4">find detail about #3#</search></plan>'
    )
    rejected = module.Counter()
    reference = module.verified_reference_from_parquet_row(
        {
            "extra_info": {
                "id": "q1",
                "question": "q",
                "gold_plan": plan,
                "answer": ["SECRET_GOLD"],
                "supporting_titles": ["SECRET_TITLE"],
            }
        },
        0,
        "eval.parquet",
        rejected,
    )
    assert reference is not None
    assert reference.plan == plan
    assert reference.messages == []
    assert "SECRET_GOLD" not in reference.plan
    assert "SECRET_TITLE" not in reference.plan
