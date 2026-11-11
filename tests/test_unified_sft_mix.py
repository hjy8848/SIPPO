import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from qa_manager.ROSA import protocol


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dagent_unified_sft_mix.py"
SPEC = importlib.util.spec_from_file_location("unified_sft_mix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


ROLE_PROMPTS = {
    "PlanningAgent": protocol.PLANNER_SYSTEM_MSG,
    "AnswerGenerationAgent": protocol.ANSWER_GENERATION_SYSTEM_MSG,
    "ReasonerAgent": protocol.REASON_SYSTEM_MSG,
    "QueryRewriteAgent": protocol.QUERY_REWRITE_SYSTEM_MSG,
    "AnswerSummarizationAgent": protocol.FINAL_SYSTEM_MSG,
}


def row(trajectory_id, agent_name, *, legacy=False):
    messages = [
        {
            "role": "system",
            "content": "legacy" if legacy else ROLE_PROMPTS[agent_name],
            "loss_mask": None,
        },
        {"role": "user", "content": "question", "loss_mask": None},
        {"role": "assistant", "content": "answer", "loss_mask": 1},
    ]
    return {
        "messages": messages,
        "enable_thinking": False,
        "agent_name": agent_name,
        "trajectory_id": trajectory_id,
    }


def causal_trajectory(trajectory_id):
    roles = [
        "PlanningAgent",
        "AnswerGenerationAgent",
        "AnswerGenerationAgent",
        "ReasonerAgent",
        "QueryRewriteAgent",
        "AnswerGenerationAgent",
        "AnswerSummarizationAgent",
    ]
    return [row(trajectory_id, role) for role in roles]


def write_split(root, split, rows):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / f"sft_{split}.parquet", index=False)


def test_build_matches_exposure_and_keeps_complete_causal_trajectories(tmp_path):
    new_dir = tmp_path / "new"
    replay_dir = tmp_path / "replay"
    write_split(new_dir, "train", causal_trajectory("new-a") + causal_trajectory("new-b"))
    write_split(new_dir, "val", causal_trajectory("new-val"))
    write_split(
        replay_dir,
        "train",
        [row("replay-a", "PlanningAgent", legacy=True), row("replay-b", "ReasonerAgent")],
    )
    write_split(replay_dir, "val", [row("replay-val", "PlanningAgent", legacy=True)])
    output_dir = tmp_path / "unified"
    report = MODULE.build(
        argparse.Namespace(
            new_data_dir=new_dir,
            replay_data_dir=replay_dir,
            output_dir=output_dir,
            new_train_trajectories=2,
            replay_train_repeats=4,
            batch_size=2,
            seed=42,
        )
    )
    train = pd.read_parquet(output_dir / "sft_train.parquet")
    assert len(train) == 22
    assert sum(train["unified_source"] == "new") == 14
    assert sum(train["unified_source"] == "replay") == 8
    assert report["optimizer_steps"] == 11
    assert report["prompt_mismatches_replaced"]["train"]["replay"] == 1
    assert all(report["hard_gates"].values())
    assert set(train.loc[train["unified_source"] == "new", "trajectory_id"]) == {
        "new-a",
        "new-b",
    }


def test_build_rejects_incomplete_causal_trajectory(tmp_path):
    new_dir = tmp_path / "new"
    replay_dir = tmp_path / "replay"
    write_split(new_dir, "train", causal_trajectory("new-a")[:-1])
    write_split(new_dir, "val", causal_trajectory("new-val"))
    write_split(replay_dir, "train", [row("replay-a", "PlanningAgent")])
    write_split(replay_dir, "val", [row("replay-val", "PlanningAgent")])
    try:
        MODULE.build(
            argparse.Namespace(
                new_data_dir=new_dir,
                replay_data_dir=replay_dir,
                output_dir=tmp_path / "unified",
                new_train_trajectories=1,
                replay_train_repeats=2,
                batch_size=1,
                seed=42,
            )
        )
    except ValueError as error:
        assert "causal trajectories are incomplete" in str(error)
        return
    raise AssertionError("incomplete causal trajectory was accepted")
