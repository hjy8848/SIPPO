import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from qa_manager.ROSA import protocol


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dagent_sft_rehearsal_mix.py"
SPEC = importlib.util.spec_from_file_location("sft_rehearsal_mix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(trajectory_id, agent_name, system_prompt, *, include_masked_history=False):
    messages = [
            {"role": "system", "content": system_prompt, "loss_mask": None},
            {"role": "user", "content": "question", "loss_mask": None},
    ]
    if include_masked_history:
        messages.extend(
            [
                {"role": "assistant", "content": "context answer", "loss_mask": 0},
                {"role": "user", "content": "follow-up", "loss_mask": None},
            ]
        )
    messages.append({"role": "assistant", "content": "answer", "loss_mask": 1})
    return {
        "messages": messages,
        "enable_thinking": False,
        "agent_name": agent_name,
        "trajectory_id": trajectory_id,
    }


def write_split(root, split, rows):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / f"sft_{split}.parquet", index=False)


def test_build_canonicalizes_replay_and_reports_matched_exposure(tmp_path):
    new_dir = tmp_path / "new"
    replay_dir = tmp_path / "replay"
    write_split(
        new_dir,
        "train",
        [row("new-train", "PlanningAgent", protocol.PLANNER_SYSTEM_MSG)] * 3,
    )
    write_split(
        replay_dir,
        "train",
        [
            row(
                "replay-train",
                "PlanningAgent",
                "legacy planner",
                include_masked_history=True,
            )
        ],
    )
    write_split(
        new_dir,
        "val",
        [row("new-val", "ReasonerAgent", protocol.REASON_SYSTEM_MSG)],
    )
    write_split(
        replay_dir,
        "val",
        [row("replay-val", "ReasonerAgent", "legacy reason")],
    )
    output_dir = tmp_path / "mixed"
    report = MODULE.build(
        argparse.Namespace(
            new_data_dir=new_dir,
            replay_data_dir=replay_dir,
            output_dir=output_dir,
            seed=42,
            reference_new_only_steps=50,
        )
    )
    train = pd.read_parquet(output_dir / "sft_train.parquet")
    val = pd.read_parquet(output_dir / "sft_val.parquet")
    assert len(train) == 4
    assert len(val) == 2
    assert report["train"]["new_fraction"] == 0.75
    assert report["matched_new_exposure_steps"] == 67
    assert report["prompt_mismatches_replaced"] == {
        "train": {"new": 0, "replay": 1},
        "val": {"new": 0, "replay": 1},
    }
    assert all(report["hard_gates"].values())
    assert all(
        list(messages)[0]["content"] == protocol.PLANNER_SYSTEM_MSG
        for messages in train["messages"]
    )
    assert list(val.iloc[1]["messages"])[0]["content"] == protocol.REASON_SYSTEM_MSG


def test_build_refuses_to_overwrite_output(tmp_path):
    output_dir = tmp_path / "mixed"
    output_dir.mkdir()
    try:
        MODULE.build(
            argparse.Namespace(
                new_data_dir=tmp_path / "new",
                replay_data_dir=tmp_path / "replay",
                output_dir=output_dir,
                seed=42,
                reference_new_only_steps=50,
            )
        )
    except FileExistsError:
        return
    raise AssertionError("existing output directory was overwritten")
