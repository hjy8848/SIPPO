import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from qa_manager.ROSA import protocol


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_dagent_canonical_sft_data.py"
SPEC = importlib.util.spec_from_file_location("canonical_sft_rebuild", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(trajectory_id, agent_name, prompt):
    return {
        "trajectory_id": trajectory_id,
        "agent_name": agent_name,
        "enable_thinking": False,
        "messages": [
            {"role": "system", "content": prompt, "loss_mask": None},
            {"role": "user", "content": "question", "loss_mask": None},
            {"role": "assistant", "content": "answer", "loss_mask": 1},
        ],
    }


def test_rebuild_preserves_rows_and_canonicalizes_prompts(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame(
        [
            row("train-a", "PlanningAgent", "legacy planner"),
            row("train-b", "ReasonerAgent", protocol.REASON_SYSTEM_MSG),
        ]
    ).to_parquet(input_dir / "sft_train.parquet", index=False)
    pd.DataFrame(
        [row("val-a", "PlanningAgent", "legacy planner")]
    ).to_parquet(input_dir / "sft_val.parquet", index=False)

    output_dir = tmp_path / "output"
    report = MODULE.build(
        argparse.Namespace(input_dir=input_dir, output_dir=output_dir)
    )
    train = pd.read_parquet(output_dir / "sft_train.parquet")
    val = pd.read_parquet(output_dir / "sft_val.parquet")
    assert len(train) == 2
    assert len(val) == 1
    assert report["prompt_mismatches_replaced"] == {"train": 1, "val": 1}
    assert all(report["hard_gates"].values())
    assert list(train.iloc[0].messages)[0]["content"] == protocol.PLANNER_SYSTEM_MSG
    assert list(val.iloc[0].messages)[0]["content"] == protocol.PLANNER_SYSTEM_MSG


def test_rebuild_refuses_to_overwrite(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    try:
        MODULE.build(
            argparse.Namespace(input_dir=tmp_path / "input", output_dir=output_dir)
        )
    except FileExistsError:
        return
    raise AssertionError("existing output directory was overwritten")
