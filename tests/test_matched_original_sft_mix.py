import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from qa_manager.ROSA import protocol


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dagent_matched_original_sft_mix.py"
SPEC = importlib.util.spec_from_file_location("matched_original_sft_mix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


PROMPTS = {
    "PlanningAgent": protocol.PLANNER_SYSTEM_MSG,
    "ReasonerAgent": protocol.REASON_SYSTEM_MSG,
}


def row(trajectory_id, role, *, source=None, legacy=False):
    value = {
        "trajectory_id": trajectory_id,
        "agent_name": role,
        "enable_thinking": False,
        "messages": [
            {
                "role": "system",
                "content": "legacy" if legacy else PROMPTS[role],
                "loss_mask": None,
            },
            {"role": "user", "content": "question", "loss_mask": None},
            {"role": "assistant", "content": "answer", "loss_mask": 1},
        ],
    }
    if source is not None:
        value["unified_source"] = source
    return value


def test_build_replaces_causal_rows_with_role_matched_originals(tmp_path):
    original = tmp_path / "original"
    unified = tmp_path / "unified"
    original.mkdir()
    unified.mkdir()
    original_train = [
        row("old-p1", "PlanningAgent", legacy=True),
        row("old-p2", "PlanningAgent"),
        row("old-r1", "ReasonerAgent"),
        row("old-r2", "ReasonerAgent"),
    ]
    pd.DataFrame(original_train).to_parquet(original / "sft_train.parquet", index=False)
    pd.DataFrame([row("old-val", "PlanningAgent")]).to_parquet(
        original / "sft_val.parquet", index=False
    )
    pd.DataFrame(
        [
            row("new-p", "PlanningAgent", source="new"),
            row("new-r", "ReasonerAgent", source="new"),
            row("old-p1", "PlanningAgent", source="replay"),
        ]
    ).to_parquet(unified / "sft_train.parquet", index=False)

    output = tmp_path / "output"
    report = MODULE.build(
        argparse.Namespace(
            original_data_dir=original,
            unified_data_dir=unified,
            output_dir=output,
            original_full_repeats=2,
            batch_size=2,
            seed=42,
        )
    )
    train = pd.read_parquet(output / "sft_train.parquet")
    extra = train[train["matched_control_source"] == "matched_extra"]
    assert len(train) == 10
    assert len(extra) == 2
    assert extra["agent_name"].value_counts().to_dict() == {
        "PlanningAgent": 1,
        "ReasonerAgent": 1,
    }
    assert report["optimizer_steps"] == 5
    assert report["prompt_mismatches_replaced"]["train"] == 1
    assert all(report["hard_gates"].values())


def test_build_refuses_existing_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    try:
        MODULE.build(
            argparse.Namespace(
                original_data_dir=tmp_path / "original",
                unified_data_dir=tmp_path / "unified",
                output_dir=output,
                original_full_repeats=2,
                batch_size=2,
                seed=42,
            )
        )
    except FileExistsError:
        return
    raise AssertionError("existing output directory was overwritten")
