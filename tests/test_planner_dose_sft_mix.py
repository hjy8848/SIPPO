import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from qa_manager.ROSA import protocol


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dagent_planner_dose_sft_mix.py"
SPEC = importlib.util.spec_from_file_location("planner_dose_sft_mix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(trajectory_id, role, source, index):
    prompts = {
        "PlanningAgent": protocol.PLANNER_SYSTEM_MSG,
        "ReasonerAgent": protocol.REASON_SYSTEM_MSG,
    }
    return {
        "messages": [
            {"role": "system", "content": prompts[role], "loss_mask": None},
            {"role": "user", "content": "question", "loss_mask": None},
            {"role": "assistant", "content": "answer", "loss_mask": 1},
        ],
        "enable_thinking": False,
        "agent_name": role,
        "trajectory_id": trajectory_id,
        "source_index": index,
        "question": "question",
        "unified_source": source,
    }


def write_inputs(root):
    unified = root / "unified"
    matched = root / "matched"
    unified.mkdir()
    matched.mkdir()
    unified_train = [
        row("causal-a", "PlanningAgent", "new", 1),
        row("causal-a", "ReasonerAgent", "new", 2),
        row("causal-b", "PlanningAgent", "new", 3),
        row("causal-b", "ReasonerAgent", "new", 4),
        row("replay-a", "PlanningAgent", "replay", 5),
        row("replay-a", "ReasonerAgent", "replay", 6),
    ]
    unified_val = [row("val-a", "PlanningAgent", "new", 7)]
    pd.DataFrame(unified_train).to_parquet(unified / "sft_train.parquet", index=False)
    pd.DataFrame(unified_val).to_parquet(unified / "sft_val.parquet", index=False)

    matched_rows = [
        row("matched-a", "PlanningAgent", "replay", 8),
        row("matched-b", "PlanningAgent", "replay", 9),
    ]
    for value in matched_rows:
        value["matched_control_source"] = "matched_extra"
    pd.DataFrame(matched_rows).to_parquet(matched / "sft_train.parquet", index=False)
    return unified, matched


def test_build_varies_only_causal_planner_exposure(tmp_path):
    unified, matched = write_inputs(tmp_path)
    output = tmp_path / "output"
    report = MODULE.build(
        argparse.Namespace(
            unified_data_dir=unified,
            matched_data_dir=matched,
            output_dir=output,
            causal_planner_count=1,
            batch_size=2,
            seed=42,
        )
    )
    train = pd.read_parquet(output / "sft_train.parquet")
    assert len(train) == 6
    assert sum(train["planner_dose_source"] == "causal_planner") == 1
    assert sum(train["planner_dose_source"] == "planner_replay") == 1
    assert sum(
        (train["unified_source"] == "new")
        & (train["agent_name"] == "ReasonerAgent")
    ) == 2
    assert report["nonplanner_causal_rows"] == 2
    assert report["optimizer_steps"] == 3
    assert report["hard_gates"]["nonplanner_rows_exact"]
    assert report["hard_gates"]["validation_rows_exact"]
    assert report["hard_gates"]["nonplanner_sampler_slots_fixed"]
    assert report["hard_gates"]["planner_sampler_slots_fixed"]
    assert report["hard_gates"]["planner_pair_ranks_exact"]
    assert all(report["hard_gates"].values())


def test_doses_keep_every_sampler_slot_aligned(tmp_path):
    unified, matched = write_inputs(tmp_path)
    outputs = {}
    for dose in (0, 1, 2):
        output = tmp_path / f"dose-{dose}"
        MODULE.build(
            argparse.Namespace(
                unified_data_dir=unified,
                matched_data_dir=matched,
                output_dir=output,
                causal_planner_count=dose,
                batch_size=2,
                seed=42,
            )
        )
        outputs[dose] = pd.read_parquet(output / "sft_train.parquet")

    for index in range(len(outputs[0])):
        rows = [outputs[dose].iloc[index] for dose in (0, 1, 2)]
        assert len({str(row["agent_name"]) for row in rows}) == 1
        if str(rows[0]["agent_name"]) != "PlanningAgent":
            assert len({str(row["trajectory_id"]) for row in rows}) == 1
            assert len({str(row["messages"]) for row in rows}) == 1

    paired_slots = outputs[0][outputs[0]["planner_dose_pair_rank"] >= 0].index
    assert list(paired_slots) == list(
        outputs[1][outputs[1]["planner_dose_pair_rank"] >= 0].index
    )
    assert list(paired_slots) == list(
        outputs[2][outputs[2]["planner_dose_pair_rank"] >= 0].index
    )


def test_build_rejects_out_of_range_dose(tmp_path):
    unified, matched = write_inputs(tmp_path)
    try:
        MODULE.build(
            argparse.Namespace(
                unified_data_dir=unified,
                matched_data_dir=matched,
                output_dir=tmp_path / "output",
                causal_planner_count=3,
                batch_size=2,
                seed=42,
            )
        )
    except ValueError as error:
        assert "causal_planner_count must be in" in str(error)
        return
    raise AssertionError("out-of-range Planner dose was accepted")
