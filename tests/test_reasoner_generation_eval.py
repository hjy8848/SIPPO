from __future__ import annotations

import importlib.util
from pathlib import Path

import json


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_dagent_reasoner_generation.py"
SPEC = importlib.util.spec_from_file_location("reasoner_generation_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row():
    return {
        "pair_id": "p",
        "group_id": "g",
        "split": "test",
        "variant": "factual_condition",
        "operation": "earlier",
        "chosen": "<conclusion>Ada Lovelace</conclusion>",
        "rejected": "<conclusion>Grace Hopper</conclusion>",
    }


def test_generation_classification_separates_format_and_semantics():
    bare = MODULE.classify_output(row(), "Ada Lovelace")
    assert bare["format_valid"] is False
    assert bare["semantic_label"] == "chosen"
    tagged = MODULE.classify_output(row(), "<conclusion>Grace Hopper</conclusion>")
    assert tagged["format_valid"] is True
    assert tagged["semantic_label"] == "rejected"
    other = MODULE.classify_output(row(), "Alan Turing")
    assert other["semantic_label"] == "other"


def test_actual_causal_sft_exposure_overrides_nominal_split(tmp_path):
    seen = MODULE.classify_output(
        row(), "<conclusion>Ada Lovelace</conclusion>", {"g"}
    )
    held_out = MODULE.classify_output(
        row(), "<conclusion>Ada Lovelace</conclusion>", {"another-group"}
    )
    assert seen["split"] == "test"
    assert seen["causal_sft_exposure"] == "seen"
    assert held_out["causal_sft_exposure"] == "held_out"

    report = tmp_path / "mix.json"
    report.write_text(
        json.dumps({"selected_new_trajectory_ids": ["g", "another-group"]}),
        encoding="utf-8",
    )
    assert MODULE.read_seen_group_ids(report) == {"g", "another-group"}
