from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_dagent_cwpo_one_batch.py"
SPEC = importlib.util.spec_from_file_location("cwpo_one_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def pair(group, variant, operation="earlier", split="train"):
    return {
        "pair_id": f"{group}-{variant}",
        "group_id": group,
        "agent_name": "ReasonerAgent",
        "variant": variant,
        "operation": operation,
        "split": split,
    }


def test_index_and_select_keep_complete_balanced_twins():
    rows = []
    for operation in MODULE.OPERATIONS:
        for index in range(3):
            group = f"{operation}-{index}"
            rows.extend(pair(group, variant, operation) for variant in MODULE.VARIANTS)
    twins = MODULE.index_twins(rows)
    selected = MODULE.select_groups(twins, split="train", groups_per_operation=2, seed=7)
    selected_rows = MODULE.rows_for_groups(twins, selected)
    assert len(selected) == 6
    assert len(selected_rows) == 12
    assert {operation: sum(x["operation"] == operation for x in selected_rows) for operation in MODULE.OPERATIONS} == {
        operation: 4 for operation in MODULE.OPERATIONS
    }
    assert sum(1 for x in selected_rows if x["variant"] == "factual_condition") == 6
    assert sum(1 for x in selected_rows if x["variant"] == "flipped_condition") == 6


def test_select_supports_operation_specific_counts():
    rows = []
    for operation in MODULE.OPERATIONS:
        for index in range(3):
            group = f"{operation}-{index}"
            rows.extend(pair(group, variant, operation) for variant in MODULE.VARIANTS)
    twins = MODULE.index_twins(rows)
    selected = MODULE.select_groups(
        twins,
        split="train",
        groups_per_operation={"closer_to_anchor": 3, "earlier": 1, "later": 0},
        seed=7,
    )
    selected_rows = MODULE.rows_for_groups(twins, selected)
    assert len(selected) == 4
    assert {
        operation: sum(row["operation"] == operation for row in selected_rows)
        for operation in MODULE.OPERATIONS
    } == {"closer_to_anchor": 6, "earlier": 2, "later": 0}


def test_twin_summary_requires_both_conditions():
    rows = [
        {**pair("a", "factual_condition"), "sum_margin": 1.0},
        {**pair("a", "flipped_condition"), "sum_margin": 2.0},
        {**pair("b", "factual_condition"), "sum_margin": 1.0},
        {**pair("b", "flipped_condition"), "sum_margin": -2.0},
    ]
    summary = MODULE.twin_summary(rows)
    assert summary == {"groups": 2, "both_correct": 1, "both_correct_rate": 0.5}


def test_paired_summary_tracks_directional_change():
    before = [
        {**pair("a", "factual_condition"), "sum_margin": -1.0},
        {**pair("a", "flipped_condition"), "sum_margin": 1.0},
    ]
    after = [
        {**pair("a", "factual_condition"), "sum_margin": 0.5},
        {**pair("a", "flipped_condition"), "sum_margin": 2.0},
    ]
    summary = MODULE.paired_summary(before, after)
    assert summary["margin_delta_mean"] == 1.25
    assert summary["fraction_margin_improved"] == 1.0
    assert summary["before"]["accuracy"] == 0.5
    assert summary["after"]["accuracy"] == 1.0


def test_length_audit_detects_no_length_correlated_change():
    before = [
        {**pair("a", "factual_condition"), "sum_margin": 0.0, "chosen_token_count": 2, "rejected_token_count": 4},
        {**pair("a", "flipped_condition"), "sum_margin": 0.0, "chosen_token_count": 4, "rejected_token_count": 2},
    ]
    after = [
        {**before[0], "sum_margin": 1.0},
        {**before[1], "sum_margin": 1.0},
    ]
    audit = MODULE.length_audit(before, after)
    assert audit["chosen_minus_rejected_token_total"] == 0
    assert audit["margin_delta_vs_token_delta_pearson"] == 0.0


def test_parameter_probe_indices_do_not_round_past_large_tensor_end():
    numel = 151_936 * 2_048
    indices = MODULE.sample_indices(numel, 256, torch.device("cpu"))
    assert indices[0].item() == 0
    assert indices[-1].item() == numel - 1
    assert indices.min().item() >= 0
    assert indices.max().item() < numel
