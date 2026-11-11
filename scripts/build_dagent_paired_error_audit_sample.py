#!/usr/bin/env python3
"""Freeze a reproducible, paired 100-question error-audit sample."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping

from analyze_dagent_3b_submission_additions import (
    DATASET_SIZES,
    align_arms,
    load_arm,
)


QUOTAS = {
    "2wikimultihopqa": {"both_wrong": 19, "pure_only": 6, "psc_only": 6},
    "hotpotqa": {"both_wrong": 19, "pure_only": 6, "psc_only": 6},
    "musique": {"both_wrong": 19, "pure_only": 6, "psc_only": 6},
    "bamboogle": {"both_wrong": 4, "pure_only": 1, "psc_only": 2},
}
TAXONOMY = [
    "correct",
    "plan_decomposition_omission",
    "wrong_relation_or_shortcut_query",
    "retrieval_miss",
    "query_rewrite_or_entity_transfer",
    "evidence_present_but_node_conclusion_wrong",
    "reasoning_comparison_or_composition",
    "final_synthesis",
    "answer_normalization_or_alias",
    "invalid_plan_or_execution_failure",
    "evidence_or_corpus_insufficiency",
    "gold_annotation_or_question_defect",
    "other",
]


def stratum(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    pure_correct = float(left["em"]) == 1.0
    psc_correct = float(right["em"]) == 1.0
    if pure_correct and psc_correct:
        return "both_correct"
    if pure_correct:
        return "pure_only"
    if psc_correct:
        return "psc_only"
    return "both_wrong"


def compact_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    trace = row.get("raw_trace") or {}
    return {
        "sample_index": row.get("sample_index"),
        "prediction": row.get("pred_answer"),
        "em": float(row["em"]),
        "f1": float(row["f1"]),
        "retrieval_calls": row.get("retrieval_calls"),
        "pipeline_error": row.get("pipeline_error"),
        "error": row.get("error"),
        "plan": trace.get("plan_output") or trace.get("raw_plan_output"),
        "plan_warnings": trace.get("plan_warnings") or [],
        "nodes_execution": trace.get("nodes_execution") or [],
        "final_answer": trace.get("final_answer"),
        "verification": trace.get("verification"),
        "sft_events": trace.get("sft_events") or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure-root", type=Path, required=True)
    parser.add_argument("--psc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    pure = load_arm(args.pure_root)
    psc = load_arm(args.psc_root)
    aligned = align_arms(pure, psc)
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    population: dict[str, dict[str, int]] = {}
    for dataset in DATASET_SIZES:
        cells: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {
            name: [] for name in ("both_wrong", "pure_only", "psc_only", "both_correct")
        }
        for left, right in aligned[dataset]:
            cells[stratum(left, right)].append((left, right))
        population[dataset] = {name: len(rows) for name, rows in cells.items()}
        for cell, quota in QUOTAS[dataset].items():
            if len(cells[cell]) < quota:
                raise RuntimeError(f"Insufficient {dataset}/{cell}: {len(cells[cell])}/{quota}")
            chosen = rng.sample(cells[cell], quota)
            weight = len(cells[cell]) / quota
            for left, right in chosen:
                selected.append(
                    {
                        "dataset": dataset,
                        "stratum": cell,
                        "selection_weight": weight,
                        "question": left["question"],
                        "gold_answer": left.get("gold_answer"),
                        "pure": compact_trace(left),
                        "psc": compact_trace(right),
                    }
                )
    rng.shuffle(selected)
    if len(selected) != 100:
        raise RuntimeError(f"Expected 100 selected cases, got {len(selected)}")
    for index, case in enumerate(selected, start=1):
        case["case_id"] = f"Q{index:03d}"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    contract = {
        "seed": args.seed,
        "alignment": "unique exact question + JSON-canonical gold answer",
        "eligible_pool": "at least one arm has EM=0",
        "quotas": QUOTAS,
        "population_cells": population,
        "weighting": "population cell size / sampled cell size",
        "taxonomy": TAXONOMY,
    }
    (args.output_dir / "sampling_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "audit_cases.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_fields = [
        "case_id", "dataset", "stratum", "selection_weight", "question", "gold_answer",
        "pure_prediction", "pure_em", "pure_f1", "psc_prediction", "psc_em", "psc_f1",
    ]
    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for case in selected:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "stratum": case["stratum"],
                    "selection_weight": case["selection_weight"],
                    "question": case["question"],
                    "gold_answer": json.dumps(case["gold_answer"], ensure_ascii=False),
                    "pure_prediction": case["pure"]["prediction"],
                    "pure_em": case["pure"]["em"],
                    "pure_f1": case["pure"]["f1"],
                    "psc_prediction": case["psc"]["prediction"],
                    "psc_em": case["psc"]["em"],
                    "psc_f1": case["psc"]["f1"],
                }
            )
    label_fields = [
        "case_id", "pure_primary", "pure_secondary", "psc_primary", "psc_secondary",
        "pair_difference", "auditor_notes",
    ]
    with (args.output_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields)
        writer.writeheader()
        for case in selected:
            writer.writerow({"case_id": case["case_id"]})


if __name__ == "__main__":
    main()
