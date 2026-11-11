#!/usr/bin/env python3
"""Build unified SFT controls that vary only causal Planner exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scripts.build_dagent_sft_rehearsal_mix import (
        assistant_loss_mask_is_valid,
        canonicalize_frame,
        prompt_mismatch_count,
    )
except ImportError:  # Direct execution sets sys.path to scripts/.
    from build_dagent_sft_rehearsal_mix import (
        assistant_loss_mask_is_valid,
        canonicalize_frame,
        prompt_mismatch_count,
    )


PLANNER_ROLE = "PlanningAgent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-data-dir", type=Path, required=True)
    parser.add_argument("--matched-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--causal-planner-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    def key(row: pd.Series) -> str:
        value = "\x1f".join(
            (
                str(seed),
                str(row.get("trajectory_id", "")),
                str(row.get("source_index", "")),
                str(row.get("question", "")),
            )
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    ordered = frame.copy()
    ordered["_planner_dose_order"] = ordered.apply(key, axis=1)
    return ordered.sort_values("_planner_dose_order").drop(
        columns="_planner_dose_order"
    )


def role_counts(frame: pd.DataFrame) -> Counter[str]:
    return Counter(map(str, frame["agent_name"]))


def frame_row_multiset(
    frame: pd.DataFrame,
    *,
    columns: set[str] | None = None,
    ignored_columns: set[str] | None = None,
) -> Counter[tuple[str, ...]]:
    ignored = ignored_columns or set()
    selected_columns = sorted(
        column
        for column in (columns if columns is not None else set(frame.columns))
        if column not in ignored
    )
    if missing := set(selected_columns) - set(frame.columns):
        raise ValueError(f"frame is missing comparison columns: {sorted(missing)}")

    def normalize(value: Any) -> str:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    return Counter(
        tuple(normalize(row[column]) for column in selected_columns)
        for _, row in frame.iterrows()
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    paths = {
        "unified_train": args.unified_data_dir / "sft_train.parquet",
        "unified_val": args.unified_data_dir / "sft_val.parquet",
        "matched_train": args.matched_data_dir / "sft_train.parquet",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    unified_train, unified_train_mismatches = canonicalize_frame(
        pd.read_parquet(paths["unified_train"]), "unified"
    )
    unified_val, unified_val_mismatches = canonicalize_frame(
        pd.read_parquet(paths["unified_val"]), "unified_val"
    )
    matched_train, matched_train_mismatches = canonicalize_frame(
        pd.read_parquet(paths["matched_train"]), "matched"
    )
    required_columns = {
        "unified_source",
        "agent_name",
        "trajectory_id",
        "messages",
    }
    if missing := required_columns - set(unified_train.columns):
        raise ValueError(f"unified training data is missing columns: {sorted(missing)}")
    if "matched_control_source" not in matched_train:
        raise ValueError("matched training data has no matched_control_source column")

    causal_planners = unified_train[
        (unified_train["unified_source"].astype(str) == "new")
        & (unified_train["agent_name"].astype(str) == PLANNER_ROLE)
    ].copy()
    replay_planners = matched_train[
        (matched_train["matched_control_source"].astype(str) == "matched_extra")
        & (matched_train["agent_name"].astype(str) == PLANNER_ROLE)
    ].copy()
    if len(causal_planners) != len(replay_planners):
        raise ValueError(
            "causal and role-matched Planner pools differ: "
            f"{len(causal_planners)} != {len(replay_planners)}"
        )
    if not 0 <= args.causal_planner_count <= len(causal_planners):
        raise ValueError(
            f"causal_planner_count must be in [0, {len(causal_planners)}]"
        )

    ordered_causal = stable_order(causal_planners, args.seed)
    ordered_replay = stable_order(replay_planners, args.seed + 1)
    retained = ordered_causal.iloc[: args.causal_planner_count].copy()
    replacement_count = len(causal_planners) - len(retained)
    replacements = ordered_replay.iloc[args.causal_planner_count :].copy()

    # Every causal Planner owns a fixed slot and a deterministic replay partner.
    # Across doses, only the contents of those slots change; all sampler indices stay aligned.
    causal_rank_by_index = {
        index: rank for rank, index in enumerate(ordered_causal.index)
    }
    output_columns = list(
        dict.fromkeys(
            [
                *unified_train.columns,
                *matched_train.columns,
                "planner_dose_source",
                "planner_dose_pair_rank",
            ]
        )
    )
    records = []
    for index, unified_row in unified_train.iterrows():
        pair_rank = causal_rank_by_index.get(index)
        if pair_rank is None:
            record = unified_row.to_dict()
            record["planner_dose_source"] = "unchanged"
            record["planner_dose_pair_rank"] = -1
        elif pair_rank < args.causal_planner_count:
            record = unified_row.to_dict()
            record["planner_dose_source"] = "causal_planner"
            record["planner_dose_pair_rank"] = pair_rank
        else:
            record = ordered_replay.iloc[pair_rank].to_dict()
            record["unified_source"] = "planner_replay"
            record["planner_dose_source"] = "planner_replay"
            record["planner_dose_pair_rank"] = pair_rank
        records.append(record)
    train = pd.DataFrame.from_records(records).reindex(columns=output_columns)

    val = unified_val.copy()
    val["planner_dose_source"] = "unchanged_validation"
    train_ids = set(train["trajectory_id"].astype(str))
    val_ids = set(val["trajectory_id"].astype(str))
    nonplanner_new = train[
        (train["unified_source"].astype(str) == "new")
        & (train["agent_name"].astype(str) != PLANNER_ROLE)
    ]
    output_nonplanners = train[train["agent_name"].astype(str) != PLANNER_ROLE]
    input_nonplanners = unified_train[
        unified_train["agent_name"].astype(str) != PLANNER_ROLE
    ]
    output_nonplanner_positions = list(
        train.index[train["agent_name"].astype(str) != PLANNER_ROLE]
    )
    input_nonplanner_positions = list(
        unified_train.index[
            unified_train["agent_name"].astype(str) != PLANNER_ROLE
        ]
    )
    output_planner_positions = list(
        train.index[train["agent_name"].astype(str) == PLANNER_ROLE]
    )
    input_planner_positions = list(
        unified_train.index[
            unified_train["agent_name"].astype(str) == PLANNER_ROLE
        ]
    )
    active_pair_ranks = sorted(
        int(value)
        for value in train.loc[
            train["planner_dose_source"].isin(
                ["causal_planner", "planner_replay"]
            ),
            "planner_dose_pair_rank",
        ]
    )
    hard_gates = {
        "canonical_train_prompts": prompt_mismatch_count(train) == 0,
        "canonical_val_prompts": prompt_mismatch_count(val) == 0,
        "valid_train_loss_masks": all(
            map(assistant_loss_mask_is_valid, train["messages"])
        ),
        "valid_val_loss_masks": all(map(assistant_loss_mask_is_valid, val["messages"])),
        "train_val_trajectory_disjoint": not bool(train_ids & val_ids),
        "train_rows_unchanged": len(train) == len(unified_train),
        "role_counts_unchanged": role_counts(train) == role_counts(unified_train),
        "causal_planner_count_exact": sum(
            train["planner_dose_source"] == "causal_planner"
        )
        == args.causal_planner_count,
        "planner_replacement_count_exact": sum(
            train["planner_dose_source"] == "planner_replay"
        )
        == replacement_count,
        "all_replacements_are_planners": set(
            train.loc[
                train["planner_dose_source"] == "planner_replay", "agent_name"
            ].astype(str)
        )
        <= {PLANNER_ROLE},
        "nonplanner_causal_rows_unchanged": len(nonplanner_new)
        == sum(
            (unified_train["unified_source"].astype(str) == "new")
            & (unified_train["agent_name"].astype(str) != PLANNER_ROLE)
        ),
        "nonplanner_rows_exact": frame_row_multiset(
            output_nonplanners,
            columns=set(unified_train.columns),
            ignored_columns={"planner_dose_source"},
        )
        == frame_row_multiset(
            input_nonplanners,
            columns=set(unified_train.columns),
            ignored_columns={"planner_dose_source"},
        ),
        "validation_rows_exact": frame_row_multiset(
            val,
            columns=set(unified_val.columns),
            ignored_columns={"planner_dose_source"},
        )
        == frame_row_multiset(
            unified_val,
            columns=set(unified_val.columns),
            ignored_columns={"planner_dose_source"},
        ),
        "nonplanner_sampler_slots_fixed": output_nonplanner_positions
        == input_nonplanner_positions,
        "planner_sampler_slots_fixed": output_planner_positions
        == input_planner_positions,
        "planner_pair_ranks_exact": active_pair_ranks
        == list(range(len(causal_planners))),
        "full_batches_exact": len(train) % args.batch_size == 0,
    }
    if not all(hard_gates.values()):
        raise ValueError(f"Planner-dose SFT hard gates failed: {hard_gates}")

    args.output_dir.mkdir(parents=True)
    output_paths = {}
    for split, frame in (("train", train), ("val", val)):
        path = args.output_dir / f"sft_{split}.parquet"
        frame.to_parquet(path, index=False)
        output_paths[split] = path

    report = {
        "method": "unified_sft_causal_planner_slot_aligned_dose_v2",
        "prompt_schema": "qa_manager.ROSA.protocol.max5.v2",
        "seed": args.seed,
        "unified_data_dir": str(args.unified_data_dir),
        "matched_data_dir": str(args.matched_data_dir),
        "causal_planner_pool": len(causal_planners),
        "causal_planner_count": len(retained),
        "planner_replacement_count": replacement_count,
        "sampler_slot_alignment": {
            "nonplanner_slots": len(output_nonplanner_positions),
            "planner_slots": len(output_planner_positions),
            "paired_causal_planner_slots": len(causal_planners),
            "distributed_sampler_seed": 0,
        },
        "planner_pairs": [
            {
                "pair_rank": rank,
                "slot_index": int(causal_index),
                "causal_trajectory_id": str(
                    ordered_causal.loc[causal_index, "trajectory_id"]
                ),
                "replay_trajectory_id": str(
                    ordered_replay.iloc[rank]["trajectory_id"]
                ),
                "active_source": (
                    "causal_planner"
                    if rank < args.causal_planner_count
                    else "planner_replay"
                ),
            }
            for rank, causal_index in enumerate(ordered_causal.index)
        ],
        "retained_causal_planner_trajectory_ids": sorted(
            retained["trajectory_id"].astype(str)
        ),
        "nonplanner_causal_rows": len(nonplanner_new),
        "train_rows": len(train),
        "train_role_rows": dict(sorted(role_counts(train).items())),
        "optimizer_steps": len(train) // args.batch_size,
        "prompt_mismatches_replaced": {
            "unified_train": unified_train_mismatches,
            "unified_val": unified_val_mismatches,
            "matched_train": matched_train_mismatches,
        },
        "hard_gates": hard_gates,
        "files": {
            split: {"path": str(path), "sha256": sha256(path)}
            for split, path in output_paths.items()
        },
    }
    (args.output_dir / "planner_dose_sft_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
