#!/usr/bin/env python3
"""Build a from-base SFT mix with matched original and causal exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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


EXPECTED_CAUSAL_ROLES = {
    "PlanningAgent": 1,
    "AnswerGenerationAgent": 3,
    "ReasonerAgent": 1,
    "QueryRewriteAgent": 1,
    "AnswerSummarizationAgent": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-data-dir", type=Path, required=True)
    parser.add_argument("--replay-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--new-train-trajectories", type=int, default=114)
    parser.add_argument("--replay-train-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_role_counts(frame: pd.DataFrame) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = {}
    for trajectory_id, rows in frame.groupby(frame["trajectory_id"].astype(str)):
        result[str(trajectory_id)] = Counter(str(value) for value in rows["agent_name"])
    return result


def select_complete_trajectories(
    frame: pd.DataFrame, count: int, seed: int
) -> tuple[pd.DataFrame, list[str]]:
    if count <= 0:
        raise ValueError("new_train_trajectories must be positive")
    role_counts = trajectory_role_counts(frame)
    malformed = {
        trajectory_id: dict(counts)
        for trajectory_id, counts in role_counts.items()
        if dict(counts) != EXPECTED_CAUSAL_ROLES
    }
    if malformed:
        examples = dict(list(malformed.items())[:5])
        raise ValueError(f"causal trajectories are incomplete: {examples}")
    trajectory_ids = sorted(role_counts)
    if count > len(trajectory_ids):
        raise ValueError(
            f"requested {count} causal trajectories, only {len(trajectory_ids)} available"
        )
    random.Random(seed).shuffle(trajectory_ids)
    selected_ids = sorted(trajectory_ids[:count])
    selected = frame[frame["trajectory_id"].astype(str).isin(selected_ids)].copy()
    return selected, selected_ids


def frame_report(frame: pd.DataFrame) -> dict[str, Any]:
    sources = Counter(str(value) for value in frame["unified_source"])
    return {
        "rows": len(frame),
        "trajectories": int(frame["trajectory_id"].astype(str).nunique()),
        "source_rows": dict(sorted(sources.items())),
        "role_rows": dict(sorted(Counter(map(str, frame["agent_name"])).items())),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.replay_train_repeats <= 0:
        raise ValueError("replay_train_repeats must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    paths: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val"):
        new_path = args.new_data_dir / f"sft_{split}.parquet"
        replay_path = args.replay_data_dir / f"sft_{split}.parquet"
        if not new_path.is_file() or not replay_path.is_file():
            raise FileNotFoundError(f"missing {split} input: {new_path} or {replay_path}")
        paths[split] = (new_path, replay_path)

    new_train, new_train_mismatches = canonicalize_frame(
        pd.read_parquet(paths["train"][0]), "new"
    )
    replay_train, replay_train_mismatches = canonicalize_frame(
        pd.read_parquet(paths["train"][1]), "replay"
    )
    selected_new, selected_ids = select_complete_trajectories(
        new_train, args.new_train_trajectories, args.seed
    )
    selected_new["unified_source"] = "new"
    selected_new["exposure_repeat"] = 0

    replay_copies = []
    for repeat in range(args.replay_train_repeats):
        copy = replay_train.copy()
        copy["unified_source"] = "replay"
        copy["exposure_repeat"] = repeat
        replay_copies.append(copy)
    train = pd.concat([selected_new, *replay_copies], ignore_index=True, sort=False)
    train = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    new_val, new_val_mismatches = canonicalize_frame(
        pd.read_parquet(paths["val"][0]), "new"
    )
    replay_val, replay_val_mismatches = canonicalize_frame(
        pd.read_parquet(paths["val"][1]), "replay"
    )
    new_val["unified_source"] = "new"
    replay_val["unified_source"] = "replay"
    new_val["exposure_repeat"] = 0
    replay_val["exposure_repeat"] = 0
    val = pd.concat([new_val, replay_val], ignore_index=True, sort=False)
    val = val.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    train_ids = set(train["trajectory_id"].astype(str))
    val_ids = set(val["trajectory_id"].astype(str))
    expected_new_rows = args.new_train_trajectories * sum(EXPECTED_CAUSAL_ROLES.values())
    expected_replay_rows = len(replay_train) * args.replay_train_repeats
    expected_train_rows = expected_new_rows + expected_replay_rows
    selected_role_counts = Counter(map(str, selected_new["agent_name"]))
    expected_selected_roles = {
        role: count * args.new_train_trajectories
        for role, count in EXPECTED_CAUSAL_ROLES.items()
    }
    hard_gates = {
        "canonical_train_prompts": prompt_mismatch_count(train) == 0,
        "canonical_val_prompts": prompt_mismatch_count(val) == 0,
        "valid_train_loss_masks": all(map(assistant_loss_mask_is_valid, train["messages"])),
        "valid_val_loss_masks": all(map(assistant_loss_mask_is_valid, val["messages"])),
        "train_val_trajectory_disjoint": not bool(train_ids & val_ids),
        "selected_causal_trajectories_exact": len(selected_ids) == args.new_train_trajectories,
        "selected_causal_roles_exact": dict(selected_role_counts) == expected_selected_roles,
        "new_rows_exact": len(selected_new) == expected_new_rows,
        "replay_rows_exact": sum(train["unified_source"] == "replay") == expected_replay_rows,
        "train_rows_exact": len(train) == expected_train_rows,
        "full_batches_exact": len(train) % args.batch_size == 0,
    }
    if not all(hard_gates.values()):
        raise ValueError(f"unified SFT mix hard gates failed: {hard_gates}")

    args.output_dir.mkdir(parents=True)
    output_paths = {}
    for split, frame in (("train", train), ("val", val)):
        path = args.output_dir / f"sft_{split}.parquet"
        frame.to_parquet(path, index=False)
        output_paths[split] = path

    report = {
        "method": "from_base_unified_sft_matched_exposure_v1",
        "prompt_schema": "qa_manager.ROSA.protocol.v1",
        "seed": args.seed,
        "new_data_dir": str(args.new_data_dir),
        "replay_data_dir": str(args.replay_data_dir),
        "new_train_trajectories": args.new_train_trajectories,
        "selected_new_trajectory_ids": selected_ids,
        "replay_train_repeats": args.replay_train_repeats,
        "batch_size": args.batch_size,
        "optimizer_steps": len(train) // args.batch_size,
        "prompt_mismatches_replaced": {
            "train": {"new": new_train_mismatches, "replay": replay_train_mismatches},
            "val": {"new": new_val_mismatches, "replay": replay_val_mismatches},
        },
        "train": frame_report(train),
        "val": frame_report(val),
        "hard_gates": hard_gates,
        "files": {
            split: {"path": str(path), "sha256": sha256(path)}
            for split, path in output_paths.items()
        },
    }
    (args.output_dir / "unified_sft_mix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
