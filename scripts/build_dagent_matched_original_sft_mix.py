#!/usr/bin/env python3
"""Build an original-only SFT control matched to a unified causal mix."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-data-dir", type=Path, required=True)
    parser.add_argument("--unified-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-full-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_counts(frame: pd.DataFrame) -> Counter[str]:
    return Counter(map(str, frame["agent_name"]))


def sample_role_matched_originals(
    original: pd.DataFrame, targets: Counter[str], seed: int
) -> pd.DataFrame:
    selected = []
    for role in sorted(targets):
        candidates = original[original["agent_name"].astype(str) == role]
        count = targets[role]
        if count > len(candidates):
            raise ValueError(
                f"need {count} original {role} rows, only {len(candidates)} available"
            )
        selected.append(candidates.sample(n=count, replace=False, random_state=seed))
        seed += 1
    return pd.concat(selected, ignore_index=True, sort=False)


def report_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "trajectories": int(frame["trajectory_id"].astype(str).nunique()),
        "role_rows": dict(sorted(role_counts(frame).items())),
        "source_rows": dict(
            sorted(Counter(map(str, frame["matched_control_source"])).items())
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.original_full_repeats <= 0 or args.batch_size <= 0:
        raise ValueError("repeat count and batch size must be positive")

    original_train_path = args.original_data_dir / "sft_train.parquet"
    original_val_path = args.original_data_dir / "sft_val.parquet"
    unified_train_path = args.unified_data_dir / "sft_train.parquet"
    for path in (original_train_path, original_val_path, unified_train_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing matched-control input: {path}")

    original_train, train_mismatches = canonicalize_frame(
        pd.read_parquet(original_train_path), "replay"
    )
    original_val, val_mismatches = canonicalize_frame(
        pd.read_parquet(original_val_path), "replay"
    )
    unified_train = pd.read_parquet(unified_train_path)
    if "unified_source" not in unified_train:
        raise ValueError("unified training data has no unified_source column")
    causal = unified_train[unified_train["unified_source"].astype(str) == "new"]
    causal_targets = role_counts(causal)
    if not causal_targets:
        raise ValueError("unified training data has no causal rows")

    matched_extra = sample_role_matched_originals(
        original_train, causal_targets, args.seed
    )
    matched_extra["matched_control_source"] = "matched_extra"
    matched_extra["exposure_repeat"] = args.original_full_repeats

    full_copies = []
    for repeat in range(args.original_full_repeats):
        copy = original_train.copy()
        copy["matched_control_source"] = "original_full"
        copy["exposure_repeat"] = repeat
        full_copies.append(copy)
    train = pd.concat([*full_copies, matched_extra], ignore_index=True, sort=False)
    train = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    val = original_val.copy()
    val["matched_control_source"] = "original_val"
    val["exposure_repeat"] = 0

    expected_rows = len(original_train) * args.original_full_repeats + len(causal)
    train_ids = set(train["trajectory_id"].astype(str))
    val_ids = set(val["trajectory_id"].astype(str))
    hard_gates = {
        "canonical_train_prompts": prompt_mismatch_count(train) == 0,
        "canonical_val_prompts": prompt_mismatch_count(val) == 0,
        "valid_train_loss_masks": all(map(assistant_loss_mask_is_valid, train["messages"])),
        "valid_val_loss_masks": all(map(assistant_loss_mask_is_valid, val["messages"])),
        "train_val_trajectory_disjoint": not bool(train_ids & val_ids),
        "matched_extra_role_counts_equal_causal": role_counts(matched_extra) == causal_targets,
        "matched_extra_rows_equal_causal": len(matched_extra) == len(causal),
        "train_rows_exact": len(train) == expected_rows,
        "full_batches_exact": len(train) % args.batch_size == 0,
    }
    if not all(hard_gates.values()):
        raise ValueError(f"matched original SFT hard gates failed: {hard_gates}")

    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "sft_train.parquet"
    val_path = args.output_dir / "sft_val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    report = {
        "method": "matched_original_only_sft_v1",
        "prompt_schema": "qa_manager.ROSA.protocol.v1",
        "seed": args.seed,
        "original_data_dir": str(args.original_data_dir),
        "unified_data_dir": str(args.unified_data_dir),
        "original_full_repeats": args.original_full_repeats,
        "causal_rows_replaced": len(causal),
        "causal_role_targets": dict(sorted(causal_targets.items())),
        "optimizer_steps": len(train) // args.batch_size,
        "prompt_mismatches_replaced": {
            "train": train_mismatches,
            "val": val_mismatches,
        },
        "train": report_frame(train),
        "val": report_frame(val),
        "hard_gates": hard_gates,
        "files": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "val": {"path": str(val_path), "sha256": sha256(val_path)},
        },
    }
    (args.output_dir / "matched_original_sft_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
