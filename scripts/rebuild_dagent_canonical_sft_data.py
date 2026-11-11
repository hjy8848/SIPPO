#!/usr/bin/env python3
"""Rebuild an existing DAGENT SFT dataset with canonical runtime prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_report(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "trajectories": int(frame["trajectory_id"].astype(str).nunique()),
        "role_rows": {
            str(key): int(value)
            for key, value in frame["agent_name"].astype(str).value_counts().sort_index().items()
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    frames: dict[str, pd.DataFrame] = {}
    mismatches: dict[str, int] = {}
    input_paths: dict[str, Path] = {}
    for split in ("train", "val"):
        input_path = args.input_dir / f"sft_{split}.parquet"
        if not input_path.is_file():
            raise FileNotFoundError(f"missing SFT split: {input_path}")
        original = pd.read_parquet(input_path)
        canonical, mismatch_count = canonicalize_frame(original, "replay")
        if len(canonical) != len(original):
            raise ValueError(f"{split} row count changed during canonicalization")
        frames[split] = canonical
        mismatches[split] = mismatch_count
        input_paths[split] = input_path

    train_ids = set(frames["train"]["trajectory_id"].astype(str))
    val_ids = set(frames["val"]["trajectory_id"].astype(str))
    hard_gates = {
        "canonical_train_prompts": prompt_mismatch_count(frames["train"]) == 0,
        "canonical_val_prompts": prompt_mismatch_count(frames["val"]) == 0,
        "valid_train_loss_masks": all(
            map(assistant_loss_mask_is_valid, frames["train"]["messages"])
        ),
        "valid_val_loss_masks": all(
            map(assistant_loss_mask_is_valid, frames["val"]["messages"])
        ),
        "train_val_trajectory_disjoint": not bool(train_ids & val_ids),
    }
    if not all(hard_gates.values()):
        raise ValueError(f"canonical SFT rebuild hard gates failed: {hard_gates}")

    args.output_dir.mkdir(parents=True)
    output_paths = {}
    for split, frame in frames.items():
        output_path = args.output_dir / f"sft_{split}.parquet"
        frame.to_parquet(output_path, index=False)
        output_paths[split] = output_path

    report = {
        "method": "canonical_original_sft_rebuild_v1",
        "prompt_schema": "qa_manager.ROSA.protocol.v1",
        "input_dir": str(args.input_dir),
        "prompt_mismatches_replaced": mismatches,
        "train": split_report(frames["train"]),
        "val": split_report(frames["val"]),
        "hard_gates": hard_gates,
        "files": {
            split: {
                "input": str(input_paths[split]),
                "input_sha256": sha256(input_paths[split]),
                "output": str(output_paths[split]),
                "output_sha256": sha256(output_paths[split]),
            }
            for split in ("train", "val")
        },
    }
    (args.output_dir / "canonical_sft_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
