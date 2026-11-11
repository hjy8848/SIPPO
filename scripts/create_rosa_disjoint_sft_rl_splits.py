#!/usr/bin/env python3
"""Create deterministic, question-disjoint 2Wiki pools for SFT and online RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if not isinstance(value, dict):
        raise ValueError(f"extra_info must be a dict, got {type(value).__name__}")
    return dict(value)


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def question_keys(frame: pd.DataFrame) -> list[str]:
    if "extra_info" not in frame.columns:
        raise ValueError("parquet file has no extra_info column")
    keys = [normalize_question(as_dict(item).get("question", "")) for item in frame["extra_info"]]
    if any(not key for key in keys):
        raise ValueError("every row must have a non-empty extra_info.question")
    return keys


def annotate(frame: pd.DataFrame, partition: str, original_indices: list[int]) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    updated: list[dict[str, Any]] = []
    for item, original_index in zip(result["extra_info"], original_indices, strict=True):
        extra_info = as_dict(item)
        extra_info["split"] = partition
        extra_info["rosa_partition"] = partition
        extra_info["original_train_index"] = int(original_index)
        updated.append(extra_info)
    result["extra_info"] = updated
    return result


def key_digest(keys: set[str]) -> str:
    payload = "\n".join(sorted(keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/multihop_qa/train_2wiki_no_test_overlap.parquet",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=PROJECT_ROOT / "data/multihop_qa/validation_2wiki_heldout.parquet",
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=PROJECT_ROOT / "data/multihop_qa/sft_source_2wiki_seed42.parquet",
    )
    parser.add_argument(
        "--rl-output",
        type=Path,
        default=PROJECT_ROOT / "data/multihop_qa/rl_train_2wiki_no_sft_seed42.parquet",
    )
    parser.add_argument("--sft-pool-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for path in (args.input, args.validation):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.sft_output.resolve() == args.rl_output.resolve():
        raise ValueError("--sft-output and --rl-output must differ")
    for path in (args.sft_output, args.rl_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {path}; use --overwrite intentionally")

    source = pd.read_parquet(args.input)
    validation = pd.read_parquet(args.validation)
    source_keys = question_keys(source)
    validation_keys = set(question_keys(validation))
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("source contains duplicate normalized questions")
    source_validation_overlap = set(source_keys) & validation_keys
    if source_validation_overlap:
        raise ValueError(
            f"source overlaps held-out validation by {len(source_validation_overlap)} questions"
        )
    if not 0 < args.sft_pool_size < len(source):
        raise ValueError("--sft-pool-size must be between 1 and input_rows - 1")

    shuffled_indices = list(range(len(source)))
    random.Random(args.seed).shuffle(shuffled_indices)
    sft_indices = sorted(shuffled_indices[: args.sft_pool_size])
    rl_indices = sorted(shuffled_indices[args.sft_pool_size :])
    sft_frame = annotate(source.iloc[sft_indices], "sft_source", sft_indices)
    rl_frame = annotate(source.iloc[rl_indices], "train", rl_indices)

    sft_keys = set(question_keys(sft_frame))
    rl_keys = set(question_keys(rl_frame))
    overlaps = {
        "sft_vs_rl": len(sft_keys & rl_keys),
        "sft_vs_validation": len(sft_keys & validation_keys),
        "rl_vs_validation": len(rl_keys & validation_keys),
    }
    if any(overlaps.values()):
        raise AssertionError(f"disjoint split invariant failed: {overlaps}")

    args.sft_output.parent.mkdir(parents=True, exist_ok=True)
    args.rl_output.parent.mkdir(parents=True, exist_ok=True)
    sft_frame.to_parquet(args.sft_output, index=False)
    rl_frame.to_parquet(args.rl_output, index=False)
    report = {
        "source": str(args.input),
        "validation": str(args.validation),
        "seed": args.seed,
        "source_rows": len(source),
        "sft_source_rows": len(sft_frame),
        "rl_train_rows": len(rl_frame),
        "validation_rows": len(validation),
        "overlaps": overlaps,
        "question_set_sha256": {
            "sft_source": key_digest(sft_keys),
            "rl_train": key_digest(rl_keys),
            "validation": key_digest(validation_keys),
        },
        "sft_output": str(args.sft_output),
        "rl_output": str(args.rl_output),
    }
    report_path = args.sft_output.with_suffix(".split_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[DONE] split report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
