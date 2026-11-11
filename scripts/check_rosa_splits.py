#!/usr/bin/env python3
"""Check ROSA train/validation question overlap and optionally filter train."""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalize_question(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def extract_question(extra_info: object) -> str:
    if hasattr(extra_info, "as_py"):
        extra_info = extra_info.as_py()
    if isinstance(extra_info, dict):
        return str(extra_info.get("question", ""))
    raise ValueError(f"extra_info must be a dict, got {type(extra_info).__name__}")


def set_split(extra_info: object, split: str) -> dict:
    if hasattr(extra_info, "as_py"):
        extra_info = extra_info.as_py()
    if not isinstance(extra_info, dict):
        raise ValueError(f"extra_info must be a dict, got {type(extra_info).__name__}")
    updated = dict(extra_info)
    updated["split"] = split
    return updated


def question_keys(frame: pd.DataFrame) -> pd.Series:
    if "extra_info" not in frame.columns:
        raise ValueError("Parquet file has no extra_info column")
    return frame["extra_info"].map(extract_question).map(normalize_question)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--write-filtered-train", type=Path)
    parser.add_argument("--write-validation", type=Path)
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    train_keys = question_keys(train)
    validation_keys = question_keys(validation)
    overlap = set(train_keys) & set(validation_keys)

    print(f"train_rows={len(train)} validation_rows={len(validation)} overlap={len(overlap)}")
    if args.write_validation is not None:
        validation = validation.copy()
        validation["extra_info"] = validation["extra_info"].map(lambda item: set_split(item, "validation"))
        args.write_validation.parent.mkdir(parents=True, exist_ok=True)
        validation.to_parquet(args.write_validation, index=False)
        print(f"wrote_validation={args.write_validation} rows={len(validation)}")
    if not overlap:
        return 0

    if args.write_filtered_train is None:
        examples = sorted(overlap)[:5]
        print(f"overlap_examples={examples}")
        return 1

    keep = ~train_keys.isin(overlap)
    filtered = train.loc[keep].reset_index(drop=True)
    args.write_filtered_train.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.write_filtered_train, index=False)
    print(f"wrote={args.write_filtered_train} rows={len(filtered)} removed={len(train) - len(filtered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
