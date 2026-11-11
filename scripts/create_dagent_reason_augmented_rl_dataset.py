#!/usr/bin/env python3
"""Mix audited Reason-to-Search rows into the existing 2Wiki/Hotpot RL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}


def to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    return value


def question_from_row(row: pd.Series) -> str:
    info = to_python(row.get("extra_info", {}))
    if isinstance(info, dict) and str(info.get("question", "")).strip():
        return str(info["question"]).strip()
    reward_model = to_python(row.get("reward_model", {}))
    if isinstance(reward_model, dict):
        for key in ("question", "ground_truth"):
            if str(reward_model.get(key, "")).strip():
                return str(reward_model[key]).strip()
    prompt = to_python(row.get("prompt", []))
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", "")).strip()
    return ""


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.copy()


def question_keys(frame: pd.DataFrame) -> list[str]:
    keys = [normalize_question(question_from_row(row)) for _, row in frame.iterrows()]
    if any(not key for key in keys):
        index = next(i for i, key in enumerate(keys) if not key)
        raise ValueError(f"row {index} has no usable question")
    return keys


def expanded_reason_rows(frame: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    if target_rows <= 0:
        raise ValueError("--target-reason-train-rows must be positive")
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)
    selected = [indices[index % len(indices)] for index in range(target_rows)]
    result = frame.iloc[selected].copy().reset_index(drop=True)
    occurrence: dict[str, int] = {}
    updated_info = []
    for _, row in result.iterrows():
        key = normalize_question(question_from_row(row))
        repeat_index = occurrence.get(key, 0)
        occurrence[key] = repeat_index + 1
        info = to_python(row.get("extra_info", {}))
        info = dict(info) if isinstance(info, dict) else {}
        info["dagent_reason_augmentation"] = True
        info["dagent_reason_repeat_index"] = repeat_index
        updated_info.append(info)
    result["extra_info"] = updated_info
    return result


def shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)
    return frame.iloc[indices].reset_index(drop=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def hash_keys(keys: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--reason-train", type=Path, required=True)
    parser.add_argument("--reason-validation", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    parser.add_argument("--target-reason-train-rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.output_train, args.output_validation):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {path}; use --overwrite intentionally")

    base_train = load(args.base_train)
    base_validation = load(args.base_validation)
    reason_train = load(args.reason_train)
    reason_validation = load(args.reason_validation)
    expanded = expanded_reason_rows(reason_train, args.target_reason_train_rows, args.seed)

    base_train_keys = set(question_keys(base_train))
    base_validation_keys = set(question_keys(base_validation))
    reason_train_keys = set(question_keys(reason_train))
    reason_validation_keys = set(question_keys(reason_validation))
    train_keys = base_train_keys | reason_train_keys
    validation_keys = base_validation_keys | reason_validation_keys
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(f"train/validation question overlap: {len(overlap)}")

    train = shuffled(pd.concat([base_train, expanded], ignore_index=True), args.seed)
    validation = shuffled(pd.concat([base_validation, reason_validation], ignore_index=True), args.seed + 1)
    atomic_parquet(train, args.output_train)
    atomic_parquet(validation, args.output_validation)

    report = {
        "seed": args.seed,
        "base_train_rows": len(base_train),
        "unique_reason_train_rows": len(reason_train),
        "expanded_reason_train_rows": len(expanded),
        "total_train_rows": len(train),
        "reason_train_fraction": len(expanded) / len(train),
        "base_validation_rows": len(base_validation),
        "reason_validation_rows": len(reason_validation),
        "total_validation_rows": len(validation),
        "train_validation_question_overlap": len(overlap),
        "question_set_sha256": {
            "train": hash_keys(train_keys),
            "validation": hash_keys(validation_keys),
        },
        "inputs": {
            "base_train": str(args.base_train),
            "base_validation": str(args.base_validation),
            "reason_train": str(args.reason_train),
            "reason_validation": str(args.reason_validation),
        },
        "outputs": {
            "train": str(args.output_train),
            "validation": str(args.output_validation),
        },
    }
    report_path = args.output_train.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
