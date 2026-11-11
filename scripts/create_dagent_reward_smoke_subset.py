#!/usr/bin/env python3
"""Create tiny deterministic train/validation subsets for reward pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import pandas as pd


def to_python(value):
    if hasattr(value, "as_py"):
        return to_python(value.as_py())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        return to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    return value


def is_reason_required(row: pd.Series) -> bool:
    if str(row.get("data_source", "")).casefold() != "2wiki":
        return False
    info = to_python(row.get("extra_info", {}))
    return isinstance(info, dict) and info.get("question_type") in {
        "comparison",
        "bridge_comparison",
    }


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def select(path: Path, count: int, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    mask = frame.apply(is_reason_required, axis=1)
    candidates = frame[mask].reset_index(drop=True)
    if len(candidates) < count:
        raise ValueError(f"{path} has only {len(candidates)} reason-required 2Wiki rows")
    return candidates.sample(n=count, random_state=seed).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=4)
    parser.add_argument("--validation-rows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.train_output, args.validation_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {path}; pass --overwrite intentionally")

    train = select(args.train_input, args.train_rows, args.seed)
    validation = select(args.validation_input, args.validation_rows, args.seed + 1)
    atomic_parquet(train, args.train_output)
    atomic_parquet(validation, args.validation_output)
    print(f"[DONE] train={args.train_output} rows={len(train)}")
    print(f"[DONE] validation={args.validation_output} rows={len(validation)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
