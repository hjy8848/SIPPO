#!/usr/bin/env python3
"""Print compact schemas and examples for DAGENT source datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def compact(value: Any, limit: int = 4000) -> str:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        value = value.tolist()
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = repr(value)
    return text[:limit]


def inspect_parquet(path: Path) -> None:
    frame = pd.read_parquet(path)
    print(f"PARQUET path={path} rows={len(frame)} columns={list(frame.columns)}")
    if len(frame):
        print(compact(frame.iloc[0].to_dict()))


def inspect_arrow(path: Path) -> None:
    from datasets import Dataset

    dataset = Dataset.from_file(str(path))
    print(f"ARROW path={path} rows={len(dataset)} columns={dataset.column_names}")
    if len(dataset):
        print(compact(dataset[0]))


def inspect_json(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            value = json.load(handle)
            print(f"JSON path={path} rows={len(value)}")
            if value:
                print(compact(value[0]))
            return
        line = next((line for line in handle if line.strip()), "")
    print(f"JSONL path={path}")
    if line:
        print(compact(json.loads(line)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        if path.suffix == ".parquet":
            inspect_parquet(path)
        elif path.suffix == ".arrow":
            inspect_arrow(path)
        elif path.suffix in {".json", ".jsonl"}:
            inspect_json(path)
        else:
            print(f"SKIP unsupported path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
