#!/usr/bin/env python3
"""Create a deterministic 2Wiki/Hotpot pilot with optional Reason-to-Search rows."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd


def normalize_source(value: object) -> str:
    source = str(value or "").strip().casefold().replace("-", "").replace("_", "")
    if source in {"2wiki", "2wikimultihopqa"}:
        return "2wiki"
    if source in {"hotpot", "hotpotqa"}:
        return "hotpotqa"
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=40)
    parser.add_argument("--reason-input", type=Path)
    parser.add_argument("--reason-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}")
    if args.reason_count < 0:
        parser.error("--reason-count must be non-negative")
    if args.reason_count and args.reason_input is None:
        parser.error("--reason-input is required when --reason-count is positive")
    frame = pd.read_parquet(args.input)
    if "data_source" not in frame:
        raise ValueError(f"Missing data_source column in {args.input}")
    normalized = frame["data_source"].map(normalize_source)
    chunks = []
    for offset, source in enumerate(("2wiki", "hotpotqa")):
        candidates = frame.loc[normalized == source]
        if len(candidates) < args.per_source:
            raise ValueError(f"Need {args.per_source} {source} rows, found {len(candidates)}")
        chunks.append(candidates.sample(n=args.per_source, random_state=args.seed + offset))
    if args.reason_count:
        reason_frame = pd.read_parquet(args.reason_input)
        if "extra_info" in reason_frame:
            reason_splits = reason_frame["extra_info"].map(
                lambda value: str(value.get("split") or "").strip().casefold()
                if hasattr(value, "get")
                else ""
            )
            if (reason_splits == "train").any():
                reason_frame = reason_frame.loc[reason_splits == "train"]
        if len(reason_frame) < args.reason_count:
            raise ValueError(
                f"Need {args.reason_count} Reason-to-Search rows, found {len(reason_frame)}"
            )
        chunks.append(
            reason_frame.sample(n=args.reason_count, random_state=args.seed + 2)
        )
    output = pd.concat(chunks, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        output.to_parquet(temporary, index=False)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "rows": len(output),
        "reason_input": str(args.reason_input) if args.reason_input else None,
        "reason_count": args.reason_count,
        "source_counts": output["data_source"].map(normalize_source).value_counts().sort_index().to_dict(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
