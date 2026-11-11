#!/usr/bin/env python3
"""Filter a DAGENT parquet split by data source without changing row contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd


def normalize_source(value: object) -> str:
    source = str(value or "").strip().casefold().replace("-", "").replace("_", "")
    if source in {"2wiki", "2wikimultihopqa"}:
        return "2wiki"
    if source in {"hotpot", "hotpotqa"}:
        return "hotpotqa"
    return source


def atomic_parquet(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".parquet", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", default="2wiki,hotpotqa")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"input parquet does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite intentionally"
        )

    selected_sources = {
        normalize_source(source) for source in args.sources.split(",") if source.strip()
    }
    if not selected_sources:
        raise ValueError("--sources must contain at least one data source")

    frame = pd.read_parquet(args.input)
    if "data_source" not in frame.columns:
        raise ValueError(f"{args.input} has no data_source column")

    normalized = frame["data_source"].map(normalize_source)
    available_counts = normalized.value_counts().sort_index().to_dict()
    filtered = frame.loc[normalized.isin(selected_sources)].copy()
    retained_counts = (
        filtered["data_source"].map(normalize_source).value_counts().sort_index().to_dict()
    )
    missing = selected_sources.difference(retained_counts)
    if missing:
        raise ValueError(
            f"requested sources are absent from {args.input}: {sorted(missing)}; "
            f"available={available_counts}"
        )

    atomic_parquet(filtered, args.output)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "selected_sources": sorted(selected_sources),
        "input_rows": len(frame),
        "output_rows": len(filtered),
        "removed_rows": len(frame) - len(filtered),
        "input_source_counts": available_counts,
        "output_source_counts": retained_counts,
        "columns_preserved": list(frame.columns) == list(filtered.columns),
    }
    report_path = args.output.with_suffix(".filter_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] data={args.output}")
    print(f"[DONE] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
