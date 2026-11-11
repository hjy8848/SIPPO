#!/usr/bin/env python3
"""Build a deterministic paired pilot subset from the frozen formal-v2 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd


DEFAULT_COUNTS = {
    "2wikimultihopqa": 50,
    "hotpotqa": 50,
    "musique": 50,
    "bamboogle": 25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def question_digest(frame: pd.DataFrame) -> str:
    payload = "\n".join(str(value).strip() for value in frame["question"])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(source_dir: Path, output_dir: Path, seed: int) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot manifest: {output_dir}")
    output_dir.mkdir(parents=True)

    report = {
        "schema_version": 1,
        "method": "deterministic_subset_of_frozen_formal_v2",
        "source_dir": str(source_dir.resolve()),
        "seed": seed,
        "datasets": {},
    }
    for offset, (dataset, requested) in enumerate(DEFAULT_COUNTS.items()):
        source_path = source_dir / f"{dataset}.parquet"
        frame = pd.read_parquet(source_path)
        if len(frame) < requested:
            raise ValueError(
                f"{dataset} has {len(frame)} rows, fewer than requested {requested}"
            )
        if "question" not in frame.columns or frame["question"].astype(str).duplicated().any():
            raise ValueError(f"{dataset} must contain unique questions")

        rng = random.Random(seed + offset)
        selected_positions = sorted(rng.sample(range(len(frame)), requested))
        selected = frame.iloc[selected_positions].reset_index(drop=True)
        selected.to_parquet(output_dir / f"{dataset}.parquet", index=False)
        report["datasets"][dataset] = {
            "source_rows": len(frame),
            "selected": len(selected),
            "selected_positions": selected_positions,
            "question_sha256": question_digest(selected),
        }

    (output_dir / "manifest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = build(args.source_dir, args.output_dir, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
