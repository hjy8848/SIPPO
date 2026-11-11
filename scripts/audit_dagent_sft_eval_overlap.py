#!/usr/bin/env python3
"""Audit exact question overlap between a DAGENT SFT mix and formal-v2."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


FORMAL_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "bamboogle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--formal-manifest-dir", type=Path, required=True)
    parser.add_argument("--source-column", default="unified_source")
    parser.add_argument("--causal-label", default="new")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Report overlap without failing the audit.",
    )
    return parser.parse_args()


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


def question_map(frame: pd.DataFrame) -> dict[str, str]:
    if "question" not in frame:
        raise ValueError("input frame has no question column")
    return {
        normalize_question(question): str(question)
        for question in frame["question"].dropna()
        if normalize_question(question)
    }


def audit(
    train_path: Path,
    formal_dir: Path,
    source_column: str,
    causal_label: str,
) -> dict[str, Any]:
    train = pd.read_parquet(train_path)
    if source_column not in train:
        raise ValueError(f"training frame has no {source_column!r} column")
    causal = train[train[source_column].astype(str) == causal_label]
    if causal.empty:
        raise ValueError(f"no rows have {source_column}={causal_label!r}")

    train_questions = question_map(train)
    causal_questions = question_map(causal)
    datasets: dict[str, Any] = {}
    any_overlap = False
    for name in FORMAL_DATASETS:
        manifest_path = formal_dir / f"{name}.parquet"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing formal manifest: {manifest_path}")
        formal_questions = question_map(pd.read_parquet(manifest_path))
        train_overlap = sorted(set(train_questions) & set(formal_questions))
        causal_overlap = sorted(set(causal_questions) & set(formal_questions))
        any_overlap = any_overlap or bool(train_overlap)
        datasets[name] = {
            "formal_questions": len(formal_questions),
            "all_train_exact_overlap": len(train_overlap),
            "causal_exact_overlap": len(causal_overlap),
            "all_train_overlap_examples": [train_questions[key] for key in train_overlap[:10]],
            "causal_overlap_examples": [causal_questions[key] for key in causal_overlap[:10]],
        }

    return {
        "schema_version": 1,
        "normalization": "NFKC + whitespace collapse + casefold",
        "train_parquet": str(train_path),
        "formal_manifest_dir": str(formal_dir),
        "source_column": source_column,
        "causal_label": causal_label,
        "train_rows": len(train),
        "train_unique_questions": len(train_questions),
        "causal_rows": len(causal),
        "causal_unique_questions": len(causal_questions),
        "datasets": datasets,
        "hard_gates": {
            "all_train_formal_exact_overlap_zero": not any_overlap,
            "causal_formal_exact_overlap_zero": all(
                row["causal_exact_overlap"] == 0 for row in datasets.values()
            ),
        },
    }


def main() -> None:
    args = parse_args()
    report = audit(
        args.train_parquet,
        args.formal_manifest_dir,
        args.source_column,
        args.causal_label,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not args.allow_overlap and not all(report["hard_gates"].values()):
        raise SystemExit("SFT/formal-v2 question-overlap hard gate failed")


if __name__ == "__main__":
    main()
