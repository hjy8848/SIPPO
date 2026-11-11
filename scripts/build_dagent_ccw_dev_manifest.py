#!/usr/bin/env python3
"""Build deterministic DAG-CCW development manifests from RL validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_TO_DATASET = {
    "2wiki": "2wiki",
    "hotpotqa": "hotpotqa",
    "musique": "musique",
}


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_question(row: pd.Series) -> str:
    extra_info = as_mapping(row.get("extra_info"))
    question = extra_info.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()

    reward_model = as_mapping(row.get("reward_model"))
    ground_truth = as_mapping(reward_model.get("ground_truth"))
    question = ground_truth.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()

    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                question = str(message.get("content", "")).strip()
                if question:
                    return question
    return ""


def question_digest(frame: pd.DataFrame) -> str:
    questions = sorted(extract_question(row) for _, row in frame.iterrows())
    return stable_digest("\n".join(questions))


def load_excluded_questions(exclude_dir: Path | None) -> dict[str, set[str]]:
    excluded = {dataset: set() for dataset in SOURCE_TO_DATASET.values()}
    if exclude_dir is None:
        return excluded
    for dataset in excluded:
        path = exclude_dir / f"{dataset}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing exclusion manifest: {path}")
        frame = pd.read_parquet(path)
        excluded[dataset] = {extract_question(row) for _, row in frame.iterrows()}
        if "" in excluded[dataset]:
            raise ValueError(f"{dataset} exclusion manifest contains an empty question")
    return excluded


def build(
    source: Path,
    output_dir: Path,
    count: int,
    seed: int,
    exclude_dir: Path | None = None,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite DAG-CCW manifest: {output_dir}")

    frame = pd.read_parquet(source)
    if "data_source" not in frame.columns:
        raise ValueError("source must contain data_source")

    excluded_questions = load_excluded_questions(exclude_dir)
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "stable_hash_subset_of_unified_em_rl_validation",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": seed,
        "requested_per_dataset": count,
        "exclude_dir": str(exclude_dir.resolve()) if exclude_dir else None,
        "excluded_sources": sorted(
            set(frame["data_source"].astype(str)) - set(SOURCE_TO_DATASET)
        ),
        "datasets": {},
    }

    all_questions: set[str] = set()
    for source_name, dataset_name in SOURCE_TO_DATASET.items():
        subset = frame[frame["data_source"].astype(str) == source_name].copy()
        subset["_ccw_question"] = [extract_question(row) for _, row in subset.iterrows()]
        if (subset["_ccw_question"] == "").any():
            raise ValueError(f"{source_name} contains an empty question")
        if subset["_ccw_question"].duplicated().any():
            raise ValueError(f"{source_name} contains duplicate questions")
        source_rows = len(subset)
        excluded = excluded_questions[dataset_name]
        subset = subset[~subset["_ccw_question"].isin(excluded)].copy()
        if len(subset) < count:
            raise ValueError(
                f"{source_name} has {len(subset)} rows, fewer than requested {count}"
            )

        subset["_ccw_key"] = [
            stable_digest(f"{seed}:{source_name}:{question}")
            for question in subset["_ccw_question"]
        ]
        selected = (
            subset.sort_values(["_ccw_key", "_ccw_question"], kind="stable")
            .head(count)
            .drop(columns=["_ccw_key", "_ccw_question"])
            .reset_index(drop=True)
        )
        selected_questions = {extract_question(row) for _, row in selected.iterrows()}
        overlap = all_questions & selected_questions
        if overlap:
            raise ValueError(f"cross-dataset question overlap: {sorted(overlap)[:3]}")
        all_questions.update(selected_questions)

        output_path = output_dir / f"{dataset_name}.parquet"
        selected.to_parquet(output_path, index=False)
        report["datasets"][dataset_name] = {
            "data_source": source_name,
            "source_rows": source_rows,
            "excluded_question_rows": source_rows - len(subset),
            "eligible_rows": len(subset),
            "selected_rows": len(selected),
            "question_sha256": question_digest(selected),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }

    (output_dir / "manifest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--exclude-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build(
        args.source,
        args.output_dir,
        args.count,
        args.seed,
        exclude_dir=args.exclude_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
