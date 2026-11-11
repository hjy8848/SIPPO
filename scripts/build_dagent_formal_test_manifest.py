#!/usr/bin/env python3
"""Freeze an independent, reproducible four-benchmark DAGENT test manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


DATASET_REPO = "RUC-NLPIR/FlashRAG_datasets"
DATASET_SIZES = {
    "2wikimultihopqa": 500,
    "hotpotqa": 500,
    "musique": 500,
    "bamboogle": 125,
}
SPLIT_PRIORITY = ("test", "dev", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    return parser.parse_args()


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"\s+", " ", text)


def ensure_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def extract_question(row: dict[str, Any]) -> str:
    for container_key in ("extra_info", "reward_model"):
        container = row.get(container_key)
        if isinstance(container, dict):
            question = container.get("question")
            if question:
                return str(question).strip()
    if row.get("question"):
        return str(row["question"]).strip()
    prompt = row.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "").strip()
    return ""


def extract_answers(row: dict[str, Any]) -> list[str]:
    for key in ("golden_answers", "answer"):
        answers = ensure_answer_list(row.get(key))
        if answers:
            return answers
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            for key in ("target", "answer", "golden_answers"):
                answers = ensure_answer_list(ground_truth.get(key))
                if answers:
                    return answers
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        return ensure_answer_list(extra_info.get("answer"))
    return []


def load_question_set(path: str) -> set[str]:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=path, split="train")
    elif suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=path, split="train")
    else:
        raise ValueError(f"Unsupported local dataset format: {path}")
    questions = {
        normalize_question(extract_question(dict(row)))
        for row in dataset
    }
    questions.discard("")
    return questions


def choose_source_split(
    repo: str,
    config: str,
    excluded_questions: set[str],
    requested: int,
) -> tuple[Any, str, list[int], dict[str, int]]:
    errors: list[str] = []
    for split in SPLIT_PRIORITY:
        try:
            dataset = load_dataset(repo, config, split=split)
        except Exception as exc:  # pragma: no cover - depends on remote dataset
            errors.append(f"{split}: {exc}")
            continue

        eligible: list[int] = []
        overlap = 0
        missing_gold = 0
        duplicate = 0
        seen: set[str] = set()
        for index, item in enumerate(dataset):
            row = dict(item)
            question = normalize_question(extract_question(row))
            if not question:
                continue
            if not extract_answers(row):
                missing_gold += 1
                continue
            if question in excluded_questions:
                overlap += 1
                continue
            if question in seen:
                duplicate += 1
                continue
            seen.add(question)
            eligible.append(index)

        minimum_required = (
            min(requested, len(dataset)) if config == "bamboogle" else requested
        )
        if len(eligible) >= minimum_required:
            return dataset, split, eligible, {
                "excluded_overlap": overlap,
                "excluded_missing_gold": missing_gold,
                "excluded_duplicate": duplicate,
            }
        errors.append(
            f"{split}: eligible={len(eligible)} requested={requested} "
            f"overlap={overlap} missing_gold={missing_gold}"
        )
    raise RuntimeError(
        f"No usable independent split for {config}. " + " | ".join(errors)
    )


def dataset_digest(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(
            {
                "sample_index": row["sample_index"],
                "question": row["question"],
                "golden_answers": row["golden_answers"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "manifest_summary.json"
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        expected = set(DATASET_SIZES)
        if expected == set(existing.get("datasets", {})):
            complete = all(
                (output_dir / f"{name}.parquet").is_file() for name in expected
            )
            if complete:
                print(f"[SKIP] Complete frozen manifest already exists: {summary_path}")
                return

    train_questions = load_question_set(args.train_file)
    validation_questions = load_question_set(args.validation_file)
    excluded_questions = train_questions | validation_questions
    summary: dict[str, Any] = {
        "version": 1,
        "seed": args.seed,
        "source_repo": args.dataset_repo,
        "train_file": str(Path(args.train_file).resolve()),
        "validation_file": str(Path(args.validation_file).resolve()),
        "train_question_count": len(train_questions),
        "validation_question_count": len(validation_questions),
        "train_validation_overlap": len(train_questions & validation_questions),
        "datasets": {},
    }

    for offset, (config, requested) in enumerate(DATASET_SIZES.items()):
        dataset, split, eligible, exclusions = choose_source_split(
            args.dataset_repo,
            config,
            excluded_questions,
            requested,
        )
        if config == "bamboogle" and len(eligible) <= requested:
            selected_indices = eligible
        else:
            rng = random.Random(args.seed + offset)
            selected_indices = rng.sample(eligible, requested)

        rows: list[dict[str, Any]] = []
        for source_index in selected_indices:
            item = dict(dataset[source_index])
            question = extract_question(item)
            answers = extract_answers(item)
            rows.append(
                {
                    "sample_index": int(source_index),
                    "question": question,
                    "golden_answers": answers,
                    "source_dataset": config,
                    "source_split": split,
                }
            )

        normalized = [normalize_question(row["question"]) for row in rows]
        if len(normalized) != len(set(normalized)):
            raise RuntimeError(f"Duplicate questions remained in {config}")
        if any(question in excluded_questions for question in normalized):
            raise RuntimeError(f"Train/validation overlap remained in {config}")
        if any(not row["golden_answers"] for row in rows):
            raise RuntimeError(f"Missing gold answer remained in {config}")

        output_path = output_dir / f"{config}.parquet"
        Dataset.from_list(rows).to_parquet(str(output_path))
        summary["datasets"][config] = {
            "source_split": split,
            "source_fingerprint": getattr(dataset, "_fingerprint", ""),
            "source_size": len(dataset),
            "requested": requested,
            "selected": len(rows),
            "selected_indices": selected_indices,
            "sha256": dataset_digest(rows),
            **exclusions,
        }
        print(
            f"[DATASET] {config} split={split} selected={len(rows)} "
            f"overlap_excluded={exclusions['excluded_overlap']}"
        )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[DONE] Frozen independent test manifest: {summary_path}")


if __name__ == "__main__":
    main()
