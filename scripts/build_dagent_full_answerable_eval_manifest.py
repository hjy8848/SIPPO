#!/usr/bin/env python3
"""Freeze the complete answerable FlashRAG evaluation splits for DAGENT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset


DATASET_REPO = "RUC-NLPIR/FlashRAG_datasets"
DATASET_CONTRACT = {
    "2wikimultihopqa": {
        "split": "dev",
        "fingerprint": "4aa704e6c97caf2c",
        "source_rows": 12_576,
        "selected_rows": 12_576,
    },
    "hotpotqa": {
        "split": "dev",
        "fingerprint": "6946368cec8dd6ee",
        "source_rows": 7_405,
        "selected_rows": 7_405,
    },
    "musique": {
        "split": "dev",
        "fingerprint": "2c7d63f740151163",
        "source_rows": 2_417,
        "selected_rows": 2_411,
    },
    "bamboogle": {
        "split": "test",
        "fingerprint": "49e4b626d52551a6",
        "source_rows": 125,
        "selected_rows": 125,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    return parser.parse_args()


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
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
        if isinstance(container, dict) and container.get("question"):
            return str(container["question"]).strip()
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
    for key in ("golden_answers", "gold_answer", "answer"):
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


def load_local_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(path), split="train")
    elif suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=str(path), split="train")
    else:
        raise ValueError(f"Unsupported local dataset format: {path}")
    return (dict(row) for row in dataset)


def load_question_set(path: Path) -> set[str]:
    questions = {normalize_question(extract_question(row)) for row in load_local_rows(path)}
    questions.discard("")
    return questions


def canonical_row_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {
            "sample_index": row["sample_index"],
            "question": row["question"],
            "golden_answers": row["golden_answers"],
            "source_dataset": row["source_dataset"],
            "source_split": row["source_split"],
        }
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    *,
    repo: str,
    name: str,
    contract: dict[str, Any],
    excluded_questions: set[str],
    output_dir: Path,
) -> dict[str, Any]:
    source = load_dataset(repo, name, split=contract["split"])
    observed_fingerprint = str(getattr(source, "_fingerprint", ""))
    if observed_fingerprint != contract["fingerprint"]:
        raise RuntimeError(
            f"{name} fingerprint changed: {observed_fingerprint} != {contract['fingerprint']}"
        )
    if len(source) != contract["source_rows"]:
        raise RuntimeError(f"{name} source size changed: {len(source)} != {contract['source_rows']}")

    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    duplicate_source_indices: list[int] = []
    overlap_source_indices: list[int] = []
    missing_gold_source_indices: list[int] = []
    missing_question_source_indices: list[int] = []
    for source_index, item in enumerate(source):
        raw = dict(item)
        question = extract_question(raw)
        normalized = normalize_question(question)
        answers = extract_answers(raw)
        if not normalized:
            missing_question_source_indices.append(source_index)
            continue
        if not answers:
            missing_gold_source_indices.append(source_index)
            continue
        if normalized in excluded_questions:
            overlap_source_indices.append(source_index)
            continue
        if normalized in seen:
            duplicate_source_indices.append(source_index)
            continue
        seen[normalized] = source_index
        rows.append(
            {
                "sample_index": source_index,
                "question": question,
                "golden_answers": answers,
                "source_dataset": name,
                "source_split": contract["split"],
            }
        )

    if overlap_source_indices:
        raise RuntimeError(
            f"{name} unexpectedly overlaps the frozen RL train/validation questions: "
            f"{overlap_source_indices[:10]}"
        )
    if missing_question_source_indices or missing_gold_source_indices:
        raise RuntimeError(
            f"{name} contains unscorable rows: missing_question={missing_question_source_indices[:10]} "
            f"missing_gold={missing_gold_source_indices[:10]}"
        )
    if len(rows) != contract["selected_rows"]:
        raise RuntimeError(
            f"{name} selected size changed: {len(rows)} != {contract['selected_rows']} "
            f"duplicates={len(duplicate_source_indices)}"
        )
    expected_duplicates = contract["source_rows"] - contract["selected_rows"]
    if len(duplicate_source_indices) != expected_duplicates:
        raise RuntimeError(
            f"{name} duplicate count changed: {len(duplicate_source_indices)} != {expected_duplicates}"
        )

    output_path = output_dir / f"{name}.parquet"
    Dataset.from_list(rows).to_parquet(str(output_path))
    return {
        "source_split": contract["split"],
        "source_fingerprint": observed_fingerprint,
        "source_rows": len(source),
        "selected_rows": len(rows),
        "selected_source_indices": [row["sample_index"] for row in rows],
        "duplicate_source_indices": duplicate_source_indices,
        "excluded_overlap_source_indices": overlap_source_indices,
        "missing_question_source_indices": missing_question_source_indices,
        "missing_gold_source_indices": missing_gold_source_indices,
        "semantic_sha256": canonical_row_digest(rows),
        "parquet_sha256": file_sha256(output_path),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite manifest evidence: {args.output_dir}")
    for path in (args.train_file, args.validation_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_questions = load_question_set(args.train_file)
    validation_questions = load_question_set(args.validation_file)
    excluded_questions = train_questions | validation_questions
    args.output_dir.mkdir(parents=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "DAGENT complete answerable FlashRAG benchmark evaluation",
        "source_repo": args.dataset_repo,
        "train_file": str(args.train_file.resolve()),
        "validation_file": str(args.validation_file.resolve()),
        "train_file_sha256": file_sha256(args.train_file),
        "validation_file_sha256": file_sha256(args.validation_file),
        "train_question_count": len(train_questions),
        "validation_question_count": len(validation_questions),
        "train_validation_overlap": len(train_questions & validation_questions),
        "normalization": "Unicode NFKC, strip, casefold, collapse whitespace",
        "deduplication": "keep the first source row for each normalized question",
        "datasets": {},
    }
    for name, contract in DATASET_CONTRACT.items():
        result = build_dataset(
            repo=args.dataset_repo,
            name=name,
            contract=contract,
            excluded_questions=excluded_questions,
            output_dir=args.output_dir,
        )
        summary["datasets"][name] = result
        print(
            f"[DATASET] {name} source={result['source_rows']} selected={result['selected_rows']} "
            f"duplicates={len(result['duplicate_source_indices'])}"
        )
    summary["total_selected_rows"] = sum(
        item["selected_rows"] for item in summary["datasets"].values()
    )
    if summary["total_selected_rows"] != 22_517:
        raise RuntimeError(f"Unexpected total row count: {summary['total_selected_rows']}")
    summary_path = args.output_dir / "manifest_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] Complete frozen manifest: {summary_path}")


if __name__ == "__main__":
    main()
