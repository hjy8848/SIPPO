#!/usr/bin/env python3
"""Build a disjoint old-data plus causal-data pure-EM RL mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

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


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def question_from_row(row: pd.Series) -> str:
    info = to_python(row.get("extra_info", {}))
    if isinstance(info, dict) and str(info.get("question", "")).strip():
        return str(info["question"]).strip()
    reward = to_python(row.get("reward_model", {}))
    if isinstance(reward, dict):
        if str(reward.get("question", "")).strip():
            return str(reward["question"]).strip()
        ground_truth = reward.get("ground_truth")
        if isinstance(ground_truth, dict) and str(ground_truth.get("question", "")).strip():
            return str(ground_truth["question"]).strip()
    prompt = to_python(row.get("prompt", []))
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", "")).strip()
    return ""


def load_rl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["_question_key"] = [
        normalize_question(question_from_row(row)) for _, row in result.iterrows()
    ]
    if any(not key for key in result["_question_key"]):
        raise ValueError(f"{path} contains a row without a usable question")
    if result["_question_key"].duplicated().any():
        raise ValueError(f"{path} contains duplicate normalized questions")
    return result


def load_sft_questions(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "question" not in frame:
        raise ValueError(f"{path} has no question column")
    return {normalize_question(value) for value in frame["question"] if normalize_question(value)}


def reason_id(row: pd.Series) -> str:
    info = to_python(row.get("extra_info", {}))
    return str(info.get("id", "")).strip() if isinstance(info, dict) else ""


def selected_sft_ids(path: Path) -> set[str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    values = report.get("selected_new_trajectory_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} has no selected_new_trajectory_ids")
    result = {str(value).strip() for value in values if str(value).strip()}
    if len(result) != len(values):
        raise ValueError(f"{path} contains duplicate or empty selected IDs")
    return result


def proportional_counts(frame: pd.DataFrame, total: int) -> dict[str, int]:
    if total <= 0 or total > len(frame):
        raise ValueError(f"invalid base sample size {total} for {len(frame)} rows")
    available = Counter(map(str, frame["data_source"]))
    exact = {source: total * count / len(frame) for source, count in available.items()}
    counts = {source: int(value) for source, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(exact, key=lambda source: (exact[source] - counts[source], source), reverse=True)
    for source in order[:remaining]:
        counts[source] += 1
    return dict(sorted(counts.items()))


def sample_base(frame: pd.DataFrame, total: int, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = proportional_counts(frame, total)
    samples = []
    for source, count in counts.items():
        pool = frame[frame["data_source"].astype(str) == source]
        if len(pool) < count:
            raise ValueError(f"not enough {source} rows: need {count}, found {len(pool)}")
        source_seed = int(hashlib.sha256(f"{seed}:{source}".encode()).hexdigest()[:16], 16)
        indices = list(pool.index)
        random.Random(source_seed).shuffle(indices)
        samples.append(frame.loc[indices[:count]].copy())
    return pd.concat(samples, ignore_index=True), counts


def annotate(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    values = []
    for value in result["extra_info"]:
        info = to_python(value)
        info = dict(info) if isinstance(info, dict) else {}
        info["dagent_unified_rl_source"] = source
        values.append(info)
    result["extra_info"] = values
    return result


def shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)
    return frame.iloc[indices].reset_index(drop=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_keys(keys: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(keys))).encode()).hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    base_train = load_rl(args.base_train)
    base_validation = load_rl(args.base_validation)
    reason_train = load_rl(args.reason_train)
    reason_validation = load_rl(args.reason_validation)
    sft_train_questions = load_sft_questions(args.sft_train)
    sft_validation_questions = load_sft_questions(args.sft_validation)
    sft_questions = sft_train_questions | sft_validation_questions
    selected_ids = selected_sft_ids(args.sft_report)

    reason_train = reason_train.copy()
    reason_train["_reason_id"] = [reason_id(row) for _, row in reason_train.iterrows()]
    if any(not value for value in reason_train["_reason_id"]):
        raise ValueError("reason train contains an empty trajectory ID")
    if reason_train["_reason_id"].duplicated().any():
        raise ValueError("reason train contains duplicate trajectory IDs")
    observed_selected = set(reason_train["_reason_id"]) & selected_ids
    if observed_selected != selected_ids:
        raise ValueError(
            f"SFT-selected reason IDs do not match reason train: "
            f"expected={len(selected_ids)}, observed={len(observed_selected)}"
        )
    reason_remaining = reason_train[~reason_train["_reason_id"].isin(selected_ids)].copy()
    if len(reason_remaining) != args.expected_reason_rows:
        raise ValueError(
            f"expected {args.expected_reason_rows} unused reason rows, found {len(reason_remaining)}"
        )

    sampled_base, base_counts = sample_base(base_train, args.base_rows, args.seed)
    train_overlap_sft = set(sampled_base["_question_key"]) & sft_questions
    reason_overlap_sft = set(reason_remaining["_question_key"]) & sft_questions
    if train_overlap_sft or reason_overlap_sft:
        raise ValueError(
            f"RL/SFT question overlap: base={len(train_overlap_sft)}, reason={len(reason_overlap_sft)}"
        )

    base_part = annotate(sampled_base, "base")
    reason_part = annotate(reason_remaining, "reason")
    train = shuffled(pd.concat([base_part, reason_part], ignore_index=True), args.seed)
    validation = shuffled(
        pd.concat(
            [annotate(base_validation, "base"), annotate(reason_validation, "reason")],
            ignore_index=True,
        ),
        args.seed + 1,
    )
    train_keys = set(train["_question_key"])
    validation_keys = set(validation["_question_key"])
    overlap = train_keys & validation_keys
    if overlap or len(train_keys) != len(train) or len(validation_keys) != len(validation):
        raise ValueError(
            f"train/validation invariant failed: overlap={len(overlap)}, "
            f"train_duplicates={len(train) - len(train_keys)}, "
            f"validation_duplicates={len(validation) - len(validation_keys)}"
        )

    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "train.parquet"
    validation_path = args.output_dir / "validation.parquet"
    hidden = ["_question_key", "_reason_id"]
    atomic_parquet(train.drop(columns=hidden, errors="ignore"), train_path)
    atomic_parquet(validation.drop(columns=hidden, errors="ignore"), validation_path)
    report = {
        "method": "dagent_unified_pure_em_rl_mix_v1",
        "seed": args.seed,
        "base_input_rows": len(base_train),
        "base_selected_rows": len(sampled_base),
        "base_selected_counts": base_counts,
        "reason_input_rows": len(reason_train),
        "reason_sft_excluded_rows": len(selected_ids),
        "reason_selected_rows": len(reason_remaining),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_source_counts": dict(sorted(Counter(
            to_python(value).get("dagent_unified_rl_source") for value in train["extra_info"]
        ).items())),
        "validation_source_counts": dict(sorted(Counter(
            to_python(value).get("dagent_unified_rl_source") for value in validation["extra_info"]
        ).items())),
        "hard_gates": {
            "base_count_exact": len(sampled_base) == args.base_rows,
            "reason_count_exact": len(reason_remaining) == args.expected_reason_rows,
            "sft_selected_ids_excluded": not bool(set(reason_remaining["_reason_id"]) & selected_ids),
            "sft_rl_question_overlap_zero": not bool(train_keys & sft_questions),
            "train_validation_question_overlap_zero": not bool(overlap),
            "train_questions_unique": len(train_keys) == len(train),
            "validation_questions_unique": len(validation_keys) == len(validation),
        },
        "question_set_sha256": {
            "train": hash_keys(train_keys),
            "validation": hash_keys(validation_keys),
        },
        "inputs": {
            "base_train": str(args.base_train),
            "base_validation": str(args.base_validation),
            "reason_train": str(args.reason_train),
            "reason_validation": str(args.reason_validation),
            "sft_train": str(args.sft_train),
            "sft_validation": str(args.sft_validation),
            "sft_report": str(args.sft_report),
        },
        "files": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "validation": {"path": str(validation_path), "sha256": sha256(validation_path)},
        },
    }
    if not all(report["hard_gates"].values()):
        raise AssertionError(f"hard gates failed: {report['hard_gates']}")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--reason-train", type=Path, required=True)
    parser.add_argument("--reason-validation", type=Path, required=True)
    parser.add_argument("--sft-train", type=Path, required=True)
    parser.add_argument("--sft-validation", type=Path, required=True)
    parser.add_argument("--sft-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-rows", type=int, default=1500)
    parser.add_argument("--expected-reason-rows", type=int, default=578)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
