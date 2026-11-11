#!/usr/bin/env python3
"""Create deterministic, leakage-audited mixed RL data for DAGENT."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "multihop_qa"
SOURCES = ("2wiki", "hotpotqa", "musique")


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


def extra_info(value: Any) -> dict[str, Any]:
    result = to_python(value)
    if not isinstance(result, dict):
        raise ValueError(f"extra_info must be a dict, got {type(result).__name__}")
    return result


def question_from_row(row: pd.Series) -> str:
    if "question" in row and str(row.get("question", "")).strip():
        return str(row["question"]).strip()
    info = extra_info(row.get("extra_info", {}))
    question = str(info.get("question", "")).strip()
    if question:
        return question
    reward_model = to_python(row.get("reward_model", {}))
    if isinstance(reward_model, dict):
        return str(reward_model.get("question", "")).strip()
    return ""


def question_keys(frame: pd.DataFrame) -> list[str]:
    keys = [normalize_question(question_from_row(row)) for _, row in frame.iterrows()]
    if any(not key for key in keys):
        bad = next(index for index, key in enumerate(keys) if not key)
        raise ValueError(f"row {bad} has no usable question")
    return keys


def source_name(value: Any) -> str:
    normalized = str(value or "").strip().casefold().replace("-", "").replace("_", "")
    aliases = {
        "2wiki": "2wiki",
        "2wikimultihopqa": "2wiki",
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "musique": "musique",
    }
    return aliases.get(normalized, str(value or "").strip().casefold())


def validate_schema(frame: pd.DataFrame, path: Path) -> None:
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_pool(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_schema(frame, path)
    frame = frame.copy()
    frame["data_source"] = frame["data_source"].map(source_name)
    frame["_question_key"] = question_keys(frame)
    before = len(frame)
    frame = frame.drop_duplicates("_question_key", keep="first").reset_index(drop=True)
    if len(frame) != before:
        print(f"[WARN] {path}: removed {before - len(frame)} duplicate normalized questions")
    return frame


def keys_from_exclusion(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    keys: set[str] = set()
    if path.suffix.casefold() == ".parquet":
        frame = pd.read_parquet(path)
        for _, row in frame.iterrows():
            key = normalize_question(question_from_row(row))
            if key:
                keys.add(key)
        return keys
    if path.suffix.casefold() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.casefold() == ".json":
                records: Iterable[Any] = json.load(handle)
            else:
                records = (json.loads(line) for line in handle if line.strip())
            for record in records:
                if not isinstance(record, dict):
                    continue
                question = record.get("question")
                if not question and isinstance(record.get("extra_info"), dict):
                    question = record["extra_info"].get("question")
                key = normalize_question(question)
                if key:
                    keys.add(key)
        return keys
    raise ValueError(f"unsupported exclusion format: {path}")


def target_counts(total_size: int, ratios: dict[str, float]) -> dict[str, int]:
    if total_size <= 0:
        raise ValueError("--total-size must be positive")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("all mixture ratios must be positive")
    ratio_sum = sum(ratios.values())
    exact = {source: total_size * value / ratio_sum for source, value in ratios.items()}
    counts = {source: int(value) for source, value in exact.items()}
    remaining = total_size - sum(counts.values())
    order = sorted(SOURCES, key=lambda source: (exact[source] - counts[source], source), reverse=True)
    for source in order[:remaining]:
        counts[source] += 1
    return counts


def deterministic_sample(frame: pd.DataFrame, count: int, seed: int, source: str) -> pd.DataFrame:
    if len(frame) < count:
        raise ValueError(
            f"not enough {source} rows after exclusions: need {count}, found {len(frame)}; "
            "reduce --total-size instead of silently oversampling"
        )
    indices = list(range(len(frame)))
    source_seed = int(hashlib.sha256(f"{seed}:{source}".encode()).hexdigest()[:16], 16)
    random.Random(source_seed).shuffle(indices)
    return frame.iloc[indices[:count]].copy().reset_index(drop=True)


def annotate(frame: pd.DataFrame, partition: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    updated = []
    for value in result["extra_info"]:
        info = extra_info(value)
        info["split"] = partition
        info["dagent_partition"] = partition
        info["dagent_mixture_seed"] = int(seed)
        updated.append(info)
    result["extra_info"] = updated
    return result


def shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)
    return frame.iloc[indices].reset_index(drop=True)


def hash_keys(keys: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(keys))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_validation(
    two_wiki_path: Path,
    mixed_path: Path,
    per_dataset: int,
    seed: int,
    excluded_keys: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    two_wiki = load_pool(two_wiki_path)
    mixed = load_pool(mixed_path)
    pools = {
        "2wiki": two_wiki[two_wiki["data_source"] == "2wiki"].reset_index(drop=True),
        "hotpotqa": mixed[mixed["data_source"] == "hotpotqa"].reset_index(drop=True),
        "musique": mixed[mixed["data_source"] == "musique"].reset_index(drop=True),
    }
    pools = {
        source: frame[~frame["_question_key"].isin(excluded_keys)].reset_index(drop=True)
        for source, frame in pools.items()
    }
    samples = [deterministic_sample(pools[source], per_dataset, seed + 1, source) for source in SOURCES]
    validation = shuffled(annotate(pd.concat(samples, ignore_index=True), "validation", seed), seed + 1)
    return validation, {source: len(pools[source]) for source in SOURCES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--two-wiki-train",
        type=Path,
        default=DATA_ROOT / "rl_train_2wiki_no_sft_seed42.parquet",
    )
    parser.add_argument(
        "--mixed-train-pool",
        type=Path,
        default=DATA_ROOT / "train_mixed_no_test_overlap.parquet",
    )
    parser.add_argument(
        "--validation-2wiki",
        type=Path,
        default=DATA_ROOT / "validation_2wiki_official_dev.parquet",
    )
    parser.add_argument(
        "--validation-mixed",
        type=Path,
        default=DATA_ROOT / "validation_heldout.parquet",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=DATA_ROOT / "rl_train_mixed_2wiki40_hotpot30_musique30_seed42.parquet",
    )
    parser.add_argument(
        "--output-validation",
        type=Path,
        default=DATA_ROOT / "validation_mixed_2wiki_hotpot_musique_seed42.parquet",
    )
    parser.add_argument("--exclude-file", type=Path, action="append", default=[])
    parser.add_argument("--total-size", type=int, default=15_000)
    parser.add_argument("--ratio-2wiki", type=float, default=0.40)
    parser.add_argument("--ratio-hotpotqa", type=float, default=0.30)
    parser.add_argument("--ratio-musique", type=float, default=0.30)
    parser.add_argument("--validation-per-dataset", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = [args.output_train]
    if not args.skip_validation:
        outputs.append(args.output_validation)
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {path}; use --overwrite intentionally")

    explicit_exclusions: set[str] = set()
    exclusion_counts: dict[str, int] = {}
    for path in args.exclude_file:
        keys = keys_from_exclusion(path)
        explicit_exclusions.update(keys)
        exclusion_counts[str(path)] = len(keys)

    validation = None
    validation_pool_counts: dict[str, int] = {}
    validation_keys: set[str] = set()
    if not args.skip_validation:
        validation, validation_pool_counts = build_validation(
            args.validation_2wiki,
            args.validation_mixed,
            args.validation_per_dataset,
            args.seed,
            explicit_exclusions,
        )
        validation_keys = set(validation["_question_key"])
    exclusions = explicit_exclusions | validation_keys

    two_wiki = load_pool(args.two_wiki_train)
    mixed = load_pool(args.mixed_train_pool)
    pools = {
        "2wiki": two_wiki[two_wiki["data_source"] == "2wiki"].reset_index(drop=True),
        "hotpotqa": mixed[mixed["data_source"] == "hotpotqa"].reset_index(drop=True),
        "musique": mixed[mixed["data_source"] == "musique"].reset_index(drop=True),
    }
    before_exclusion = {source: len(frame) for source, frame in pools.items()}
    pools = {
        source: frame[~frame["_question_key"].isin(exclusions)].reset_index(drop=True)
        for source, frame in pools.items()
    }
    after_exclusion = {source: len(frame) for source, frame in pools.items()}

    ratios = {
        "2wiki": args.ratio_2wiki,
        "hotpotqa": args.ratio_hotpotqa,
        "musique": args.ratio_musique,
    }
    counts = target_counts(args.total_size, ratios)
    samples = [deterministic_sample(pools[source], counts[source], args.seed, source) for source in SOURCES]
    train = shuffled(annotate(pd.concat(samples, ignore_index=True), "train", args.seed), args.seed)

    train_keys = set(train["_question_key"])
    duplicate_count = len(train) - len(train_keys)
    train_validation_overlap = len(train_keys & validation_keys)
    train_exclusion_overlap = len(train_keys & explicit_exclusions)
    validation_exclusion_overlap = len(validation_keys & explicit_exclusions)
    if duplicate_count or train_validation_overlap or train_exclusion_overlap or validation_exclusion_overlap:
        raise AssertionError(
            "split invariant failed: "
            f"duplicates={duplicate_count}, train_validation={train_validation_overlap}, "
            f"train_exclusions={train_exclusion_overlap}, "
            f"validation_exclusions={validation_exclusion_overlap}"
        )

    train_output = train.drop(columns=["_question_key"])
    atomic_parquet(train_output, args.output_train)
    if validation is not None:
        atomic_parquet(validation.drop(columns=["_question_key"]), args.output_validation)

    report = {
        "seed": args.seed,
        "train_output": str(args.output_train),
        "validation_output": None if args.skip_validation else str(args.output_validation),
        "train_rows": len(train),
        "target_counts": counts,
        "actual_counts": train["data_source"].value_counts().to_dict(),
        "actual_ratios": {
            source: float((train["data_source"] == source).mean()) for source in SOURCES
        },
        "pool_rows_before_exclusion": before_exclusion,
        "pool_rows_after_exclusion": after_exclusion,
        "explicit_exclusion_counts": exclusion_counts,
        "validation_rows": 0 if validation is None else len(validation),
        "validation_counts": {} if validation is None else validation["data_source"].value_counts().to_dict(),
        "validation_pool_rows": validation_pool_counts,
        "overlaps": {
            "duplicate_train_questions": duplicate_count,
            "train_vs_validation": train_validation_overlap,
            "train_vs_explicit_exclusions": train_exclusion_overlap,
            "validation_vs_explicit_exclusions": validation_exclusion_overlap,
        },
        "question_set_sha256": {
            "train": hash_keys(train_keys),
            "validation": None if validation is None else hash_keys(validation_keys),
        },
        "inputs": {
            "two_wiki_train": str(args.two_wiki_train),
            "mixed_train_pool": str(args.mixed_train_pool),
            "validation_2wiki": None if args.skip_validation else str(args.validation_2wiki),
            "validation_mixed": None if args.skip_validation else str(args.validation_mixed),
        },
    }
    report_path = args.output_train.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] train={args.output_train}")
    if validation is not None:
        print(f"[DONE] validation={args.output_validation}")
    print(f"[DONE] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
