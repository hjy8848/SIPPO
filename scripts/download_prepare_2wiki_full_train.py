#!/usr/bin/env python3
"""Download the official full 2WikiMultiHopQA train split and convert it for ROSA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ARCHIVE_URL = (
    "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
)


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if not isinstance(value, dict):
        raise ValueError(f"expected dict, got {type(value).__name__}")
    return dict(value)


def evaluation_question_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"evaluation exclusion file not found: {path}")
        frame = pd.read_parquet(path, columns=["extra_info"])
        for item in frame["extra_info"]:
            question = as_dict(item).get("question", "")
            key = normalize_question(question)
            if not key:
                raise ValueError(f"empty question in evaluation file: {path}")
            keys.add(key)
    return keys


def download_file(url: str, target: Path, retries: int = 5) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", "0") or 0)
                written = 0
                next_report = 100 * 1024 * 1024
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if written >= next_report:
                            total_text = f"/{total / 2**20:.0f} MiB" if total else ""
                            print(f"[DOWNLOAD] {written / 2**20:.0f} MiB{total_text}")
                            next_report += 100 * 1024 * 1024
            partial.replace(target)
            return
        except requests.RequestException as exc:
            if attempt >= retries:
                raise RuntimeError(f"download failed after {retries + 1} attempts: {exc}") from exc
            delay = min(60, 2**attempt)
            print(f"[WARN] download attempt {attempt + 1} failed; retrying in {delay}s: {exc}")
            time.sleep(delay)


def extract_split_json(archive: Path, target: Path, filename: str) -> str:
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"downloaded file is not a valid zip archive: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        candidates = [
            info
            for info in bundle.infolist()
            if not info.is_dir() and Path(info.filename).name.lower() == filename.lower()
        ]
        if not candidates:
            raise ValueError(f"official archive contains no {filename}")
        candidates.sort(
            key=lambda info: (
                "data_ids" not in info.filename.lower(),
                len(Path(info.filename).parts),
                info.filename,
            )
        )
        selected = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with bundle.open(selected) as source, partial.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        partial.replace(target)
        return selected.filename


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> bool:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        refill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if not refill():
                raise ValueError("empty JSON file")
        if buffer[position] != "[":
            raise ValueError("official train.json must contain a top-level JSON array")
        position += 1

        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("unterminated JSON array")
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                refill()
                continue
            if not isinstance(value, dict):
                raise ValueError(f"train item must be an object, got {type(value).__name__}")
            yield value
            position = end
            if position > chunk_size:
                refill()


def answer_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        answers = [str(item).strip() for item in value if str(item).strip()]
    else:
        answer = str(value or "").strip()
        answers = [answer] if answer else []
    return answers


def convert_item(
    item: dict[str, Any],
    index: int,
    split: str = "train",
) -> tuple[dict[str, Any], str]:
    question = str(item.get("question", "")).strip()
    answers = answer_list(item.get("answer"))
    if not question or not answers:
        raise ValueError(f"row {index} has no question or answer")
    raw_id = str(item.get("_id", item.get("id", index)))
    row = {
        "data_source": "2wiki",
        "prompt": [{"role": "user", "content": question}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": answers,
                "answerable": True,
            },
            "question": question,
        },
        "extra_info": {
            "split": split,
            "index": int(index),
            "raw_id": raw_id,
            "question": question,
            "answer": answers,
            "source": "2wiki_official_data_ids_april7",
        },
    }
    return row, normalize_question(question)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=target.stem + ".",
        suffix=".parquet",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=OFFICIAL_ARCHIVE_URL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/2wikimultihopqa_official",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=PROJECT_ROOT / "data/multihop_qa/train_2wiki_official_full.parquet",
    )
    parser.add_argument(
        "--safe-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/multihop_qa/train_2wiki_official_full_no_eval_overlap.parquet"
        ),
    )
    parser.add_argument(
        "--dev-output",
        type=Path,
        default=PROJECT_ROOT / "data/multihop_qa/validation_2wiki_official_dev.parquet",
    )
    parser.add_argument(
        "--exclude-eval",
        type=Path,
        action="append",
        default=None,
        help="Parquet whose questions are removed from --safe-output; may be repeated.",
    )
    parser.add_argument("--min-train-rows", type=int, default=100_000)
    parser.add_argument("--min-dev-rows", type=int, default=1_000)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "train.json"
        sample = [
            {"_id": "a", "question": "Who is A?", "answer": "Alpha"},
            {"_id": "b", "question": "Where is B?", "answer": "Beta"},
        ]
        raw.write_text(json.dumps(sample), encoding="utf-8")
        archive = root / "sample.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("data_ids/train.json", json.dumps(sample))
            bundle.writestr("data_ids/dev.json", json.dumps(sample[:1]))
        extracted = root / "extracted_dev.json"
        assert extract_split_json(archive, extracted, "dev.json") == "data_ids/dev.json"
        assert json.loads(extracted.read_text(encoding="utf-8")) == sample[:1]
        loaded = list(iter_json_array(raw, chunk_size=7))
        assert loaded == sample
        row, key = convert_item(loaded[0], 0)
        assert row["extra_info"]["question"] == "Who is A?"
        assert row["reward_model"]["ground_truth"]["target"] == ["Alpha"]
        assert key == "who is a?"
        parquet = root / "sample.parquet"
        atomic_parquet(pd.DataFrame([row]), parquet)
        restored = pd.read_parquet(parquet).iloc[0]
        assert as_dict(restored["extra_info"])["raw_id"] == "a"
    print("[SELF-TEST] passed")


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.min_train_rows <= 0 or args.min_dev_rows <= 0:
        raise ValueError("--min-train-rows and --min-dev-rows must be positive")
    if args.full_output.resolve() == args.safe_output.resolve():
        raise ValueError("--full-output and --safe-output must differ")
    for output in (args.full_output, args.safe_output, args.dev_output):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite intentionally")

    exclusions = args.exclude_eval or [
        PROJECT_ROOT / "data/multihop_qa/validation_2wiki_heldout.parquet"
    ]
    excluded_keys = evaluation_question_keys(exclusions)
    archive = args.cache_dir / "data_ids_april7.zip"
    train_json = args.cache_dir / "train.json"
    dev_json = args.cache_dir / "dev.json"
    if args.force_download or not archive.exists():
        print(f"[DOWNLOAD] {args.url}")
        download_file(args.url, archive)
    else:
        print(f"[CACHE] archive={archive}")
    selected_train_member = extract_split_json(archive, train_json, "train.json")
    selected_dev_member = extract_split_json(archive, dev_json, "dev.json")
    print(f"[EXTRACT] {selected_train_member} -> {train_json}")
    print(f"[EXTRACT] {selected_dev_member} -> {dev_json}")

    rows: list[dict[str, Any]] = []
    keys: list[str] = []
    for index, item in enumerate(iter_json_array(train_json)):
        row, key = convert_item(item, index)
        rows.append(row)
        keys.append(key)
        if (index + 1) % 25_000 == 0:
            print(f"[CONVERT] rows={index + 1}")
    if len(rows) < args.min_train_rows:
        raise ValueError(
            f"official train unexpectedly contains only {len(rows)} rows; "
            f"expected at least {args.min_train_rows}"
        )

    dev_rows: list[dict[str, Any]] = []
    dev_keys: list[str] = []
    for index, item in enumerate(iter_json_array(dev_json)):
        row, key = convert_item(item, index, split="validation")
        dev_rows.append(row)
        dev_keys.append(key)
    if len(dev_rows) < args.min_dev_rows:
        raise ValueError(
            f"official dev unexpectedly contains only {len(dev_rows)} rows; "
            f"expected at least {args.min_dev_rows}"
        )

    frame = pd.DataFrame(rows)
    safe_mask = [key not in excluded_keys for key in keys]
    safe_frame = frame.loc[safe_mask].reset_index(drop=True)
    overlap_count = len(frame) - len(safe_frame)
    unique_count = len(set(keys))
    train_dev_overlap = len(set(keys) & set(dev_keys))
    if train_dev_overlap:
        raise ValueError(f"official train/dev overlap by {train_dev_overlap} normalized questions")
    atomic_parquet(frame, args.full_output)
    atomic_parquet(safe_frame, args.safe_output)
    atomic_parquet(pd.DataFrame(dev_rows), args.dev_output)

    report = {
        "official_source": args.url,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "archive_members": {
            "train": selected_train_member,
            "dev": selected_dev_member,
        },
        "full_train_rows": len(frame),
        "official_dev_rows": len(dev_rows),
        "official_train_dev_overlap": train_dev_overlap,
        "unique_normalized_questions": unique_count,
        "duplicate_normalized_questions": len(frame) - unique_count,
        "evaluation_exclusion_files": [str(path) for path in exclusions],
        "evaluation_question_count": len(excluded_keys),
        "full_train_evaluation_overlap": overlap_count,
        "safe_train_rows": len(safe_frame),
        "safe_train_evaluation_overlap": 0,
        "full_output": str(args.full_output),
        "safe_output": str(args.safe_output),
        "official_dev_output": str(args.dev_output),
    }
    report_path = args.safe_output.with_suffix(".download_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[DONE] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
