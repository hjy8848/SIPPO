#!/usr/bin/env python3
"""Align an existing DAGENT JSON run to a frozen parquet question subset."""

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return re.sub(r"\s+", " ", text)


def record_question(record: dict[str, Any]) -> str:
    if str(record.get("question", "")).strip():
        return str(record["question"]).strip()
    extra = record.get("extra_info")
    if isinstance(extra, dict) and str(extra.get("question", "")).strip():
        return str(extra["question"]).strip()
    prompt = record.get("prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and str(first.get("content", "")).strip():
            return str(first["content"]).strip()
    raise ValueError("Dataset record has no supported question field")


def filter_run(source_json: Path, dataset: Path, output_json: Path) -> dict[str, Any]:
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_json}")
    source_rows = json.loads(source_json.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError("Source run must be a JSON list")
    by_question: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        key = normalize_text(row.get("question"))
        if not key or key in by_question:
            raise ValueError("Source run has an empty or duplicate question")
        by_question[key] = row

    frame = pd.read_parquet(dataset)
    questions = [record_question(record) for record in frame.to_dict(orient="records")]
    if len({normalize_text(question) for question in questions}) != len(questions):
        raise ValueError("Dataset has duplicate questions")

    selected = []
    for index, question in enumerate(questions):
        key = normalize_text(question)
        if key not in by_question:
            raise ValueError(f"Source run is missing question: {question!r}")
        row = copy.deepcopy(by_question[key])
        row["source_sample_index"] = row.get("sample_index")
        row["sample_index"] = index
        selected.append(row)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "source_json": str(source_json.resolve()),
        "dataset": str(dataset.resolve()),
        "output_json": str(output_json.resolve()),
        "source_rows": len(source_rows),
        "selected_rows": len(selected),
    }
    report_path = output_json.with_suffix(".filter_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def filter_metadata(
    source_jsonl: Path, dataset: Path, output_jsonl: Path
) -> dict[str, Any]:
    if output_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_jsonl}")
    source_rows = [
        json.loads(line)
        for line in source_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_question: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        key = normalize_text(row.get("question"))
        if not key or key in by_question:
            raise ValueError("Metadata has an empty or duplicate question")
        by_question[key] = row
    frame = pd.read_parquet(dataset)
    questions = [record_question(record) for record in frame.to_dict(orient="records")]
    selected = []
    for question in questions:
        key = normalize_text(question)
        if key not in by_question:
            raise ValueError(f"Metadata is missing question: {question!r}")
        selected.append(by_question[key])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "source_jsonl": str(source_jsonl.resolve()),
        "dataset": str(dataset.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "source_rows": len(source_rows),
        "selected_rows": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()
    run_mode = args.source_json is not None or args.output_json is not None
    metadata_mode = args.source_jsonl is not None or args.output_jsonl is not None
    if run_mode == metadata_mode:
        parser.error("select exactly one source/output pair: JSON run or JSONL metadata")
    if run_mode and (args.source_json is None or args.output_json is None):
        parser.error("--source-json and --output-json must be used together")
    if metadata_mode and (args.source_jsonl is None or args.output_jsonl is None):
        parser.error("--source-jsonl and --output-jsonl must be used together")
    report = (
        filter_run(args.source_json, args.dataset, args.output_json)
        if run_mode
        else filter_metadata(args.source_jsonl, args.dataset, args.output_jsonl)
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
