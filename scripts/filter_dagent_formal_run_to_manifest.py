#!/usr/bin/env python3
"""Filter an existing complete DAGENT formal run to a frozen pilot manifest."""

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "bamboogle")


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return re.sub(r"\s+", " ", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def locate_result(source_run: Path, dataset: str) -> Path:
    matches = sorted((source_run / "trajectories").glob(f"full_logs_*_{dataset}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {dataset} result under {source_run}, found {matches}")
    return matches[0]


def filter_run(
    source_run: Path,
    manifest_dir: Path,
    output_dir: Path,
    run_name: str,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite filtered run: {output_dir}")
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True)

    report = {
        "schema_version": 1,
        "method": "filter_complete_formal_run_to_frozen_manifest",
        "source_run": str(source_run.resolve()),
        "manifest_dir": str(manifest_dir.resolve()),
        "run_name": run_name,
        "datasets": {},
    }
    for dataset in DATASETS:
        manifest = pd.read_parquet(manifest_dir / f"{dataset}.parquet")
        source_path = locate_result(source_run, dataset)
        rows = json.loads(source_path.read_text(encoding="utf-8"))
        by_question = {}
        for row in rows:
            question_key = normalize_text(row.get("question"))
            if not question_key or question_key in by_question:
                raise RuntimeError(f"Empty or duplicate source question/{dataset}")
            by_question[question_key] = row

        selected = []
        for pilot_index, record in enumerate(manifest.to_dict(orient="records")):
            question_key = normalize_text(record["question"])
            if question_key not in by_question:
                raise RuntimeError(f"Missing question={record['question']!r}/{dataset}")
            row = copy.deepcopy(by_question[question_key])
            row["source_sample_index"] = row.get("sample_index")
            row["sample_index"] = pilot_index
            selected.append(row)

        output_path = trajectory_dir / f"full_logs_{run_name}_{dataset}.json"
        output_path.write_text(
            json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report["datasets"][dataset] = {
            "source_rows": len(rows),
            "selected": len(selected),
            "sample_indices": [int(row["sample_index"]) for row in selected],
            "source_sample_indices": [row["source_sample_index"] for row in selected],
        }

    (output_dir / "filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = filter_run(
        args.source_run, args.manifest_dir, args.output_dir, args.run_name
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
