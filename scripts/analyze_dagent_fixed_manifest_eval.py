#!/usr/bin/env python3
"""Audit and summarize deterministic DAGENT evaluations on frozen manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_dataset_spec(value: str) -> tuple[str, int]:
    name, separator, count = value.rpartition("=")
    if not separator or not name or not count.isdigit():
        raise argparse.ArgumentTypeError(f"Expected DATASET=COUNT, got {value!r}")
    return name, int(count)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for dataset, expected_count in args.dataset:
        result_path = (
            args.trajectory_root / f"full_logs_{args.run_name}_{dataset}.json"
        )
        rows = load_rows(result_path)
        manifest = pq.read_table(
            args.manifest_root / f"{dataset}.parquet",
            columns=["question"],
            use_threads=False,
        )
        if len(rows) != expected_count or manifest.num_rows != expected_count:
            raise RuntimeError(
                f"{dataset}: expected {expected_count}, got results={len(rows)} "
                f"manifest={manifest.num_rows}"
            )
        expected_questions = Counter(
            str(value).strip() for value in manifest.column("question").to_pylist()
        )
        actual_questions = Counter(str(row.get("question") or "").strip() for row in rows)
        if actual_questions != expected_questions:
            raise RuntimeError(f"{dataset}: manifest/result question mismatch")
        if any(
            row.get(field) not in (None, "", False)
            for row in rows
            for field in ("error", "pipeline_error")
        ):
            raise RuntimeError(f"{dataset}: API or pipeline error")
        if any(
            not math.isfinite(float(row[metric]))
            for row in rows
            for metric in ("em", "f1")
        ):
            raise RuntimeError(f"{dataset}: non-finite metric")

        query_rewrites = sum(
            (event or {}).get("agent_name") == "QueryRewriteAgent"
            for row in rows
            for event in (row.get("raw_trace") or {}).get("sft_events", [])
        )
        execution_errors = sum(
            bool((row.get("raw_trace") or {}).get("execution_error")) for row in rows
        )
        results[dataset] = {
            "samples": len(rows),
            "em": sum(float(row["em"]) for row in rows) / len(rows),
            "f1": sum(float(row["f1"]) for row in rows) / len(rows),
            "api_or_pipeline_errors": 0,
            "execution_error_rows": execution_errors,
            "query_rewrite_events": query_rewrites,
            "retrieval_calls_mean": sum(
                float(row.get("retrieval_calls") or 0) for row in rows
            )
            / len(rows),
        }
        all_rows.extend(rows)

    report = {
        "schema": "dagent-fixed-manifest-eval-v1",
        "run_name": args.run_name,
        "datasets": results,
        "macro_em": sum(item["em"] for item in results.values()) / len(results),
        "macro_f1": sum(item["f1"] for item in results.values()) / len(results),
        "micro_em": sum(float(row["em"]) for row in all_rows) / len(all_rows),
        "micro_f1": sum(float(row["f1"]) for row in all_rows) / len(all_rows),
        "total_samples": len(all_rows),
        "api_or_pipeline_errors": 0,
        "execution_error_rows": sum(
            item["execution_error_rows"] for item in results.values()
        ),
        "query_rewrite_events": sum(
            item["query_rewrite_events"] for item in results.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
