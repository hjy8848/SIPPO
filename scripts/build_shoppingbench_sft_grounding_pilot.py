#!/usr/bin/env python3
"""Build a fixed validation-only pilot for ShoppingBench grounded SFT collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


TASKS = ("product", "shop", "voucher")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(input_path: Path, output_dir: Path, per_task: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    if per_task <= 0:
        raise ValueError("per_task must be positive")
    rows = load_jsonl(input_path)
    by_task = {task: [] for task in TASKS}
    seen_ids: set[str] = set()
    for row in rows:
        spec_id = str(row.get("spec_id") or "")
        task = str(row.get("task") or "")
        if row.get("split") != "val" or task not in by_task:
            raise ValueError(f"{spec_id}: pilot source must contain only valid val rows")
        if not spec_id or spec_id in seen_ids:
            raise ValueError(f"missing or duplicate spec_id: {spec_id!r}")
        seen_ids.add(spec_id)
        if row.get("planner_input_fields") != ["query"] or row.get("gold_runtime_access") is not False:
            raise ValueError(f"{spec_id}: runtime leakage contract failed")
        by_task[task].append(row)

    selected = {}
    for task in TASKS:
        ordered = sorted(by_task[task], key=lambda row: str(row["spec_id"]))
        if len(ordered) < per_task:
            raise ValueError(f"{task}: requested {per_task}, found {len(ordered)}")
        selected[task] = ordered[:per_task]

    output_dir.mkdir(parents=True)
    paths = {}
    for task, task_rows in selected.items():
        path = output_dir / f"{task}.jsonl"
        write_jsonl(path, task_rows)
        paths[task] = path
    manifest = {
        "schema": "shoppingbench-cp-sft-grounding-pilot-v1",
        "purpose": "validation-only online grounding and SFT-event engineering gate",
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "selection": "first rows after lexical spec_id sort; no retrieval or label filtering",
        "split": "val",
        "per_task": per_task,
        "total_rows": per_task * len(TASKS),
        "planner_input_fields": ["query"],
        "gold_runtime_access": False,
        "external_api_calls": 0,
        "files": {
            task: {"path": str(path), "sha256": sha256(path), "rows": len(selected[task])}
            for task, path in paths.items()
        },
        "spec_ids": {task: [row["spec_id"] for row in selected[task]] for task in TASKS},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build(args.input, args.output_dir, args.per_task),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
