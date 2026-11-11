#!/usr/bin/env python3
"""Freeze raw ShoppingBench Planner outputs before the system compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_many(
    predictions: list[Path], output_dir: Path, expected_rows: int
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    overrides: dict[str, str] = {}
    for path in predictions:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id") or "")
                trace = row.get("raw_trace") if isinstance(row.get("raw_trace"), dict) else {}
                raw_plan = str(trace.get("raw_plan_output") or "")
                if not sample_id or sample_id in overrides:
                    raise ValueError(
                        f"missing or duplicate sample_id at {path}:{line_number}"
                    )
                if "<plan" not in raw_plan:
                    raise ValueError(f"{sample_id}: missing raw planner output")
                overrides[sample_id] = raw_plan
    if len(overrides) != expected_rows:
        raise ValueError(f"expected {expected_rows} planner outputs, found {len(overrides)}")
    output_dir.mkdir(parents=True)
    override_path = output_dir / "planner_output_overrides.json"
    override_path.write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "shoppingbench-frozen-planner-output-override-v1",
        "source_predictions": [str(path) for path in predictions],
        "source_predictions_sha256": {
            str(path): sha256(path) for path in predictions
        },
        "rows": len(overrides),
        "fields_read": ["sample_id", "raw_trace.raw_plan_output"],
        "gold_fields_read": False,
        "plans_modified": False,
        "runtime_request_field": "planner_output_override",
        "planner_output_overrides_sha256": sha256(override_path),
        "purpose": "identity-controlled runtime intervention; not SFT data or method outcome evidence",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build(predictions: Path, output_dir: Path, expected_rows: int) -> dict[str, Any]:
    return build_many([predictions], output_dir, expected_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_many(args.predictions, args.output_dir, args.expected_rows), indent=2))


if __name__ == "__main__":
    main()
