#!/usr/bin/env python3
"""Validate complete OPSD rollout shards before pair mining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rollout_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sample_index")),
        int(row.get("rollout_index", 0)),
        str(row.get("question") or ""),
    )


def main() -> int:
    args = parse_args()
    manifest_path = args.collection_root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    all_keys = set()
    all_questions: dict[str, str] = {}
    logs = []
    api_errors = 0
    pipeline_errors = 0
    for shard in manifest["shards"]:
        shard_id = int(shard["shard_id"])
        result_path = (
            args.collection_root
            / "rollouts"
            / f"shard_{shard_id:03d}"
            / f"full_logs_dagent_opsd_full_shard_{shard_id:03d}_"
            f"dagent_opsd_full_shard_{shard_id:03d}.json"
        )
        if not result_path.is_file():
            raise RuntimeError(f"Missing rollout log: {result_path}")
        with result_path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if len(rows) != int(shard["expected_rollouts"]):
            raise RuntimeError(
                f"Shard {shard_id} is incomplete: "
                f"{len(rows)}/{shard['expected_rollouts']}"
            )
        shard_keys = {rollout_key(row) for row in rows}
        if len(shard_keys) != len(rows):
            raise RuntimeError(f"Shard {shard_id} has duplicate rollout keys")
        if all_keys.intersection(shard_keys):
            raise RuntimeError(f"Shard {shard_id} overlaps a previous shard")
        all_keys.update(shard_keys)
        for row in rows:
            collection_id = str(row.get("sample_index"))
            question = str(row.get("question") or "")
            previous = all_questions.setdefault(collection_id, question)
            if previous != question:
                raise RuntimeError(
                    f"Collection ID {collection_id} maps to multiple questions"
                )
            api_errors += int(bool(row.get("error")))
            pipeline_errors += int(bool(row.get("pipeline_error")))
        logs.append(str(result_path))

    if len(all_keys) != int(manifest["expected_rollouts"]):
        raise RuntimeError(
            f"Global rollout coverage mismatch: "
            f"{len(all_keys)}/{manifest['expected_rollouts']}"
        )
    if len(all_questions) != int(manifest["question_count"]):
        raise RuntimeError(
            f"Global question coverage mismatch: "
            f"{len(all_questions)}/{manifest['question_count']}"
        )
    if api_errors:
        raise RuntimeError(f"Collection still contains {api_errors} API errors")

    summary = {
        "schema_version": 1,
        "status": "complete",
        "question_count": len(all_questions),
        "rollout_count": len(all_keys),
        "api_errors": api_errors,
        "pipeline_errors": pipeline_errors,
        "rollout_logs": logs,
        "manifest_sha256": sha256_file(manifest_path),
    }
    if args.write_summary:
        output = args.collection_root / "collection_audit.json"
        temporary = output.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
