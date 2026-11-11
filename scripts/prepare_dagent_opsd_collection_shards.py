#!/usr/bin/env python3
"""Create deterministic, auditable shards for full OPSD rollout collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--rollouts-per-question", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if args.rollouts_per_question < 2:
        parser.error("--rollouts-per-question must be at least 2")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_question(row: dict[str, Any]) -> str:
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict):
        question = extra_info.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    prompt = row.get("prompt") or []
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
    return ""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    from datasets import Dataset, load_dataset

    source = load_dataset("parquet", data_files=str(args.input), split="train")
    if len(source) < args.num_shards:
        raise RuntimeError(
            f"Dataset is smaller than shard count: {len(source)} < {args.num_shards}"
        )

    normalized_seen: dict[str, int] = {}
    collection_ids = []
    for index, row in enumerate(source):
        question = extract_question(row)
        if not question:
            raise RuntimeError(f"Missing question at source row {index}")
        normalized = " ".join(question.lower().split())
        if normalized in normalized_seen:
            raise RuntimeError(
                "Duplicate question would leak across OPSD groups: "
                f"rows {normalized_seen[normalized]} and {index}"
            )
        normalized_seen[normalized] = index
        collection_ids.append(
            hashlib.sha256(
                f"{args.seed}\x1f{index}\x1f{question}".encode("utf-8")
            ).hexdigest()[:24]
        )

    shuffled = list(range(len(source)))
    random.Random(args.seed).shuffle(shuffled)
    shard_indices = [
        shuffled[offset:: args.num_shards] for offset in range(args.num_shards)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    manifest_shards = []
    all_ids = set()
    for shard_id, indices in enumerate(shard_indices):
        rows = []
        ids = []
        for source_index in indices:
            row = dict(source[int(source_index)])
            collection_id = collection_ids[int(source_index)]
            row["opsd_collection_id"] = collection_id
            row["opsd_source_row_index"] = int(source_index)
            row["opsd_collection_shard"] = shard_id
            rows.append(row)
            ids.append(collection_id)
        if all_ids.intersection(ids):
            raise RuntimeError(f"Collection IDs overlap before shard {shard_id}")
        all_ids.update(ids)

        shard_path = shard_root / f"shard_{shard_id:03d}.parquet"
        temporary = shard_path.with_suffix(".parquet.tmp")
        Dataset.from_list(rows).to_parquet(str(temporary))
        os.replace(temporary, shard_path)
        manifest_shards.append(
            {
                "shard_id": shard_id,
                "path": str(shard_path),
                "questions": len(rows),
                "expected_rollouts": len(rows) * args.rollouts_per_question,
                "collection_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(ids)).encode("utf-8")
                ).hexdigest(),
                "file_sha256": sha256_file(shard_path),
            }
        )

    if len(all_ids) != len(source):
        raise RuntimeError(f"Shard coverage mismatch: {len(all_ids)}/{len(source)}")

    manifest = {
        "schema_version": 1,
        "method": "dagent_aligned_success_opsd_full_collection",
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "seed": args.seed,
        "question_count": len(source),
        "num_shards": args.num_shards,
        "rollouts_per_question": args.rollouts_per_question,
        "expected_rollouts": len(source) * args.rollouts_per_question,
        "question_identity": "opsd_collection_id",
        "duplicate_question_count": 0,
        "shards": manifest_shards,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
