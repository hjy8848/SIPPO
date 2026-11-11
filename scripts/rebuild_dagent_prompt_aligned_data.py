#!/usr/bin/env python3
"""Rebuild accepted DAGENT trajectories and SFT files with canonical prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts import generate_dagent_reason_search_dataset as generator
except ImportError:
    import generate_dagent_reason_search_dataset as generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-accepted-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rebuild(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    records = generator.read_jsonl(args.source_accepted_jsonl)
    if not records:
        raise ValueError("source accepted workflow bank is empty")
    trajectory_ids = [str(record["trajectory_id"]) for record in records]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("source contains duplicate trajectory IDs")

    aligned = [generator.canonicalize_record_events(record) for record in records]
    args.output_dir.mkdir(parents=True)
    accepted_path = args.output_dir / "accepted_trajectories.jsonl"
    write_jsonl(accepted_path, aligned)
    output_stats = generator.rebuild_outputs(
        aligned,
        args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )
    role_counts = Counter(
        event["agent_name"] for record in aligned for event in record["sft_events"]
    )
    report = {
        "method": "dagent_prompt_aligned_rebuild_v1",
        "prompt_schema": "qa_manager.ROSA.protocol.v1",
        "source_accepted_jsonl": str(args.source_accepted_jsonl),
        "source_sha256": sha256(args.source_accepted_jsonl),
        "accepted_trajectories": len(aligned),
        "accepted_sha256": sha256(accepted_path),
        "role_counts": dict(role_counts),
        "canonical_prompt_hashes": generator.prompt_hashes(aligned),
        "output_stats": output_stats,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    expected = {
        "PlanningAgent": len(aligned),
        "AnswerGenerationAgent": len(aligned) * 3,
        "ReasonerAgent": len(aligned),
        "QueryRewriteAgent": len(aligned),
        "AnswerSummarizationAgent": len(aligned),
    }
    report["hard_gates"] = {
        "all_trajectory_ids_unique": len(trajectory_ids) == len(set(trajectory_ids)),
        "all_records_marked_canonical": all(
            record.get("prompt_schema") == "qa_manager.ROSA.protocol.v1" for record in aligned
        ),
        "all_events_exactly_rebuilt_from_protocol": all(
            record["sft_events"] == generator.build_events(record) for record in aligned
        ),
        "role_counts_exact": dict(role_counts) == expected,
        "sft_rows_exact": output_stats["sft_rows"] == len(aligned) * 7,
        "split_title_overlap_zero": not any(output_stats["supporting_title_overlap"].values()),
    }
    if not all(report["hard_gates"].values()):
        raise ValueError(f"prompt-aligned rebuild gates failed: {report['hard_gates']}")
    write_json(args.output_dir / "prompt_alignment_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    if not args.source_accepted_jsonl.is_file():
        raise SystemExit(f"source accepted JSONL not found: {args.source_accepted_jsonl}")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("val/test ratios must be nonnegative and sum to less than one")
    report = rebuild(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
