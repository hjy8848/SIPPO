#!/usr/bin/env python3
"""Repair v1 Flat-Trace rows to satisfy the multiturn SFT message contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def repair_messages(messages: list[dict]) -> list[dict]:
    if len(messages) < 4:
        raise ValueError("expected at least system, two user turns, and one assistant turn")
    roles = [message.get("role") for message in messages]
    if roles[:4] != ["system", "user", "user", "assistant"]:
        raise ValueError(f"unexpected v1 prefix: {roles[:4]}")
    merged_user = dict(messages[1])
    merged_user["content"] = (
        f"{merged_user.get('content', '')}\n\n{messages[2].get('content', '')}"
    )
    repaired = [messages[0], merged_user, *messages[3:]]
    repaired_roles = [message.get("role") for message in repaired]
    if repaired_roles[:3] != ["system", "user", "assistant"]:
        raise AssertionError(repaired_roles[:3])
    if any(
        repaired_roles[index] == repaired_roles[index + 1] == "user"
        for index in range(len(repaired_roles) - 1)
    ):
        raise AssertionError(f"consecutive user turns remain: {repaired_roles}")
    return repaired


def repair_file(source: Path, target: Path) -> int:
    frame = pd.read_parquet(source)
    repaired = []
    for row in frame.to_dict(orient="records"):
        row["messages"] = repair_messages(row["messages"])
        row["flat_trace_version"] = "v2"
        repaired.append(row)
    pd.DataFrame(repaired).to_parquet(target, index=False)
    return len(repaired)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    counts = {}
    for split in ("all", "train", "val"):
        counts[split] = repair_file(
            args.source_dir / f"flat_sft_{split}.parquet",
            args.output_dir / f"flat_sft_{split}.parquet",
        )
    report = {
        "flat_trace_version": "v2",
        "source_dir": str(args.source_dir),
        "row_counts": counts,
        "message_contract": "system,user,assistant with no consecutive user turns",
    }
    (args.output_dir / "flat_trace_repair_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
