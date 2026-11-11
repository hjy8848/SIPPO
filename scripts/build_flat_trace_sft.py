#!/usr/bin/env python3
"""Build a single-transcript SFT view of accepted ROSA workflows.

This keeps the executed observations from each event but removes the
event-level sample decomposition used by execution-grounded SFT. One parquet
row contains one complete workflow and masks every assistant action response.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


FLAT_SYSTEM = (
    "You are learning to reproduce an executed multi-hop QA workflow. "
    "Continue the transcript in order. User turns provide the original task "
    "or an execution observation; assistant turns are the workflow actions. "
    "Preserve the role tag and output format shown by each target action."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _content(message: dict[str, Any]) -> str:
    return str(message.get("content") or "").strip()


def event_context(event: dict[str, Any]) -> str:
    """Serialize the exact pre-response event context without its target."""

    role = str(event.get("agent_name") or "UnknownAgent")
    parts = [f"[Execution observation for {role}]"]
    for message in event.get("messages") or []:
        if not isinstance(message, dict):
            continue
        message_role = str(message.get("role") or "")
        content = _content(message)
        if not content or message_role == "system":
            continue
        parts.append(f"[{message_role}]\n{content}")
    if len(parts) == 1:
        parts.append("(no additional observation)")
    return "\n\n".join(parts)


def selected_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = record.get("selected_sft_events") or record.get("sft_events") or []
    if not isinstance(events, list) or not events:
        raise ValueError(f"trajectory {record.get('trajectory_id')} has no SFT events")
    return [event for event in events if isinstance(event, dict) and str(event.get("response") or "").strip()]


def build_row(record: dict[str, Any]) -> dict[str, Any]:
    question = str(record.get("question") or "").strip()
    if not question:
        raise ValueError(f"trajectory {record.get('trajectory_id')} has no question")
    events = selected_events(record)
    # The multiturn SFT loader requires the first exchange to be
    # system -> user -> assistant.  Put the first event's observation in
    # that initial user turn; later events remain user-observation/assistant
    # pairs.  This preserves the flat transcript without consecutive users.
    first_context = event_context(events[0])
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": FLAT_SYSTEM},
        {
            "role": "user",
            "content": f"[Original question]\n{question}\n\n{first_context}",
        },
    ]
    roles: list[str] = []
    for event_index, event in enumerate(events):
        role = str(event.get("agent_name") or "UnknownAgent")
        response = str(event.get("response") or "").strip()
        roles.append(role)
        if event_index > 0:
            messages.append({"role": "user", "content": event_context(event)})
        messages.append({"role": "assistant", "content": f"[{role}]\n{response}", "loss_mask": 1})

    return {
        "messages": messages,
        "agent_name": "FlatTraceAgent",
        "trajectory_id": str(record.get("trajectory_id") or record.get("source_index") or ""),
        "source_index": record.get("source_index", -1),
        "question": question,
        "event_count": len(events),
        "event_roles": roles,
        "flat_trace_version": "v2",
    }


def split_by_trajectory(rows: list[dict[str, Any]], seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    val_count = max(1, round(len(rows) * val_ratio)) if rows else 0
    val_indices = set(order[:val_count])
    train = [row for index, row in enumerate(rows) if index not in val_indices]
    val = [row for index, row in enumerate(rows) if index in val_indices]
    return train, val


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, seed: int, val_ratio: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val = split_by_trajectory(rows, seed, val_ratio)
    pd.DataFrame(rows).to_parquet(output_dir / "flat_sft_all.parquet", index=False)
    pd.DataFrame(train).to_parquet(output_dir / "flat_sft_train.parquet", index=False)
    pd.DataFrame(val).to_parquet(output_dir / "flat_sft_val.parquet", index=False)
    role_counts = Counter(role for row in rows for role in row["event_roles"])
    stats = {
        "flat_trace_version": "v1",
        "source_trajectories": len(rows),
        "all_rows": len(rows),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_trajectories": len(train),
        "val_trajectories": len(val),
        "event_role_counts": dict(role_counts),
        "seed": seed,
        "val_ratio": val_ratio,
        "max_event_count": max((row["event_count"] for row in rows), default=0),
        "total_target_turns": sum(row["event_count"] for row in rows),
    }
    (output_dir / "flat_trace_report.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    source = read_jsonl(args.input_jsonl)
    rows = [build_row(record) for record in source]
    stats = write_outputs(rows, args.output_dir, args.seed, args.val_ratio)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
