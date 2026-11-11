#!/usr/bin/env python3
"""Convert ROSA multiturn SFT parquet files to LLaMA-Factory OpenAI JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_ROLES = {"system", "user", "assistant"}


def as_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")

    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("every message must be an object")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unsupported message role: {role!r}")
        if not content:
            raise ValueError(f"empty {role!r} message")
        messages.append({"role": role, "content": content})

    if messages[-1]["role"] != "assistant":
        raise ValueError("the final message must be the supervised assistant response")
    if sum(message["role"] == "assistant" for message in messages) < 1:
        raise ValueError("sample has no assistant response")
    return messages


def convert_split(source: Path, target: Path) -> dict[str, Any]:
    frame = pd.read_parquet(source)
    if "messages" not in frame.columns:
        raise ValueError(f"{source} has no 'messages' column")

    role_counts: dict[str, int] = {}
    history_assistant_rows = 0
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row_number, value in enumerate(frame["messages"], start=1):
            try:
                messages = as_messages(value)
            except ValueError as exc:
                raise ValueError(f"{source}, row {row_number}: {exc}") from exc
            if sum(message["role"] == "assistant" for message in messages) > 1:
                history_assistant_rows += 1
            for message in messages:
                role_counts[message["role"]] = role_counts.get(message["role"], 0) + 1
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    temp.replace(target)
    return {
        "source": str(source),
        "output": str(target),
        "rows": len(frame),
        "rows_with_assistant_history": history_assistant_rows,
        "message_role_counts": role_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing sft_train.parquet and sft_val.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="LLaMA-Factory dataset directory (default: INPUT_DIR/llamafactory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "llamafactory").resolve()
    train_source = input_dir / "sft_train.parquet"
    val_source = input_dir / "sft_val.parquet"
    for source in (train_source, val_source):
        if not source.is_file():
            raise FileNotFoundError(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_stats = convert_split(train_source, output_dir / "rosa_sft_train.jsonl")
    val_stats = convert_split(val_source, output_dir / "rosa_sft_val.jsonl")

    dataset_entry = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    dataset_info = {
        "rosa_sft_train": {
            "file_name": "rosa_sft_train.jsonl",
            **dataset_entry,
        },
        "rosa_sft_val": {
            "file_name": "rosa_sft_val.jsonl",
            **dataset_entry,
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train": train_stats,
        "validation": val_stats,
        "important": "Train with mask_history=true so only the final assistant turn contributes to loss.",
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
