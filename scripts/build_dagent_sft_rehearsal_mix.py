#!/usr/bin/env python3
"""Build a canonical DAGENT SFT mix with replay from the original SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from qa_manager.ROSA import protocol


SYSTEM_PROMPTS = {
    "PlanningAgent": protocol.PLANNER_SYSTEM_MSG,
    "QueryRewriteAgent": protocol.QUERY_REWRITE_SYSTEM_MSG,
    "AnswerGenerationAgent": protocol.ANSWER_GENERATION_SYSTEM_MSG,
    "ReasonerAgent": protocol.REASON_SYSTEM_MSG,
    "AnswerSummarizationAgent": protocol.FINAL_SYSTEM_MSG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-data-dir", type=Path, required=True)
    parser.add_argument("--replay-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-new-only-steps", type=int, default=50)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_messages(value: Any, agent_name: str) -> tuple[list[dict[str, Any]], bool]:
    if agent_name not in SYSTEM_PROMPTS:
        raise ValueError(f"unsupported agent_name: {agent_name}")
    messages = [dict(message) for message in list(value)]
    if len(messages) < 3 or messages[0].get("role") != "system":
        raise ValueError(f"invalid message sequence for {agent_name}")
    assistant_masks = [
        float(message.get("loss_mask") or 0.0)
        for message in messages
        if message.get("role") == "assistant"
    ]
    if not assistant_masks or sum(mask > 0 for mask in assistant_masks) != 1:
        raise ValueError(f"expected exactly one supervised assistant response for {agent_name}")
    if any(mask not in {0.0, 1.0} for mask in assistant_masks):
        raise ValueError(f"assistant loss masks must be binary for {agent_name}")
    mismatch = messages[0].get("content") != SYSTEM_PROMPTS[agent_name]
    messages[0]["content"] = SYSTEM_PROMPTS[agent_name]
    return messages, mismatch


def canonicalize_frame(frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, int]:
    required = {"messages", "agent_name", "trajectory_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} frame is missing columns: {missing}")
    result = frame.copy()
    normalized = []
    mismatches = 0
    for row in result.itertuples(index=False):
        messages, mismatch = normalize_messages(row.messages, str(row.agent_name))
        normalized.append(messages)
        mismatches += int(mismatch)
    result["messages"] = normalized
    result["rehearsal_source"] = source
    return result, mismatches


def prompt_mismatch_count(frame: pd.DataFrame) -> int:
    return sum(
        list(row.messages)[0].get("content") != SYSTEM_PROMPTS[str(row.agent_name)]
        for row in frame.itertuples(index=False)
    )


def assistant_loss_mask_is_valid(messages: Any) -> bool:
    values = [
        message.get("loss_mask")
        for message in list(messages)
        if message.get("role") == "assistant"
    ]
    numeric = [float(value or 0.0) for value in values]
    return (
        bool(numeric)
        and sum(value > 0 for value in numeric) == 1
        and all(value in {0.0, 1.0} for value in numeric)
    )


def frame_report(frame: pd.DataFrame) -> dict[str, Any]:
    sources = Counter(str(value) for value in frame["rehearsal_source"])
    roles = Counter(str(value) for value in frame["agent_name"])
    return {
        "rows": len(frame),
        "trajectories": int(frame["trajectory_id"].astype(str).nunique()),
        "source_rows": dict(sorted(sources.items())),
        "role_rows": dict(sorted(roles.items())),
        "new_fraction": sources.get("new", 0) / max(1, len(frame)),
        "replay_fraction": sources.get("replay", 0) / max(1, len(frame)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.reference_new_only_steps <= 0:
        raise ValueError("reference_new_only_steps must be positive")

    inputs = {}
    for split in ("train", "val"):
        new_path = args.new_data_dir / f"sft_{split}.parquet"
        replay_path = args.replay_data_dir / f"sft_{split}.parquet"
        if not new_path.is_file() or not replay_path.is_file():
            raise FileNotFoundError(f"missing {split} input: {new_path} or {replay_path}")
        inputs[split] = (new_path, replay_path)

    outputs: dict[str, pd.DataFrame] = {}
    mismatch_before: dict[str, dict[str, int]] = {}
    for split, (new_path, replay_path) in inputs.items():
        new_frame, new_mismatch = canonicalize_frame(pd.read_parquet(new_path), "new")
        replay_frame, replay_mismatch = canonicalize_frame(
            pd.read_parquet(replay_path), "replay"
        )
        mismatch_before[split] = {"new": new_mismatch, "replay": replay_mismatch}
        combined = pd.concat([new_frame, replay_frame], ignore_index=True, sort=False)
        outputs[split] = combined.sample(
            frac=1.0, random_state=args.seed + (0 if split == "train" else 1)
        ).reset_index(drop=True)

    train = outputs["train"]
    val = outputs["val"]
    train_fraction = frame_report(train)["new_fraction"]
    matched_steps = math.ceil(args.reference_new_only_steps / train_fraction)
    train_ids = set(train["trajectory_id"].astype(str))
    val_ids = set(val["trajectory_id"].astype(str))
    hard_gates = {
        "canonical_train_prompts": prompt_mismatch_count(train) == 0,
        "canonical_val_prompts": prompt_mismatch_count(val) == 0,
        "valid_train_loss_masks": all(map(assistant_loss_mask_is_valid, train["messages"])),
        "valid_val_loss_masks": all(map(assistant_loss_mask_is_valid, val["messages"])),
        "both_train_sources_present": set(train["rehearsal_source"]) == {"new", "replay"},
        "both_val_sources_present": set(val["rehearsal_source"]) == {"new", "replay"},
        "train_val_trajectory_disjoint": not bool(train_ids & val_ids),
    }
    if not all(hard_gates.values()):
        raise ValueError(f"rehearsal mix hard gates failed: {hard_gates}")

    args.output_dir.mkdir(parents=True)
    output_paths = {}
    for split, frame in outputs.items():
        path = args.output_dir / f"sft_{split}.parquet"
        frame.to_parquet(path, index=False)
        output_paths[split] = path

    report = {
        "method": "canonical_sft_rehearsal_mix_v1",
        "prompt_schema": "qa_manager.ROSA.protocol.v1",
        "seed": args.seed,
        "new_data_dir": str(args.new_data_dir),
        "replay_data_dir": str(args.replay_data_dir),
        "prompt_mismatches_replaced": mismatch_before,
        "train": frame_report(train),
        "val": frame_report(val),
        "reference_new_only_steps": args.reference_new_only_steps,
        "matched_new_exposure_steps": matched_steps,
        "hard_gates": hard_gates,
        "files": {
            split: {"path": str(path), "sha256": sha256(path)}
            for split, path in output_paths.items()
        },
    }
    (args.output_dir / "rehearsal_mix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
