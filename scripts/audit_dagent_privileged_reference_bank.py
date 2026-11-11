#!/usr/bin/env python3
"""Verify an RL parquet has a complete immutable privileged-reference join."""

import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.trainer.ppo.workflow_sod import (
    load_privileged_reference_bank,
    normalize_reference_question,
)


REQUIRED_ROLES = {
    "PlanningAgent",
    "AnswerGenerationAgent",
    "AnswerSummarizationAgent",
}


def parquet_questions(path: Path) -> list[str]:
    table = pq.read_table(path, columns=["prompt"])
    questions = []
    for row_index, prompt in enumerate(table.column("prompt").to_pylist()):
        user_messages = [
            str(message.get("content", ""))
            for message in (prompt or [])
            if message.get("role") == "user"
        ]
        if len(user_messages) != 1:
            raise ValueError(
                f"parquet row {row_index} must contain exactly one user question"
            )
        questions.append(normalize_reference_question(user_messages[0]))
    return questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--reference-bank", type=Path, required=True)
    parser.add_argument("--require-count", type=int, default=0)
    args = parser.parse_args()

    bank = load_privileged_reference_bank(args.reference_bank)
    questions = parquet_questions(args.train_parquet)
    if len(set(questions)) != len(questions):
        raise ValueError("training parquet contains duplicate normalized questions")
    missing = sorted(set(questions) - set(bank))
    extra = sorted(set(bank) - set(questions))
    if missing or extra:
        raise ValueError(
            "reference-bank join is not exact: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"missing_examples={missing[:3]}, extra_examples={extra[:3]}"
        )
    if args.require_count and len(questions) != args.require_count:
        raise ValueError(
            f"expected {args.require_count} joined questions, found {len(questions)}"
        )

    role_counts = {}
    for entry in bank.values():
        roles = {record["role"] for record in entry["action_records"]}
        absent = REQUIRED_ROLES - roles
        if absent:
            raise ValueError(
                f"reference {entry['trajectory_id']} is missing roles {sorted(absent)}"
            )
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
    print(
        json.dumps(
            {
                "train_samples": len(questions),
                "reference_records": len(bank),
                "metadata_joined": len(questions),
                "missing": 0,
                "extra": 0,
                "role_trajectory_counts": role_counts,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
