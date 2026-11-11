#!/usr/bin/env python3
"""Freeze 2Wiki questions that exercised QueryRewrite under a logged policy run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def rewrite_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = row.get("raw_trace") or {}
    return [
        event
        for event in trace.get("sft_events", [])
        if isinstance(event, dict) and event.get("agent_name") == "QueryRewriteAgent"
    ]


def stable_key(row: dict[str, Any], seed: int) -> str:
    payload = f"{seed}\0{row.get('sample_index')}\0{row.get('question', '')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(args.trajectories.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("Expected a JSON list of trajectory rows")

    eligible = []
    for row in rows:
        trace = row.get("raw_trace") or {}
        plan = str(trace.get("raw_plan_output") or "")
        gold = answers(row.get("gold_answer") or row.get("golden_answers"))
        if not rewrite_events(row) or not plan.strip() or not gold:
            continue
        if row.get("pipeline_error") not in (None, "", False):
            continue
        eligible.append(row)

    eligible.sort(key=lambda row: stable_key(row, args.seed))
    selected = eligible[: args.samples]
    if len(selected) != args.samples:
        raise RuntimeError(f"Only {len(selected)} eligible rows for {args.samples} requested samples")
    if len({str(row.get("question", "")).strip() for row in selected}) != args.samples:
        raise RuntimeError("Selected questions are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    table_rows = []
    audit_rows = []
    for rank, row in enumerate(selected):
        trace = row["raw_trace"]
        events = rewrite_events(row)
        sample_id = str(row.get("sample_index"))
        table_rows.append(
            {
                "sample_index": sample_id,
                "question": str(row["question"]),
                "golden_answers": answers(row.get("gold_answer")),
                "gold_plan": str(trace["raw_plan_output"]),
            }
        )
        audit_rows.append(
            {
                "selection_rank": rank,
                "stable_hash": stable_key(row, args.seed),
                "sample_index": sample_id,
                "question": str(row["question"]),
                "golden_answers": answers(row.get("gold_answer")),
                "logged_prediction": str(row.get("pred_answer") or ""),
                "logged_em": float(row.get("em", 0.0)),
                "logged_f1": float(row.get("f1", 0.0)),
                "rewrite_event_count": len(events),
                "rewrite_responses": [str(event.get("response") or "") for event in events],
                "raw_plan_output": str(trace["raw_plan_output"]),
            }
        )

    parquet_path = args.output_dir / "2wikimultihopqa.parquet"
    pq.write_table(pa.Table.from_pylist(table_rows), parquet_path, compression="zstd")
    audit_path = args.output_dir / "selection_audit.json"
    audit_path.write_text(json.dumps(audit_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "schema": "dagent-query-rewrite-2wiki100-manifest-v1",
        "source_trajectories": str(args.trajectories),
        "source_trajectories_sha256": sha256_file(args.trajectories),
        "source_rows": len(rows),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "seed": args.seed,
        "selection": "ascending SHA256(seed, sample_index, question)",
        "requires_logged_query_rewrite": True,
        "freezes_logged_raw_plan_output_as_gold_plan": True,
        "parquet_sha256": sha256_file(parquet_path),
        "selection_audit_sha256": sha256_file(audit_path),
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
