#!/usr/bin/env python3
"""Retry only HTTP completion failures from a completed external evaluation.

The original trajectory files remain untouched. This runner selects rows whose
recorded error is an HTTPStatusError, replays them with the frozen protocol,
and writes a fresh recovery artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evaluate_external_baseline_full_answerable import (
    CompletionClient,
    RetrieverClient,
    answer_values,
    load_tokenizer,
    run_one,
    score_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("search_r1", "zerosearch"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-errors", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config.setdefault("top_k", 0)
    config.setdefault("repetition_penalty", 1.0)
    config.setdefault("include_stop_str_in_output", True)
    return config


def select_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((input_dir / "trajectories").glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("pipeline_error") and "HTTPStatusError" in str(row.get("error")):
                item = dict(row)
                item["source_trajectory"] = str(path)
                rows.append(item)
    return rows


async def main_async(args: argparse.Namespace) -> None:
    rows = select_rows(args.input_dir)
    if len(rows) != args.expected_errors:
        raise RuntimeError(f"selected {len(rows)} HTTP errors, expected {args.expected_errors}")
    output_dir = args.output_dir
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output_dir}")
    (output_dir / "trajectories").mkdir(parents=True)
    config = load_config(args)
    tokenizer = load_tokenizer(args.tokenizer)
    completion = CompletionClient(args.api_base_url, args.model, args.timeout, concurrency=1)
    retriever = RetrieverClient(
        args.retriever_url,
        config,
        timeout=args.timeout,
        retries=3,
        concurrency=1,
    )
    recovered: list[dict[str, Any]] = []
    try:
        for row in rows:
            started = time.time()
            native = await run_one(row, config, tokenizer, completion, retriever)
            prediction = str(native["answer"])
            em, f1 = await score_answer(prediction, answer_values(row["golden_answers"]))
            recovered.append(
                {
                    "dataset": row["dataset"],
                    "sample_index": int(row["sample_index"]),
                    "question": row["question"],
                    "golden_answers": answer_values(row["golden_answers"]),
                    "prediction": prediction,
                    "em": em,
                    "f1": f1,
                    "pipeline_error": bool(native["format_failure"]),
                    "format_failure": bool(native["format_failure"]),
                    "turns": int(native["turns"]),
                    "search_calls": int(native["search_calls"]),
                    "trace": native["trace"],
                    "elapsed_seconds": time.time() - started,
                    "error": "format_failure" if native["format_failure"] else None,
                    "source_trajectory": row["source_trajectory"],
                    "source_error": row["error"],
                }
            )
    finally:
        await completion.aclose()
        await retriever.aclose()

    path = output_dir / "trajectories" / f"{args.baseline}_http_retries.json"
    path.write_text(json.dumps(recovered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "baseline": args.baseline,
        "source_dir": str(args.input_dir),
        "rows": len(recovered),
        "source_http_errors": len(rows),
        "recovered_pipeline_errors": sum(bool(row["pipeline_error"]) for row in recovered),
        "recovered_em_sum": sum(float(row["em"]) for row in recovered),
        "recovered_f1_sum": sum(float(row["f1"]) for row in recovered),
        "output": str(path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
