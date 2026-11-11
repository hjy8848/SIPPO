#!/usr/bin/env python3
"""Run native Search-R1 tool use against serially selected EWC overlay worlds.

The overlay service holds the selected counterfactual world as process state,
so this evaluator deliberately runs one instance at a time.  Search-R1 still
uses its native Qwen3 thinking and tool-call protocol; selection only changes
the retrieval world before a trajectory starts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import httpx

from evaluate_ewc_pilot import em, paired_summary
from evaluate_search_r1_full_answerable import (
    OpenAICompletionCompatBackend,
    extract_answer,
    run_one,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--overlay-url", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-instances", type=int, help="Optional smoke limit; omit for all worlds.")
    return parser.parse_args()


def load_instances(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"instance_id", "pair_id", "variant", "question", "gold_answer"}
    for index, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{path}:{index}: missing {missing}")
    return rows[:limit] if limit is not None else rows


async def select_world(client: httpx.AsyncClient, overlay_url: str, instance_id: str) -> None:
    response = await client.post(
        overlay_url.rstrip("/") + "/select",
        json={"instance_id": instance_id},
    )
    response.raise_for_status()
    if response.json().get("active_instance_id") != instance_id:
        raise RuntimeError(f"overlay selected an unexpected world for {instance_id}")


async def evaluate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    backend = OpenAICompletionCompatBackend(
        model=args.model,
        base_url=args.api_base_url,
        timeout=args.timeout,
    )
    overlay_client = httpx.AsyncClient(timeout=args.timeout)
    results: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, 1):
            final_answer = ""
            trace: list[dict[str, Any]] = []
            turns = 0
            error: str | None = None
            try:
                await select_world(overlay_client, args.overlay_url, str(row["instance_id"]))
                native = await run_one(
                    backend,
                    args.overlay_url,
                    str(row["question"]),
                    args.max_turns,
                    args.max_tokens,
                )
                final_answer = extract_answer(native["final_text"])
                trace = native["trace"]
                turns = int(native["turns"])
            except Exception as exc:  # Keep the failed world for fail-closed scoring.
                error = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "instance_id": row["instance_id"],
                    "pair_id": row["pair_id"],
                    "variant": row["variant"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "final_answer": final_answer,
                    "em": em(final_answer, list(row["gold_answer"])),
                    "pipeline_error": error is not None,
                    "error": error,
                    "turns": turns,
                    "trace": trace,
                }
            )
            print(json.dumps({"completed": index, "total": len(rows), "variant": row["variant"]}), flush=True)
    finally:
        await backend.aclose()
        await overlay_client.aclose()
    return results


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    rows = load_instances(args.instances, args.max_instances)
    if not rows:
        raise SystemExit("no EWC instances selected")
    results = asyncio.run(evaluate(args, rows))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results) + "\n",
        encoding="utf-8",
    )
    summary = {
        "benchmark": "EWC-QA validation v6",
        "instances": len(results),
        "em": sum(int(row["em"]) for row in results),
        "em_rate": sum(int(row["em"]) for row in results) / len(results),
        "pipeline_errors": sum(bool(row["pipeline_error"]) for row in results),
        "paired": paired_summary(results) if args.max_instances is None else None,
        "protocol": {
            "runtime": "native OpenRLHF-Agent Qwen3Protocol(enable_thinking=true)",
            "retriever": "EWC overlay /retrieve",
            "selection": "serial POST /select before each trajectory",
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "instances_sha256": sha256(args.instances),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["pipeline_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
