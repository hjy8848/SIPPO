#!/usr/bin/env python3
"""Fail-closed audit for one complete DAGENT full-split evaluation arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_dagent_full_3b_eval import DATASETS, load_arm, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-planner-system-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, summary = load_manifest(args.manifest_dir)
    arm, result_shas = load_arm(args.run_root, manifest)
    prompt_hashes: Counter[str] = Counter()
    query_rewrite_calls = 0
    execution_errors: Counter[str] = Counter()
    contract_protocol_rows = 0
    samples = 0
    for dataset in DATASETS:
        for row in arm[dataset]:
            samples += 1
            trace = row.get("raw_trace") or {}
            contract_protocol_rows += bool(trace.get("contract_protocol"))
            execution_error = str(trace.get("execution_error") or "").strip()
            if execution_error:
                execution_errors[execution_error] += 1
            events = trace.get("sft_events") or []
            planning = [
                event
                for event in events
                if (event or {}).get("agent_name") == "PlanningAgent"
            ]
            if len(planning) != 1:
                raise RuntimeError(
                    f"Expected one PlanningAgent event, got {len(planning)}: {row.get('question')}"
                )
            messages = planning[0].get("messages") or []
            if not messages or not isinstance(messages[0], dict):
                raise RuntimeError(f"Missing PlanningAgent system message: {row.get('question')}")
            system = str(messages[0].get("content") or "")
            prompt_hashes[hashlib.sha256(system.encode()).hexdigest()] += 1
            query_rewrite_calls += sum(
                (event or {}).get("agent_name") == "QueryRewriteAgent" for event in events
            )
            for metric in ("em", "f1"):
                if not math.isfinite(float(row[metric])):
                    raise RuntimeError(f"Non-finite {metric}: {row.get('question')}")
    expected_prompt_hashes = Counter({args.expected_planner_system_sha256: samples})
    report: dict[str, Any] = {
        "complete": False,
        "samples": samples,
        "expected_samples": int(summary["total_selected_rows"]),
        "dataset_counts": {name: len(arm[name]) for name in DATASETS},
        "result_sha256": result_shas,
        "planner_system_sha256_counts": dict(prompt_hashes),
        "query_rewrite_calls": query_rewrite_calls,
        "execution_error_rows": sum(execution_errors.values()),
        "execution_error_reasons": dict(execution_errors.most_common()),
        "pipeline_errors": 0,
        "contract_protocol_rows": contract_protocol_rows,
        "legacy_contract": {
            "runtime_max_nodes_per_turn": 0,
            "planner_prompt_max_nodes": 0,
            "contract_preserving_dag_v1": 0,
            "typed_reason_handoff_v1": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "topk": 3,
            "thinking": False,
        },
    }
    if samples != int(summary["total_selected_rows"]):
        raise RuntimeError(f"Full-split sample mismatch: {samples}")
    if prompt_hashes != expected_prompt_hashes:
        raise RuntimeError(f"Planner prompt hash mismatch: {dict(prompt_hashes)}")
    if contract_protocol_rows:
        raise RuntimeError(f"CP protocol unexpectedly active in {contract_protocol_rows} rows")
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
