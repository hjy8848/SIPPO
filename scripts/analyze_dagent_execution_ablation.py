#!/usr/bin/env python3
"""Validate and summarize paired frozen-policy execution ablations."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list: {path}")
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sample_index")),
        int(row.get("rollout_index", 0)),
        str(row.get("question") or "").strip(),
    )


def rewrite_events(row: dict[str, Any]) -> int:
    return sum(
        isinstance(event, dict) and event.get("agent_name") == "QueryRewriteAgent"
        for event in (row.get("raw_trace") or {}).get("sft_events", [])
    )


def executed_nodes(row: dict[str, Any]) -> int:
    nodes = (row.get("raw_trace") or {}).get("nodes_execution") or []
    return len(nodes) if isinstance(nodes, list) else 0


def summarize(rows: list[dict[str, Any]], expected: int, *, node_cap: int | None) -> dict[str, Any]:
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(rows)}")
    if len({row_key(row) for row in rows}) != expected:
        raise RuntimeError("Duplicate or missing rollout identity")
    errors = [row for row in rows if row.get("error") or row.get("pipeline_error")]
    if errors:
        raise RuntimeError(f"Execution/API errors: {len(errors)}")
    for row in rows:
        trace = row.get("raw_trace") or {}
        if trace.get("plan_override_used") is not True:
            raise RuntimeError("A row did not acknowledge the frozen planner output")
        if not all(math.isfinite(float(row.get(metric, float("nan")))) for metric in ("em", "f1")):
            raise RuntimeError("Non-finite EM/F1")
    if node_cap is not None:
        observed = [executed_nodes(row) for row in rows]
        if max(observed, default=0) > node_cap:
            raise RuntimeError(f"Node cap violated: max={max(observed)} cap={node_cap}")
    return {
        "n": expected,
        "em": sum(float(row["em"]) for row in rows) / expected,
        "f1": sum(float(row["f1"]) for row in rows) / expected,
        "execution_errors": 0,
        "query_rewrite_events": sum(rewrite_events(row) for row in rows),
        "mean_search_nodes": sum(float(row.get("search_steps") or 0) for row in rows) / expected,
        "mean_retrieval_calls": sum(float(row.get("retrieval_calls") or 0) for row in rows) / expected,
        "mean_executed_nodes": sum(executed_nodes(row) for row in rows) / expected,
        "node_cap": node_cap,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def paired_summary(base: list[dict[str, Any]], variant: list[dict[str, Any]], *, seed: int = 42) -> dict[str, Any]:
    base_by_key = {row_key(row): row for row in base}
    variant_by_key = {row_key(row): row for row in variant}
    keys = sorted(base_by_key)
    if set(keys) != set(variant_by_key):
        raise RuntimeError("Paired comparison received mismatched identities")

    em_delta = [float(variant_by_key[key]["em"]) - float(base_by_key[key]["em"]) for key in keys]
    f1_delta = [float(variant_by_key[key]["f1"]) - float(base_by_key[key]["f1"]) for key in keys]
    base_only = sum(float(base_by_key[key]["em"]) > 0.5 and float(variant_by_key[key]["em"]) <= 0.5 for key in keys)
    variant_only = sum(float(base_by_key[key]["em"]) <= 0.5 and float(variant_by_key[key]["em"]) > 0.5 for key in keys)
    wins = sum(delta > 0.0 for delta in em_delta)
    losses = sum(delta < 0.0 for delta in em_delta)
    ties = len(em_delta) - wins - losses

    # Percentile bootstrap over question-level paired deltas, with a fixed seed.
    rng = random.Random(seed)
    em_boot: list[float] = []
    f1_boot: list[float] = []
    n = len(keys)
    for _ in range(2000):
        sample = [rng.randrange(n) for _ in range(n)]
        em_boot.append(sum(em_delta[i] for i in sample) / n)
        f1_boot.append(sum(f1_delta[i] for i in sample) / n)

    discordant = base_only + variant_only
    if discordant == 0:
        mcnemar_p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(base_only, variant_only) + 1)) / (2.0 ** discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    return {
        "n": n,
        "em_delta_variant_minus_baseline": sum(em_delta) / n,
        "em_delta_bootstrap_ci95": [percentile(em_boot, 0.025), percentile(em_boot, 0.975)],
        "f1_delta_variant_minus_baseline": sum(f1_delta) / n,
        "f1_delta_bootstrap_ci95": [percentile(f1_boot, 0.025), percentile(f1_boot, 0.975)],
        "em_wins": wins,
        "em_losses": losses,
        "em_ties": ties,
        "baseline_only_em": base_only,
        "variant_only_em": variant_only,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=2000)
    args = parser.parse_args()

    protocol = json.loads((args.run_root / "evaluation_protocol.json").read_text(encoding="utf-8"))
    arms = protocol["arms"]
    loaded: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, config in arms.items():
        path = args.run_root / name / "trajectories" / f"full_logs_{protocol['run_tag']}-{name}_2wikimultihopqa.json"
        rows = load_rows(path)
        loaded[name] = rows
        cap = int(config["max_nodes_per_turn"]) if config.get("apply_node_cap_always") else None
        summaries[name] = summarize(rows, args.expected, node_cap=cap)
        if name == "no_query_rewrite" and summaries[name]["query_rewrite_events"] != 0:
            raise RuntimeError("No-rewrite arm emitted QueryRewrite events")

    keys = {name: {row_key(row) for row in rows} for name, rows in loaded.items()}
    first = next(iter(keys.values()))
    if any(value != first for value in keys.values()):
        raise RuntimeError("Arm identity sets are not paired")

    baseline = loaded["baseline"]
    paired = {
        name: paired_summary(baseline, rows)
        for name, rows in loaded.items()
        if name != "baseline"
    }

    report = {
        "schema": "dagent-frozen-policy-execution-ablation-v1",
        "contract": protocol,
        "arms": summaries,
        "paired_identity_count": len(first),
        "paired_vs_baseline": paired,
    }
    (args.run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
