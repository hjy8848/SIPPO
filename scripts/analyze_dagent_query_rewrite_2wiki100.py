#!/usr/bin/env python3
"""Audit paired final-answer effects of model rewrite versus direct substitution."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list: {path}")
    return rows


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_index")), str(row.get("question") or "").strip()


def rewrites(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in (row.get("raw_trace") or {}).get("sft_events", [])
        if isinstance(event, dict) and event.get("agent_name") == "QueryRewriteAgent"
    ]


def bootstrap_ci(values: list[float], *, seed: int, samples: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples))
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def exact_mcnemar_p(rewrite_only: int, direct_only: int) -> float:
    discordant = rewrite_only + direct_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(rewrite_only, direct_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def node_queries(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(node.get("id")): str(node.get("executed_query") or "")
        for node in (row.get("raw_trace") or {}).get("nodes_execution", [])
        if node.get("type") == "search"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-rewrite", type=Path, required=True)
    parser.add_argument("--substitution-only", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rewrite_rows = load(args.with_rewrite)
    direct_rows = load(args.substitution_only)
    if len(rewrite_rows) != args.expected or len(direct_rows) != args.expected:
        raise RuntimeError(
            f"Expected {args.expected} rows per arm, got {len(rewrite_rows)} and {len(direct_rows)}"
        )
    rewrite_by_key = {key(row): row for row in rewrite_rows}
    direct_by_key = {key(row): row for row in direct_rows}
    if len(rewrite_by_key) != args.expected or rewrite_by_key.keys() != direct_by_key.keys():
        raise RuntimeError("Paired key mismatch or duplicate keys")

    paired = []
    for item_key in sorted(rewrite_by_key):
        rewrite = rewrite_by_key[item_key]
        direct = direct_by_key[item_key]
        rewrite_trace = rewrite.get("raw_trace") or {}
        direct_trace = direct.get("raw_trace") or {}
        if rewrite_trace.get("raw_plan_output") != direct_trace.get("raw_plan_output"):
            raise RuntimeError(f"Frozen plan mismatch: {item_key}")
        if rewrite_trace.get("plan_override_used") is not True or direct_trace.get("plan_override_used") is not True:
            raise RuntimeError(f"Plan override was not acknowledged: {item_key}")
        if not rewrites(rewrite):
            raise RuntimeError(f"Rewrite arm did not invoke QueryRewriteAgent: {item_key}")
        if rewrites(direct):
            raise RuntimeError(f"Substitution-only arm invoked QueryRewriteAgent: {item_key}")
        if (direct_trace.get("ablation_config") or {}).get("disable_query_rewrite") is not True:
            raise RuntimeError(f"Substitution-only arm did not record the ablation: {item_key}")
        if any(
            row.get(field) not in (None, "", False)
            for row in (rewrite, direct)
            for field in ("pipeline_error", "error")
        ):
            raise RuntimeError(f"Pipeline/API failure in pair: {item_key}")

        rewrite_queries = node_queries(rewrite)
        direct_queries = node_queries(direct)
        common_nodes = rewrite_queries.keys() & direct_queries.keys()
        changed_nodes = [node_id for node_id in common_nodes if rewrite_queries[node_id] != direct_queries[node_id]]
        paired.append(
            {
                "sample_index": item_key[0],
                "question": item_key[1],
                "gold_answer": rewrite.get("gold_answer"),
                "rewrite_prediction": rewrite.get("pred_answer"),
                "substitution_prediction": direct.get("pred_answer"),
                "rewrite_em": float(rewrite.get("em", 0.0)),
                "substitution_em": float(direct.get("em", 0.0)),
                "em_delta": float(rewrite.get("em", 0.0)) - float(direct.get("em", 0.0)),
                "rewrite_f1": float(rewrite.get("f1", 0.0)),
                "substitution_f1": float(direct.get("f1", 0.0)),
                "f1_delta": float(rewrite.get("f1", 0.0)) - float(direct.get("f1", 0.0)),
                "answer_changed": str(rewrite.get("pred_answer")) != str(direct.get("pred_answer")),
                "rewrite_event_count": len(rewrites(rewrite)),
                "changed_executed_query_nodes": len(changed_nodes),
                "rewrite_retrieval_calls": int(rewrite.get("retrieval_calls", 0)),
                "substitution_retrieval_calls": int(direct.get("retrieval_calls", 0)),
            }
        )

    em_deltas = [row["em_delta"] for row in paired]
    f1_deltas = [row["f1_delta"] for row in paired]
    rewrite_only = sum(delta > 0 for delta in em_deltas)
    direct_only = sum(delta < 0 for delta in em_deltas)
    report = {
        "schema": "dagent-query-rewrite-2wiki100-paired-v1",
        "samples": len(paired),
        "contract": {
            "same_checkpoint": True,
            "same_questions": True,
            "same_frozen_planner_outputs": True,
            "only_changed_variable": "DISABLE_QUERY_REWRITE",
            "with_rewrite_value": False,
            "substitution_only_value": True,
        },
        "with_rewrite": {
            "em": sum(row["rewrite_em"] for row in paired) / len(paired),
            "f1": sum(row["rewrite_f1"] for row in paired) / len(paired),
            "query_rewrite_events": sum(row["rewrite_event_count"] for row in paired),
            "retrieval_calls": sum(row["rewrite_retrieval_calls"] for row in paired),
        },
        "substitution_only": {
            "em": sum(row["substitution_em"] for row in paired) / len(paired),
            "f1": sum(row["substitution_f1"] for row in paired) / len(paired),
            "query_rewrite_events": 0,
            "retrieval_calls": sum(row["substitution_retrieval_calls"] for row in paired),
        },
        "paired_difference_rewrite_minus_substitution": {
            "em": sum(em_deltas) / len(paired),
            "em_bootstrap_95_ci": bootstrap_ci(
                em_deltas, seed=args.seed, samples=args.bootstrap_samples
            ),
            "f1": sum(f1_deltas) / len(paired),
            "f1_bootstrap_95_ci": bootstrap_ci(
                f1_deltas, seed=args.seed + 1, samples=args.bootstrap_samples
            ),
            "rewrite_only_correct": rewrite_only,
            "substitution_only_correct": direct_only,
            "em_ties": len(paired) - rewrite_only - direct_only,
            "exact_mcnemar_p": exact_mcnemar_p(rewrite_only, direct_only),
            "answer_changed": sum(row["answer_changed"] for row in paired),
            "questions_with_changed_executed_query": sum(
                row["changed_executed_query_nodes"] > 0 for row in paired
            ),
        },
        "outcome_quadrants": dict(
            Counter(
                "rewrite_only" if row["em_delta"] > 0 else
                "substitution_only" if row["em_delta"] < 0 else
                "both_correct" if row["rewrite_em"] else "both_wrong"
                for row in paired
            )
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "paired_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in paired),
        encoding="utf-8",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
