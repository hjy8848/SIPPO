#!/usr/bin/env python3
"""Audit strict EM-then-F1 advantages on current-policy four-sibling rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.audit_dagent_lexicographic_f1_reward import audit_groups


DEFAULT_DATASETS = {
    "2wikimultihopqa": 32,
    "hotpotqa": 32,
    "musique": 32,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_assignment(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=COUNT, got {value!r}")
    name, raw_count = value.split("=", 1)
    count = int(raw_count)
    if not name.strip() or count <= 0:
        raise argparse.ArgumentTypeError(f"invalid dataset assignment: {value!r}")
    return name.strip(), count


def _is_error(value: Any) -> bool:
    return value not in (None, "", False)


def load_rollout_groups(
    root: Path,
    run_name: str,
    *,
    datasets: Mapping[str, int] = DEFAULT_DATASETS,
    rollout_n: int = 4,
    protocol_version: str = "cp-dagent-v2-no-declared-out",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    versions: Counter[str] = Counter()
    query_rewrite_events = 0
    pipeline_errors = 0
    api_errors = 0
    execution_errors = 0

    for dataset, expected_questions in datasets.items():
        path = root / f"full_logs_{run_name}_{dataset}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_rows = expected_questions * rollout_n
        if not isinstance(payload, list) or len(payload) != expected_rows:
            raise RuntimeError(
                f"unexpected row count for {dataset}: "
                f"{len(payload) if isinstance(payload, list) else 'non-list'}/{expected_rows}"
            )
        files[dataset] = {"path": str(path), "sha256": sha256(path), "rows": len(payload)}
        by_question: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
        for position, row in enumerate(payload):
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object row {position} in {path}")
            sample_index = str(row.get("sample_index"))
            question = str(row.get("question") or "").strip()
            gold = json.dumps(row.get("gold_answer"), ensure_ascii=False, sort_keys=True)
            if not question:
                raise RuntimeError(f"empty question at row {position} in {path}")
            rollout_index = int(row.get("rollout_index", -1))
            if not 0 <= rollout_index < rollout_n:
                raise RuntimeError(f"invalid rollout_index at row {position} in {path}")
            key = (sample_index, question, gold)
            if rollout_index in by_question[key]:
                raise RuntimeError(f"duplicate rollout {rollout_index} for {key!r}")
            for metric in ("em", "f1"):
                value = float(row.get(metric, math.nan))
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise RuntimeError(f"invalid {metric} at row {position} in {path}")
            if float(row["em"]) not in (0.0, 1.0):
                raise RuntimeError(f"non-binary EM at row {position} in {path}")

            trace = row.get("raw_trace") or {}
            versions[str((trace.get("contract_protocol") or {}).get("version") or "<missing>")] += 1
            query_rewrite_events += sum(
                (event or {}).get("agent_name") == "QueryRewriteAgent"
                for event in (trace.get("sft_events") or [])
            )
            pipeline_errors += int(_is_error(row.get("pipeline_error")))
            api_errors += int(_is_error(row.get("error")))
            execution_errors += int(bool(trace.get("execution_error")))
            by_question[key][rollout_index] = row

        if len(by_question) != expected_questions:
            raise RuntimeError(
                f"unexpected question count for {dataset}: {len(by_question)}/{expected_questions}"
            )
        for (sample_index, question, gold), siblings in by_question.items():
            if set(siblings) != set(range(rollout_n)):
                raise RuntimeError(f"incomplete sibling set for {dataset}/{sample_index}")
            groups.append(
                {
                    "dataset": dataset,
                    "sample_index": sample_index,
                    "question": question,
                    "gold_answer": gold,
                    "models": {
                        f"rollout_{index}": {
                            "prediction": str(siblings[index].get("pred_answer") or ""),
                            "em": float(siblings[index]["em"]),
                            "f1": float(siblings[index]["f1"]),
                        }
                        for index in range(rollout_n)
                    },
                }
            )

    expected_rows = sum(datasets.values()) * rollout_n
    contract = {
        "expected_rows": expected_rows,
        "protocol_version_expected": protocol_version,
        "protocol_versions_observed": dict(sorted(versions.items())),
        "query_rewrite_events": query_rewrite_events,
        "pipeline_errors": pipeline_errors,
        "api_errors": api_errors,
        "execution_error_rows": execution_errors,
        "passed": bool(
            versions == Counter({protocol_version: expected_rows})
            and query_rewrite_events == 0
            and pipeline_errors == 0
            and api_errors == 0
        ),
    }
    return groups, {"files": files, "contract": contract}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--protocol-version", default="cp-dagent-v2-no-declared-out")
    parser.add_argument("--em-weight", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    datasets = dict(parse_assignment(value) for value in args.datasets) if args.datasets else dict(DEFAULT_DATASETS)
    groups, evidence = load_rollout_groups(
        args.trajectory_root,
        args.run_name,
        datasets=datasets,
        rollout_n=args.rollout_n,
        protocol_version=args.protocol_version,
    )
    labels = tuple(f"rollout_{index}" for index in range(args.rollout_n))
    reward_audit = audit_groups(
        groups,
        models=labels,
        em_weight=args.em_weight,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        minimum_new_signal_groups=5,
        minimum_datasets_with_new_signal=2,
        new_signal_rate_ci_low_must_exceed=0.0,
    )
    passed = bool(evidence["contract"]["passed"] and reward_audit["passed"])
    report = {
        "schema_version": 1,
        "method": "strict_groupwise_em_then_f1_current_policy_rollout_audit_v1",
        "run_name": args.run_name,
        "datasets": datasets,
        "rollout_n": args.rollout_n,
        **evidence,
        "reward_audit": reward_audit,
        "scientific_scope": {
            "current_policy_sibling_evidence": True,
            "terminal_outcome_reward": True,
            "process_or_topology_reward": False,
            "optimizer_updates": 0,
            "one_batch_gradient_check": False,
            "short_training": False,
        },
        "passed": passed,
        "advance_to_one_batch_gradient_check": passed,
        "interpretation": (
            "Passing this rollout-only gate authorizes one matched one-batch causal "
            "gradient check. It does not authorize a short training run."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
