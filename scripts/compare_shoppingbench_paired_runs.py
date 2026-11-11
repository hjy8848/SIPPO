"""Create a strict paired comparison for original and contract ShoppingBench runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable


TASKS = ("product", "shop", "voucher")
ERROR_KEYS = ("format_error", "contract_error", "pipeline_error", "request_error")
PROTOCOL_PAIR_IGNORE = {"arm", "contract_preserving_dag_v1"}


def load_arm(directory: Path, arm: str, expected_per_task: int) -> tuple[dict[str, dict], dict]:
    rows: dict[str, dict] = {}
    protocols = {}
    expected_contract = arm == "contract"
    for task in TASKS:
        task_dir = directory / task
        protocol = json.loads((task_dir / "evaluation_protocol.json").read_text(encoding="utf-8"))
        if protocol.get("arm") != arm:
            raise ValueError(f"{task}: expected arm={arm}, got {protocol.get('arm')}")
        if protocol.get("contract_preserving_dag_v1") is not expected_contract:
            raise ValueError(f"{task}: contract flag does not match arm={arm}")
        if int(protocol.get("selected_count", -1)) != expected_per_task:
            raise ValueError(f"{task}: selected_count does not equal {expected_per_task}")
        protocols[task] = protocol
        task_rows = []
        with (task_dir / "results.jsonl").open(encoding="utf-8") as handle:
            task_rows.extend(json.loads(line) for line in handle if line.strip())
        if len(task_rows) != expected_per_task:
            raise ValueError(f"{task}: expected {expected_per_task} rows, got {len(task_rows)}")
        for row in task_rows:
            sample_id = str(row.get("sample_id") or "")
            if not sample_id.startswith(task + "-") or sample_id in rows:
                raise ValueError(f"Invalid or duplicate sample_id: {sample_id}")
            rows[sample_id] = row
    return rows, protocols


def paired_bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty paired sample")
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    score = row.get("score") or {}
    errors = {key: row.get(key) for key in ERROR_KEYS if row.get(key)}
    return {
        "exact_id_fraction": float(score.get("exact_id_fraction", 0.0)),
        "exact_sequence": float(bool(score.get("exact_sequence_success"))),
        "first_id_exact": float(bool(score.get("first_id_exact"))),
        "valid": float(not errors),
        "execution_failure": float(bool(errors)),
        "pipeline_or_request_failure": float(
            bool(row.get("pipeline_error") or row.get("request_error"))
        ),
        "retrieval_calls": float(row.get("retrieval_calls") or 0),
        "search_nodes": float(row.get("search_nodes") or 0),
    }


def summarize_pairs(pairs: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    metrics = tuple(row_metrics(pairs[0]["original"]))
    report = {"pairs": len(pairs), "metrics": {}}
    for metric in metrics:
        original = [row_metrics(pair["original"])[metric] for pair in pairs]
        contract = [row_metrics(pair["contract"])[metric] for pair in pairs]
        deltas = [right - left for left, right in zip(original, contract)]
        report["metrics"][metric] = {
            "original_mean": sum(original) / len(original),
            "contract_mean": sum(contract) / len(contract),
            "paired_delta": sum(deltas) / len(deltas),
            "paired_bootstrap_95ci": paired_bootstrap_ci(deltas, samples, seed),
            "improved_pairs": sum(delta > 0 for delta in deltas),
            "worsened_pairs": sum(delta < 0 for delta in deltas),
            "tied_pairs": sum(delta == 0 for delta in deltas),
        }
        seed += 1
    return report


def compare_runs(
    original: dict[str, dict],
    contract: dict[str, dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(original) != set(contract):
        missing_left = sorted(set(contract) - set(original))
        missing_right = sorted(set(original) - set(contract))
        raise ValueError(f"Pair mismatch: original_missing={missing_left}, contract_missing={missing_right}")
    pairs = []
    flat_rows = []
    for sample_id in sorted(original):
        left = original[sample_id]
        right = contract[sample_id]
        for field in ("query", "expected_product_ids"):
            if left.get(field) != right.get(field):
                raise ValueError(f"{sample_id}: paired field differs: {field}")
        task = sample_id.split("-", 1)[0]
        pairs.append({"sample_id": sample_id, "task": task, "original": left, "contract": right})
        left_metrics = row_metrics(left)
        right_metrics = row_metrics(right)
        flat_rows.append({
            "sample_id": sample_id,
            "task": task,
            **{f"original_{key}": value for key, value in left_metrics.items()},
            **{f"contract_{key}": value for key, value in right_metrics.items()},
            **{f"delta_{key}": right_metrics[key] - left_metrics[key] for key in left_metrics},
        })
    report = {
        "overall": summarize_pairs(pairs, bootstrap_samples, seed),
        "by_task": {
            task: summarize_pairs(
                [pair for pair in pairs if pair["task"] == task],
                bootstrap_samples,
                seed + 100 * (index + 1),
            )
            for index, task in enumerate(TASKS)
        },
    }
    return report, flat_rows


def validate_runtime_contracts(
    original: dict[str, dict], contract: dict[str, dict]
) -> dict[str, Any]:
    mismatches = []
    path_counts = {
        "completed": 0,
        "failed_after_compilation": 0,
        "failed_during_compilation": 0,
    }
    for sample_id, row in contract.items():
        trace = row.get("raw_trace") or {}
        compiler = (trace.get("contract_protocol") or {}).get("plan_compiler") or {}
        ablation = trace.get("ablation_config") or {}
        runtime_errors = {
            key: row.get(key)
            for key in ("pipeline_error", "request_error")
            if row.get(key)
        }
        trace_error = trace.get("error") or trace.get("execution_error")
        runtime_failed = bool(runtime_errors or trace_error)

        checks = {"trace_present": bool(trace), "compiler_enabled": compiler.get("enabled") is True}
        if not runtime_failed:
            path_counts["completed"] += 1
            checks.update({
                "shopping_profile": trace.get("protocol_profile") == "shoppingbench",
                "contract_enabled": ablation.get("contract_preserving_dag_v1") is True,
                "compiler_applied": compiler.get("applied") is True,
                "compiler_used_no_gold": compiler.get("gold_used") is False,
                "plan_override_unused": trace.get("plan_override_used") is False,
            })
        else:
            # Early-return traces are intentionally smaller than completed traces. Reject
            # contradictions, then require proof that execution stopped at or after the
            # enabled compiler instead of demanding success-only metadata.
            if "protocol_profile" in trace:
                checks["shopping_profile_not_contradicted"] = (
                    trace.get("protocol_profile") == "shoppingbench"
                )
            if "contract_preserving_dag_v1" in ablation:
                checks["contract_enabled_not_contradicted"] = (
                    ablation.get("contract_preserving_dag_v1") is True
                )
            if "plan_override_used" in trace:
                checks["plan_override_unused_not_contradicted"] = (
                    trace.get("plan_override_used") is False
                )

            if compiler.get("applied") is True:
                path_counts["failed_after_compilation"] += 1
                checks["compiler_used_no_gold"] = compiler.get("gold_used") is False
            else:
                path_counts["failed_during_compilation"] += 1
                failure_text = " ".join(
                    str(value) for value in (*runtime_errors.values(), trace_error) if value
                )
                checks.update({
                    "compiler_failure_recorded": "system plan compiler failed:" in failure_text,
                    "no_nodes_executed_after_compiler_failure": not (trace.get("nodes_execution") or []),
                })
        mismatches.extend(
            f"{sample_id}: {name}" for name, passed in checks.items() if not passed
        )
    for sample_id, row in original.items():
        trace = row.get("raw_trace") or {}
        if not trace:
            continue
        ablation = trace.get("ablation_config") or {}
        compiler = (trace.get("contract_protocol") or {}).get("plan_compiler") or {}
        if ablation and ablation.get("contract_preserving_dag_v1") is not False:
            mismatches.append(f"{sample_id}: original contract flag is not false")
        if compiler.get("applied") is True:
            mismatches.append(f"{sample_id}: compiler applied in original arm")
        if "plan_override_used" in trace and trace.get("plan_override_used") is not False:
            mismatches.append(f"{sample_id}: original plan override used")
    return {
        "passed": not mismatches,
        "path_counts": path_counts,
        "mismatches": mismatches,
    }


def canonical_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in protocol.items() if key not in PROTOCOL_PAIR_IGNORE}


def validate_protocols(
    original: dict[str, dict],
    contract: dict[str, dict],
    run_protocol: dict[str, Any],
    expected_per_task: int,
) -> dict[str, Any]:
    mismatches = []
    for task in TASKS:
        if canonical_protocol(original[task]) != canonical_protocol(contract[task]):
            mismatches.append(f"{task}: arm protocols differ beyond the contract flag")
        if original[task].get("plan_override_manifest") is not None:
            mismatches.append(f"{task}: original uses a plan override")
        if contract[task].get("plan_override_manifest") is not None:
            mismatches.append(f"{task}: contract uses a plan override")
    shared = run_protocol.get("shared") or {}
    checks = {
        "sample_count_per_task": run_protocol.get("sample_count_per_task") == expected_per_task,
        "both_arms": run_protocol.get("arms") == ["original", "contract"],
        "temperature_zero": float(shared.get("temperature", -1)) == 0.0,
        "top_p_one": float(shared.get("top_p", -1)) == 1.0,
        "external_api_calls_zero": int(shared.get("external_shopping_api_calls", -1)) == 0,
        "no_gold_query_fields": int(shared.get("gold_query_fields", 0)) == 0,
        "retriever_is_local_lucene": shared.get("retriever_backend") == "local_lucene",
        "network_fallback_disabled": shared.get("network_fallback_enabled") is False,
        "planner_api_disabled": shared.get("planner_api_enabled") is False,
        "no_plan_override": run_protocol.get("plan_override_manifest") is None,
    }
    mismatches.extend(key for key, passed in checks.items() if not passed)
    return {"passed": not mismatches, "checks": checks, "mismatches": mismatches}


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--run-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-per-task", type=int, default=250)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to reuse output directory: {args.output_dir}")
    if args.bootstrap_samples < 1000:
        raise SystemExit("bootstrap-samples must be at least 1000")

    original, original_protocols = load_arm(
        args.original_dir, "original", args.expected_per_task
    )
    contract, contract_protocols = load_arm(
        args.contract_dir, "contract", args.expected_per_task
    )
    run_protocol = json.loads(args.run_protocol.read_text(encoding="utf-8"))
    protocol_gate = validate_protocols(
        original_protocols, contract_protocols, run_protocol, args.expected_per_task
    )
    report, rows = compare_runs(
        original,
        contract,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    primary = report["overall"]["metrics"]["exact_id_fraction"]
    failure = report["overall"]["metrics"]["execution_failure"]
    validity = report["overall"]["metrics"]["valid"]
    runtime_gate = validate_runtime_contracts(original, contract)
    report.update({
        "protocol_gate": protocol_gate,
        "runtime_gate": runtime_gate,
        "decision": {
            "protocol_matched": protocol_gate["passed"] and runtime_gate["passed"],
            "point_estimate_improved": primary["paired_delta"] > 0,
            "paired_ci_nonnegative": primary["paired_bootstrap_95ci"][0] >= 0,
            "execution_failures_not_increased": failure["paired_delta"] <= 0,
            "validity_not_decreased": validity["paired_delta"] >= 0,
        },
        "evidence_level": (
            "strict paired 750-example evaluation"
            if args.expected_per_task == 250
            and len(rows) == 750
            and protocol_gate["passed"]
            and runtime_gate["passed"]
            else "paired diagnostic with contract mismatch"
        ),
    })
    report["decision"]["passed"] = all(report["decision"].values())
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "paired_examples.csv", rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
