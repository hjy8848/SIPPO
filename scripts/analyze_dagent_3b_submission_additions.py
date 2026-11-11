#!/usr/bin/env python3
"""Compute matched outcome and inference-cost statistics for DAGENT formal runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


DATASET_SIZES = {
    "2wikimultihopqa": 500,
    "hotpotqa": 500,
    "musique": 500,
    "bamboogle": 125,
}
METRICS = ("em", "f1")
PLAN_NODE_RE = re.compile(
    r"<(search|reason)\s+id=[\"'](\d+)[\"']>(.*?)</\1>", re.IGNORECASE | re.DOTALL
)
DEPENDENCY_RE = re.compile(r"#(\d+)#")


def question_gold_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("question", "")).strip(),
        json.dumps(row.get("gold_answer"), ensure_ascii=False, sort_keys=True),
    )


def infer_dataset(path: Path) -> str:
    lower = path.name.casefold()
    matches = [name for name in DATASET_SIZES if name in lower]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot infer one dataset from {path}")
    return matches[0]


def load_arm(root: Path) -> dict[str, list[dict[str, Any]]]:
    trajectory_root = root / "trajectories" if (root / "trajectories").is_dir() else root
    files = sorted(trajectory_root.glob("full_logs_*.json"))
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for path in files:
        dataset = infer_dataset(path)
        if dataset in by_dataset:
            raise RuntimeError(f"Multiple files found for {dataset} under {trajectory_root}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != DATASET_SIZES[dataset]:
            raise RuntimeError(
                f"Unexpected row count for {dataset}: {len(payload)}/{DATASET_SIZES[dataset]}"
            )
        seen: set[tuple[str, str]] = set()
        for row in payload:
            key = question_gold_key(row)
            if key in seen:
                raise RuntimeError(f"Duplicate question+gold in {path}: {key[0]}")
            seen.add(key)
            if row.get("pipeline_error") not in (None, "", False):
                raise RuntimeError(f"Pipeline error in {path}: {key[0]}")
            for metric in METRICS:
                value = float(row.get(metric, math.nan))
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {metric} in {path}: {key[0]}")
        by_dataset[dataset] = payload
    if set(by_dataset) != set(DATASET_SIZES):
        raise RuntimeError(f"Missing datasets under {trajectory_root}: {set(DATASET_SIZES) - set(by_dataset)}")
    return by_dataset


def align_arms(
    pure: Mapping[str, Sequence[Mapping[str, Any]]],
    psc: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    aligned: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for dataset in DATASET_SIZES:
        right = {question_gold_key(row): row for row in psc[dataset]}
        left_keys = {question_gold_key(row) for row in pure[dataset]}
        if left_keys != set(right):
            raise RuntimeError(f"question+gold mismatch for {dataset}")
        aligned[dataset] = [(row, right[question_gold_key(row)]) for row in pure[dataset]]
    return aligned


def bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(samples, start + 512)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def metric_report(
    aligned: Mapping[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"per_dataset": {}}
    all_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for dataset, pairs in aligned.items():
        all_pairs.extend(pairs)
        dataset_report: dict[str, Any] = {"samples": len(pairs)}
        for metric in METRICS:
            left = [float(pair[0][metric]) for pair in pairs]
            right = [float(pair[1][metric]) for pair in pairs]
            differences = [b - a for a, b in zip(left, right)]
            dataset_report[metric] = {
                "pure": statistics.fmean(left),
                "psc": statistics.fmean(right),
                "delta": statistics.fmean(differences),
                "paired_bootstrap_95ci": bootstrap_mean_ci(
                    differences, bootstrap_samples, seed + len(all_pairs) + len(metric)
                ),
            }
        report["per_dataset"][dataset] = dataset_report

    report["micro"] = {"samples": len(all_pairs)}
    for metric in METRICS:
        left = [float(pair[0][metric]) for pair in all_pairs]
        right = [float(pair[1][metric]) for pair in all_pairs]
        differences = [b - a for a, b in zip(left, right)]
        report["micro"][metric] = {
            "pure": statistics.fmean(left),
            "psc": statistics.fmean(right),
            "delta": statistics.fmean(differences),
            "paired_bootstrap_95ci": bootstrap_mean_ci(
                differences, bootstrap_samples, seed + 1000 + len(metric)
            ),
        }
    report["macro"] = {}
    for metric in METRICS:
        pure_values = [report["per_dataset"][name][metric]["pure"] for name in DATASET_SIZES]
        psc_values = [report["per_dataset"][name][metric]["psc"] for name in DATASET_SIZES]
        report["macro"][metric] = {
            "pure": statistics.fmean(pure_values),
            "psc": statistics.fmean(psc_values),
            "delta": statistics.fmean(psc_values) - statistics.fmean(pure_values),
        }

    quadrants = Counter()
    for left, right in all_pairs:
        left_correct = float(left["em"]) == 1.0
        right_correct = float(right["em"]) == 1.0
        name = (
            "both_correct" if left_correct and right_correct else
            "pure_only" if left_correct else
            "psc_only" if right_correct else
            "both_wrong"
        )
        quadrants[name] += 1
    report["em_quadrants"] = dict(quadrants)
    report["mcnemar"] = {
        "pure_only": quadrants["pure_only"],
        "psc_only": quadrants["psc_only"],
        "exact_two_sided_p": exact_mcnemar_pvalue(
            quadrants["pure_only"], quadrants["psc_only"]
        ),
    }
    return report


def arm_metric_report(arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"per_dataset": {}}
    all_rows: list[Mapping[str, Any]] = []
    for dataset in DATASET_SIZES:
        rows = list(arm[dataset])
        all_rows.extend(rows)
        report["per_dataset"][dataset] = {
            "samples": len(rows),
            **{metric: statistics.fmean(float(row[metric]) for row in rows) for metric in METRICS},
        }
    report["micro"] = {
        "samples": len(all_rows),
        **{metric: statistics.fmean(float(row[metric]) for row in all_rows) for metric in METRICS},
    }
    report["macro"] = {
        metric: statistics.fmean(
            report["per_dataset"][dataset][metric] for dataset in DATASET_SIZES
        )
        for metric in METRICS
    }
    return report


def parse_plan_nodes(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = str(trace.get("plan_output") or trace.get("raw_plan_output") or "")
    nodes = []
    for kind, node_id, body in PLAN_NODE_RE.findall(text):
        nodes.append(
            {
                "type": kind.casefold(),
                "id": int(node_id),
                "dependencies": [int(value) for value in DEPENDENCY_RE.findall(body)],
            }
        )
    return nodes


def graph_depth(nodes: Sequence[Mapping[str, Any]]) -> int:
    depths: dict[int, int] = {}
    for node in sorted(nodes, key=lambda item: int(item["id"])):
        dependencies = [depths.get(int(dep), 0) for dep in node["dependencies"]]
        depths[int(node["id"])] = 1 + (max(dependencies) if dependencies else 0)
    return max(depths.values(), default=0)


def make_token_counter(tokenizer_json: Path | None) -> tuple[Callable[[str], int] | None, str]:
    if tokenizer_json is None:
        return None, "not_requested"
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("Exact token counting requires the 'tokenizers' package") from error
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids), str(tokenizer_json.resolve())


def describe(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(ordered),
        "sum": math.fsum(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def compute_row(row: Mapping[str, Any], token_counter: Callable[[str], int] | None) -> dict[str, Any]:
    trace = row.get("raw_trace") or {}
    plan_nodes = parse_plan_nodes(trace)
    executed_nodes = trace.get("nodes_execution") or []
    events = trace.get("sft_events") or []
    role_calls = Counter(str(event.get("agent_name") or "unknown") for event in events)
    role_chars = Counter()
    role_tokens = Counter()
    for event in events:
        role = str(event.get("agent_name") or "unknown")
        response = str(event.get("response") or "")
        role_chars[role] += len(response)
        if token_counter is not None:
            role_tokens[role] += token_counter(response)

    attempted_extra = 0
    rewrite_queries = 0
    for node in executed_nodes:
        attempted = node.get("attempted_queries") or []
        attempted_extra += max(0, len(attempted) - 1)
        rewrite_queries += len(node.get("rewrite_queries") or [])
    answer_retries = sum(
        int((event.get("metadata") or {}).get("retry", 0) or 0) > 0
        for event in events
        if event.get("agent_name") == "AnswerGenerationAgent"
    )
    final_retries = sum(
        int((event.get("metadata") or {}).get("attempt", 0) or 0) > 0
        for event in events
        if event.get("agent_name") == "AnswerSummarizationAgent"
    )
    return {
        "planned_nodes": len(plan_nodes),
        "planned_search_nodes": sum(node["type"] == "search" for node in plan_nodes),
        "planned_reason_nodes": sum(node["type"] == "reason" for node in plan_nodes),
        "graph_depth": graph_depth(plan_nodes),
        "executed_nodes": len(executed_nodes),
        "executed_search_nodes": sum(str(node.get("type", "")).casefold() == "search" for node in executed_nodes),
        "executed_reason_nodes": sum(str(node.get("type", "")).casefold() == "reason" for node in executed_nodes),
        "retrieval_calls": int(row.get("retrieval_calls") or 0),
        "model_calls": len(events),
        "role_calls": role_calls,
        "role_output_chars": role_chars,
        "role_output_tokens": role_tokens,
        "planning_repairs": role_calls["PlanningRepairAgent"],
        "query_rewrites": role_calls["QueryRewriteAgent"],
        "retrieval_retry_queries": attempted_extra,
        "recorded_rewrite_queries": rewrite_queries,
        "answer_generation_retries": answer_retries,
        "final_answer_retries": final_retries,
        "plan_warnings": len(trace.get("plan_warnings") or []),
        "row_error": int(row.get("error") not in (None, "", False)),
    }


def compute_report(
    arm: Mapping[str, Sequence[Mapping[str, Any]]],
    token_counter: Callable[[str], int] | None,
) -> dict[str, Any]:
    scalar_names = (
        "planned_nodes", "planned_search_nodes", "planned_reason_nodes", "graph_depth",
        "executed_nodes", "executed_search_nodes", "executed_reason_nodes", "retrieval_calls",
        "model_calls", "planning_repairs", "query_rewrites", "retrieval_retry_queries",
        "recorded_rewrite_queries", "answer_generation_retries", "final_answer_retries",
        "plan_warnings", "row_error",
    )
    output: dict[str, Any] = {"per_dataset": {}}
    all_rows: list[dict[str, Any]] = []
    for dataset, rows in arm.items():
        computed = [compute_row(row, token_counter) for row in rows]
        all_rows.extend(computed)
        output["per_dataset"][dataset] = aggregate_compute(computed, scalar_names)
    output["overall"] = aggregate_compute(all_rows, scalar_names)
    return output


def aggregate_compute(rows: Sequence[Mapping[str, Any]], scalar_names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {name: describe([row[name] for row in rows]) for name in scalar_names}
    for field in ("role_calls", "role_output_chars", "role_output_tokens"):
        roles: defaultdict[str, list[float]] = defaultdict(list)
        all_role_names = sorted({role for row in rows for role in row[field]})
        for role in all_role_names:
            roles[role] = [float(row[field].get(role, 0)) for row in rows]
        result[field] = {role: describe(values) for role, values in roles.items()}
    return result


def markdown_report(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    lines = ["# Qwen2.5-3B submission additions", "", "## Outcome metrics", ""]
    if "sft" in report:
        lines.append("| Scope | SFT EM | Pure EM | PSC EM | SFT F1 | Pure F1 | PSC F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| Scope | Pure EM | PSC EM | Delta | Pure F1 | PSC F1 | Delta |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASET_SIZES:
        row = metrics["per_dataset"][dataset]
        if "sft" in report:
            sft = report["sft"]["metrics"]["per_dataset"][dataset]
            lines.append(
                f"| {dataset} | {sft['em']:.4f} | {row['em']['pure']:.4f} | {row['em']['psc']:.4f} | "
                f"{sft['f1']:.4f} | {row['f1']['pure']:.4f} | {row['f1']['psc']:.4f} |"
            )
        else:
            lines.append(
                f"| {dataset} | {row['em']['pure']:.4f} | {row['em']['psc']:.4f} | {row['em']['delta']:+.4f} | "
                f"{row['f1']['pure']:.4f} | {row['f1']['psc']:.4f} | {row['f1']['delta']:+.4f} |"
            )
    for scope in ("micro", "macro"):
        row = metrics[scope]
        if "sft" in report:
            sft = report["sft"]["metrics"][scope]
            lines.append(
                f"| {scope} | {sft['em']:.4f} | {row['em']['pure']:.4f} | {row['em']['psc']:.4f} | "
                f"{sft['f1']:.4f} | {row['f1']['pure']:.4f} | {row['f1']['psc']:.4f} |"
            )
        else:
            lines.append(
                f"| {scope} | {row['em']['pure']:.4f} | {row['em']['psc']:.4f} | {row['em']['delta']:+.4f} | "
                f"{row['f1']['pure']:.4f} | {row['f1']['psc']:.4f} | {row['f1']['delta']:+.4f} |"
            )
    lines.extend(["", "## Compute accounting", ""])
    lines.append("| Arm | Model calls | Output tokens | Retrieval calls | Executed nodes | Reason nodes | Graph depth |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in (("sft",) if "sft" in report else ()) + ("pure", "psc"):
        row = report["compute"][arm]["overall"]
        tokens = sum(value["mean"] for value in row["role_output_tokens"].values())
        lines.append(
            f"| {arm} | {row['model_calls']['mean']:.3f} | {tokens:.3f} | "
            f"{row['retrieval_calls']['mean']:.3f} | {row['executed_nodes']['mean']:.3f} | "
            f"{row['executed_reason_nodes']['mean']:.3f} | {row['graph_depth']['mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure-root", type=Path, required=True)
    parser.add_argument("--psc-root", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    pure = load_arm(args.pure_root)
    psc = load_arm(args.psc_root)
    aligned = align_arms(pure, psc)
    token_counter, tokenizer_source = make_token_counter(args.tokenizer_json)
    report = {
        "contract": {
            "alignment": "unique exact question + JSON-canonical gold answer",
            "dataset_sizes": DATASET_SIZES,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "tokenizer": tokenizer_source,
            "output_tokens_include": "all model-generated sft_event.response strings; environment context excluded",
        },
        "metrics": metric_report(
            aligned, bootstrap_samples=args.bootstrap_samples, seed=args.seed
        ),
        "compute": {
            "pure": compute_report(pure, token_counter),
            "psc": compute_report(psc, token_counter),
        },
    }
    if args.sft_root is not None:
        sft = load_arm(args.sft_root)
        report["sft"] = {
            "metrics": arm_metric_report(sft),
            "paired_to_pure": {
                "left": "sft",
                "right": "pure",
                "right_minus_left": metric_report(
                    align_arms(sft, pure),
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 2000,
                ),
            },
            "paired_to_psc": {
                "left": "sft",
                "right": "psc",
                "right_minus_left": metric_report(
                    align_arms(sft, psc),
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 3000,
                ),
            },
        }
        report["compute"]["sft"] = compute_report(sft, token_counter)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
