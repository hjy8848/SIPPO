#!/usr/bin/env python3
"""Audit where PSC changes DAGENT workflow behavior on paired Train1000 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


NODE_RE = re.compile(r'<(search|reason)\s+id="(\d+)">(.*?)</\1>', re.I | re.S)
REF_RE = re.compile(r"#(\d+)#")
METRICS = ("em", "f1")


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def is_error(value: Any) -> bool:
    return value not in (None, "", False, [], {})


def parse_plan(plan: str) -> list[tuple[str, str, str]]:
    return [(kind.lower(), node_id, body.strip()) for kind, node_id, body in NODE_RE.findall(plan)]


def dependency_violations(nodes: Sequence[tuple[str, str, str]]) -> list[str]:
    ids = [node_id for _, node_id, _ in nodes]
    violations: list[str] = []
    for position, (_, node_id, body) in enumerate(nodes):
        available = set(ids[:position])
        for reference in REF_RE.findall(body):
            if reference == node_id:
                violations.append(f"self:{node_id}")
            elif reference not in available:
                category = "future" if reference in ids[position + 1 :] else "missing"
                violations.append(f"{category}:{node_id}->{reference}")
    return violations


def reason_search_edges(nodes: Sequence[tuple[str, str, str]]) -> list[tuple[str, str]]:
    reason_ids = {node_id for kind, node_id, _ in nodes if kind == "reason"}
    return [
        (reference, node_id)
        for kind, node_id, body in nodes
        if kind == "search"
        for reference in REF_RE.findall(body)
        if reference in reason_ids
    ]


def executed_reason_search_edges(nodes: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Find executed searches whose raw query consumes an earlier reason node."""

    reason_ids: set[str] = set()
    edges: list[tuple[str, str]] = []
    for node in nodes:
        node_type = str(node.get("type", "")).lower()
        node_id = str(node.get("id", ""))
        if node_type == "reason" and node_id:
            reason_ids.add(node_id)
            continue
        if node_type != "search":
            continue
        raw_query = str(node.get("query") or "")
        edges.extend(
            (reason_id, node_id)
            for reason_id in reason_ids
            if f"#{reason_id}#" in raw_query
        )
    return edges


def locate_trajectory(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted((path / "trajectories").glob("full_logs_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one trajectory JSON under {path}, found {matches}")
    return matches[0]


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = normalize(row.get("question"))
            if not key or key in rows:
                raise RuntimeError(f"Invalid or duplicate metadata question at line {line_number}")
            blueprint = row.get("blueprint") or {}
            downstream = blueprint.get("downstream") or []
            rows[key] = {
                "sample_id": str(row.get("trajectory_id", "")),
                "question": str(row.get("question", "")),
                "reasoning_mode": str((row.get("reasoning") or {}).get("mode", "unknown")),
                "operation": str((row.get("reasoning") or {}).get("operation", "unknown")),
                "relation": str(downstream[0].get("relation", "unknown")) if downstream else "unknown",
                "winner": str(blueprint.get("selected_candidate") or ""),
                "candidate_a": str(blueprint.get("candidate_a") or ""),
                "candidate_b": str(blueprint.get("candidate_b") or ""),
            }
    if not rows:
        raise RuntimeError(f"No metadata rows in {path}")
    return rows


def extract_record(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    trace = row.get("raw_trace") or {}
    plan = str(trace.get("plan_output") or trace.get("raw_plan_output") or "")
    planned = parse_plan(plan)
    planned_edges = reason_search_edges(planned)
    executed_nodes = list(trace.get("nodes_execution") or [])
    executed_edges = executed_reason_search_edges(executed_nodes)
    executed_types = [str(node.get("type", "")).lower() for node in executed_nodes]
    reason_seen = False
    executed_search_after_reason = False
    for node_type in executed_types:
        if node_type == "reason":
            reason_seen = True
        elif node_type == "search" and reason_seen:
            executed_search_after_reason = True
    violations = dependency_violations(planned)
    warnings = list(trace.get("plan_warnings") or [])
    winner_key = normalize(metadata.get("winner"))
    candidate_keys = {
        normalize(metadata.get("candidate_a")),
        normalize(metadata.get("candidate_b")),
    } - {""}

    reason_conclusions = [
        str(node.get("conclusion") or node.get("raw_output") or "")
        for node in executed_nodes
        if str(node.get("type", "")).lower() == "reason"
    ]
    downstream_queries = [
        str(node.get("resolved_query") or node.get("executed_query") or node.get("query") or "")
        for node in executed_nodes
        if any(str(node.get("id")) == search_id for _, search_id in executed_edges)
    ]
    execution_error = trace.get("execution_error") or row.get("execution_error")
    pipeline_error = row.get("pipeline_error") or row.get("error")
    return {
        **metadata,
        "sample_index": str(row.get("sample_index", "")),
        "em": float(row.get("em", 0.0) or 0.0),
        "f1": float(row.get("f1", 0.0) or 0.0),
        "planned_topology": "-".join(kind for kind, _, _ in planned) or "none",
        "executed_topology": "-".join(executed_types) or "none",
        "planned_reason": any(kind == "reason" for kind, _, _ in planned),
        "executed_reason": "reason" in executed_types,
        "planned_reason_to_search": bool(planned_edges),
        "executed_reason_to_search": bool(executed_edges),
        "executed_search_after_reason": executed_search_after_reason,
        "planned_exact_topology": [kind for kind, _, _ in planned] == ["search", "search", "reason", "search"],
        "executed_exact_topology": executed_types == ["search", "search", "reason", "search"],
        "dependency_violation": bool(violations),
        "dependency_violations": ";".join(violations),
        "plan_warning": bool(warnings),
        "plan_warnings": json.dumps(warnings, ensure_ascii=False),
        "execution_error": is_error(execution_error),
        "execution_error_text": str(execution_error or ""),
        "pipeline_error": is_error(pipeline_error),
        "reason_selects_winner": bool(
            winner_key and any(winner_key in normalize(value) for value in reason_conclusions)
        ),
        "reason_mentions_candidate": any(
            candidate in normalize(value)
            for candidate in candidate_keys
            for value in reason_conclusions
        ),
        "downstream_query_contains_winner": bool(
            winner_key and any(winner_key in normalize(value) for value in downstream_queries)
        ),
        "downstream_query_mentions_candidate": any(
            candidate in normalize(value)
            for candidate in candidate_keys
            for value in downstream_queries
        ),
        "downstream_queries": " || ".join(downstream_queries),
        "reason_conclusions": " || ".join(reason_conclusions),
        "plan_output": plan,
        "pred_answer": str(row.get("pred_answer", "")),
        "gold_answer": json.dumps(row.get("gold_answer") or [], ensure_ascii=False),
    }


def load_run(path: Path, metadata: Mapping[str, Mapping[str, Any]]) -> tuple[Path, dict[str, dict[str, Any]]]:
    trajectory = locate_trajectory(path)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {trajectory}")
    records: dict[str, dict[str, Any]] = {}
    for row in payload:
        key = normalize(row.get("question"))
        if key not in metadata or key in records:
            raise RuntimeError(f"Unmatched or duplicate evaluation question: {row.get('question')}")
        records[key] = extract_record(row, metadata[key])
    if set(records) != set(metadata):
        raise RuntimeError(f"Run/metadata mismatch: {len(records)}/{len(metadata)}")
    return trajectory.resolve(), records


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return (seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(start + 512, samples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def mcnemar_pvalue(candidate_only: int, reference_only: int) -> float:
    discordant = candidate_only + reference_only
    if not discordant:
        return 1.0
    tail = min(candidate_only, reference_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / 2**discordant)


def summarize_run(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    boolean_fields = (
        "planned_reason", "executed_reason", "planned_reason_to_search",
        "executed_reason_to_search", "executed_search_after_reason",
        "planned_exact_topology", "executed_exact_topology",
        "dependency_violation", "plan_warning", "execution_error", "pipeline_error",
        "reason_selects_winner", "reason_mentions_candidate",
        "downstream_query_contains_winner", "downstream_query_mentions_candidate",
    )
    return {
        "samples": len(rows),
        "em": statistics.fmean(float(row["em"]) for row in rows),
        "f1": statistics.fmean(float(row["f1"]) for row in rows),
        **{field: sum(bool(row[field]) for row in rows) for field in boolean_fields},
        "planned_topologies": dict(Counter(str(row["planned_topology"]) for row in rows)),
        "executed_topologies": dict(Counter(str(row["executed_topology"]) for row in rows)),
    }


def paired_summary(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
    *,
    samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {"samples": len(keys)}
    for metric in METRICS:
        differences = [float(candidate[key][metric]) - float(reference[key][metric]) for key in keys]
        report[metric] = {
            "reference": statistics.fmean(float(reference[key][metric]) for key in keys),
            "candidate": statistics.fmean(float(candidate[key][metric]) for key in keys),
            "delta": statistics.fmean(differences),
            "bootstrap_95ci": bootstrap_ci(
                differences, samples, stable_seed(seed, label, metric)
            ),
        }
    candidate_only = sum(candidate[key]["em"] == 1.0 and reference[key]["em"] == 0.0 for key in keys)
    reference_only = sum(reference[key]["em"] == 1.0 and candidate[key]["em"] == 0.0 for key in keys)
    report["mcnemar"] = {
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "pvalue": mcnemar_pvalue(candidate_only, reference_only),
    }
    return report


def behavior_slices(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    groups = {
        "candidate_executed_reason_to_search": [key for key, row in candidate.items() if row["executed_reason_to_search"]],
        "candidate_no_executed_reason_to_search": [key for key, row in candidate.items() if not row["executed_reason_to_search"]],
        "candidate_downstream_query_contains_winner": [key for key, row in candidate.items() if row["downstream_query_contains_winner"]],
        "candidate_reason_mentions_candidate": [key for key, row in candidate.items() if row["reason_mentions_candidate"]],
        "candidate_downstream_query_mentions_candidate": [key for key, row in candidate.items() if row["downstream_query_mentions_candidate"]],
        "candidate_execution_error": [key for key, row in candidate.items() if row["execution_error"]],
    }
    return {
        label: paired_summary(reference, candidate, keys, samples=samples, seed=seed, label=label)
        for label, keys in groups.items()
        if keys
    }


def select_cases(
    reference: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for key in sorted(reference):
        left, right = reference[key], candidate[key]
        category = None
        if right["executed_reason_to_search"] and right["em"] == 1.0 and left["em"] == 0.0:
            category = "reason_to_search_candidate_only_correct"
        elif right["executed_reason_to_search"] and right["em"] == 0.0:
            category = "reason_to_search_candidate_wrong"
        elif not right["executed_reason_to_search"] and left["em"] == 1.0 and right["em"] == 0.0:
            category = "no_reason_to_search_reference_only_correct"
        if category:
            cases.append({
                "category": category,
                "sample_id": right["sample_id"],
                "question": right["question"],
                "operation": right["operation"],
                "winner": right["winner"],
                "reference_em": left["em"],
                "candidate_em": right["em"],
                "candidate_reason_selects_winner": right["reason_selects_winner"],
                "candidate_query_contains_winner": right["downstream_query_contains_winner"],
                "candidate_execution_error": right["execution_error_text"],
                "candidate_downstream_queries": right["downstream_queries"],
                "candidate_reason_conclusions": right["reason_conclusions"],
                "candidate_plan": right["plan_output"],
                "reference_plan": left["plan_output"],
            })
    return cases


def render_markdown(report: Mapping[str, Any]) -> str:
    labels = report["labels"]
    lines = [
        "# PSC Failure-Mode Audit",
        "",
        "Behavior-conditioned rows are post-treatment associations, not causal effects.",
        "",
        "## Overall",
        "",
        "| Slice | n | Reference EM | Candidate EM | Delta EM | 95% CI | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["paired_slices"].items():
        em, mc = row["em"], row["mcnemar"]
        lines.append(
            f"| {name} | {row['samples']} | {em['reference']:.6f} | {em['candidate']:.6f} | "
            f"{em['delta']:+.6f} | [{em['bootstrap_95ci'][0]:+.6f}, {em['bootstrap_95ci'][1]:+.6f}] | "
            f"{mc['pvalue']:.6g} |"
        )
    lines.extend(["", "## Workflow Coverage", "", "| Model | Planned R->S | Executed dependent R->S | Any Search after Reason | Exact executed | Dependency violations | Execution errors |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label in (labels["reference"], labels["candidate"]):
        row = report["runs"][label]["summary"]
        lines.append(
            f"| {label} | {row['planned_reason_to_search']} | {row['executed_reason_to_search']} | "
            f"{row['executed_search_after_reason']} | {row['executed_exact_topology']} | "
            f"{row['dependency_violation']} | {row['execution_error']} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, report: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "psc_failure_modes.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "psc_failure_modes.md").write_text(render_markdown(report), encoding="utf-8")
    if cases:
        with (output_dir / "psc_cases.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-label", default="PureEM")
    parser.add_argument("--candidate-label", default="PSC")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")

    metadata = load_metadata(args.metadata)
    reference_path, reference = load_run(args.reference, metadata)
    candidate_path, candidate = load_run(args.candidate, metadata)
    all_keys = sorted(metadata)
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in all_keys:
        grouped[candidate[key]["operation"]].append(key)
    behavior_by_operation = {
        f"operation:{operation}:candidate_executed_reason_to_search": [
            key for key in keys if candidate[key]["executed_reason_to_search"]
        ]
        for operation, keys in sorted(grouped.items())
    }
    paired_slices = {
        "all": paired_summary(reference, candidate, all_keys, samples=args.bootstrap_samples, seed=args.seed, label="all"),
        **{
            f"operation:{operation}": paired_summary(
                reference, candidate, keys, samples=args.bootstrap_samples, seed=args.seed, label=operation
            )
            for operation, keys in sorted(grouped.items())
        },
        **behavior_slices(reference, candidate, samples=args.bootstrap_samples, seed=args.seed),
        **{
            label: paired_summary(
                reference, candidate, keys, samples=args.bootstrap_samples, seed=args.seed, label=label
            )
            for label, keys in behavior_by_operation.items()
            if keys
        },
    }
    cases = select_cases(reference, candidate)
    report = {
        "schema_version": 1,
        "labels": {"reference": args.reference_label, "candidate": args.candidate_label},
        "metadata": str(args.metadata.resolve()),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "runs": {
            args.reference_label: {"trajectory": str(reference_path), "summary": summarize_run(reference.values())},
            args.candidate_label: {"trajectory": str(candidate_path), "summary": summarize_run(candidate.values())},
        },
        "paired_slices": paired_slices,
        "case_counts": dict(Counter(case["category"] for case in cases)),
        "interpretation_guard": "Behavior-conditioned slices are post-treatment associations, not causal effects.",
    }
    write_outputs(args.output_dir, report, cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
