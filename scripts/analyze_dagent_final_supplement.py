#!/usr/bin/env python3
"""Build reproducible supplemental analyses from paired DAGENT formal runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_DATASETS = {
    "2wikimultihopqa": 500,
    "hotpotqa": 500,
    "musique": 500,
    "bamboogle": 125,
}
METRICS = ("em", "f1")
ALIGNMENT_KEYS = ("sample_index", "question_gold")


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def normalize_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in text
    )
    return " ".join(
        token for token in text.split() if token not in {"a", "an", "the"}
    )


def locate_result_file(root: Path, dataset: str) -> Path:
    matches = sorted((root / "trajectories").glob(f"full_logs_*_{dataset}.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one trajectory file for {dataset} under {root}, found {matches}"
        )
    return matches[0]


def load_rows(path: Path, expected: int) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != expected:
        raise RuntimeError(f"Unexpected rows in {path}: {len(payload)}/{expected}")
    rows: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(payload):
        sample_index = int(row.get("sample_index", position))
        if sample_index in rows:
            raise RuntimeError(f"Duplicate sample_index={sample_index} in {path}")
        if row.get("pipeline_error") not in (None, "", False):
            raise RuntimeError(f"Pipeline error at sample_index={sample_index} in {path}")
        for metric in METRICS:
            if not math.isfinite(float(row.get(metric, math.nan))):
                raise RuntimeError(f"Invalid {metric} at sample_index={sample_index} in {path}")
        rows[sample_index] = row
    return rows


def load_run(root: Path, datasets: Mapping[str, int]) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        dataset: load_rows(locate_result_file(root, dataset), expected)
        for dataset, expected in datasets.items()
    }


def validate_alignment(
    reference: Mapping[str, Mapping[int, Mapping[str, Any]]],
    candidate: Mapping[str, Mapping[int, Mapping[str, Any]]],
    alignment_key: str = "sample_index",
) -> tuple[
    dict[str, dict[Any, Mapping[str, Any]]],
    dict[str, dict[Any, Mapping[str, Any]]],
]:
    if alignment_key not in ALIGNMENT_KEYS:
        raise ValueError(f"Unsupported alignment key: {alignment_key}")
    aligned_reference: dict[str, dict[Any, Mapping[str, Any]]] = {}
    aligned_candidate: dict[str, dict[Any, Mapping[str, Any]]] = {}
    for dataset, reference_rows in reference.items():
        candidate_rows = candidate[dataset]
        indexed_reference = index_rows(reference_rows, alignment_key, dataset, "reference")
        indexed_candidate = index_rows(candidate_rows, alignment_key, dataset, "candidate")
        if set(indexed_reference) != set(indexed_candidate):
            missing = sorted(set(indexed_reference) - set(indexed_candidate), key=str)[:10]
            extra = sorted(set(indexed_candidate) - set(indexed_reference), key=str)[:10]
            raise RuntimeError(
                f"{alignment_key} mismatch for {dataset}: missing={missing}, extra={extra}"
            )
        for key, left in indexed_reference.items():
            right = indexed_candidate[key]
            if normalize_question(left["question"]) != normalize_question(right["question"]):
                raise RuntimeError(
                    f"Question mismatch for {dataset} alignment_key={key!r}"
                )
            if left.get("gold_answer") != right.get("gold_answer"):
                raise RuntimeError(
                    f"Gold-answer mismatch for {dataset} alignment_key={key!r}"
                )
        aligned_reference[dataset] = indexed_reference
        aligned_candidate[dataset] = {
            key: indexed_candidate[key] for key in indexed_reference
        }
    return aligned_reference, aligned_candidate


def row_alignment_key(row: Mapping[str, Any], alignment_key: str) -> Any:
    if alignment_key == "sample_index":
        return int(row["sample_index"])
    if alignment_key == "question_gold":
        return (
            str(row["question"]).strip(),
            json.dumps(row.get("gold_answer"), ensure_ascii=False, sort_keys=True),
        )
    raise ValueError(f"Unsupported alignment key: {alignment_key}")


def index_rows(
    rows: Mapping[int, Mapping[str, Any]],
    alignment_key: str,
    dataset: str,
    label: str,
) -> dict[Any, Mapping[str, Any]]:
    indexed: dict[Any, Mapping[str, Any]] = {}
    for row in rows.values():
        key = row_alignment_key(row, alignment_key)
        if key in indexed:
            raise RuntimeError(
                f"Duplicate {alignment_key} for {label}/{dataset}: {key!r}"
            )
        indexed[key] = row
    return indexed


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise RuntimeError("Cannot bootstrap an empty sequence")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(samples, start + 512)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def exact_mcnemar_pvalue(candidate_only: int, reference_only: int) -> float:
    discordant = candidate_only + reference_only
    if discordant == 0:
        return 1.0
    tail = min(candidate_only, reference_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def load_2wiki_types(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    types: dict[str, str] = {}
    for row in payload:
        question = normalize_question(row["question"])
        question_type = str(row["type"])
        if question in types and types[question] != question_type:
            raise RuntimeError(f"Conflicting official type for question: {row['question']}")
        types[question] = question_type
    return types


def paired_type_analysis(
    reference_rows: Mapping[int, Mapping[str, Any]],
    candidate_rows: Mapping[int, Mapping[str, Any]],
    official_types: Mapping[str, str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for sample_index, reference in reference_rows.items():
        key = normalize_question(reference["question"])
        if key not in official_types:
            raise RuntimeError(f"No official 2Wiki type for: {reference['question']}")
        groups[official_types[key]].append((reference, candidate_rows[sample_index]))

    report: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for offset, (question_type, pairs) in enumerate(sorted(groups.items())):
        summary: dict[str, Any] = {"samples": len(pairs)}
        for metric in METRICS:
            reference_values = [float(left[metric]) for left, _ in pairs]
            candidate_values = [float(right[metric]) for _, right in pairs]
            differences = [right - left for left, right in zip(reference_values, candidate_values)]
            summary[metric] = {
                "reference": statistics.fmean(reference_values),
                "candidate": statistics.fmean(candidate_values),
                "delta": statistics.fmean(differences),
                "bootstrap_95ci": bootstrap_ci(
                    differences, bootstrap_samples, seed + offset * 17 + len(metric)
                ),
            }
        candidate_only = sum(
            float(right["em"]) == 1.0 and float(left["em"]) == 0.0
            for left, right in pairs
        )
        reference_only = sum(
            float(left["em"]) == 1.0 and float(right["em"]) == 0.0
            for left, right in pairs
        )
        pvalue = exact_mcnemar_pvalue(candidate_only, reference_only)
        summary["mcnemar"] = {
            "candidate_only": candidate_only,
            "reference_only": reference_only,
            "pvalue": pvalue,
        }
        raw_pvalues[question_type] = pvalue
        report[question_type] = summary

    comparisons = len(raw_pvalues)
    for question_type, pvalue in raw_pvalues.items():
        report[question_type]["mcnemar"]["bonferroni_pvalue"] = min(
            1.0, pvalue * comparisons
        )
    return report


def gold_in_retrieved_context(row: Mapping[str, Any]) -> bool | None:
    answers = row.get("gold_answer") or []
    if isinstance(answers, str):
        answers = [answers]
    normalized_answers = [normalize_answer(answer) for answer in answers]
    if any(answer in {"yes", "no"} for answer in normalized_answers):
        return None
    trace = row.get("raw_trace") or {}
    context = normalize_answer(
        " ".join(
            str(node.get("search_context") or "")
            for node in trace.get("nodes_execution", [])
        )
    )
    return any(answer and answer in context for answer in normalized_answers)


def coverage_analysis(rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    labeled = [(row, gold_in_retrieved_context(row)) for row in rows.values()]
    labeled = [(row, present) for row, present in labeled if present is not None]
    present_rows = [row for row, present in labeled if present]
    absent_rows = [row for row, present in labeled if not present]
    return {
        "non_binary_samples": len(labeled),
        "gold_string_coverage": len(present_rows) / len(labeled),
        "em_when_present": statistics.fmean(float(row["em"]) for row in present_rows)
        if present_rows
        else None,
        "em_when_absent": statistics.fmean(float(row["em"]) for row in absent_rows)
        if absent_rows
        else None,
    }


PLACEHOLDER_RE = re.compile(r"#\d+#")


def runtime_analysis(datasets: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    rows = [row for dataset in datasets.values() for row in dataset.values()]
    valid = 0
    referenced_nodes = 0
    resolved_nodes = 0
    unresolved_workflows = 0
    executed_nodes = 0
    retrieval_calls = 0.0
    for row in rows:
        trace = row.get("raw_trace") or {}
        warnings = trace.get("plan_warnings") or []
        if not warnings and row.get("error") in (None, "", False):
            valid += 1
        workflow_unresolved = False
        nodes = trace.get("nodes_execution") or []
        executed_nodes += len(nodes)
        retrieval_calls += float(row.get("retrieval_calls", 0.0))
        for node in nodes:
            query = str(node.get("query") or "")
            if PLACEHOLDER_RE.search(query):
                referenced_nodes += 1
                resolved_query = str(node.get("resolved_query") or "")
                if not PLACEHOLDER_RE.search(resolved_query):
                    resolved_nodes += 1
                else:
                    workflow_unresolved = True
        unresolved_workflows += int(workflow_unresolved)
    return {
        "samples": len(rows),
        "pipeline_errors": 0,
        "warning_free_plan_rate": valid / len(rows),
        "trace_dependency_resolution_rate": resolved_nodes / referenced_nodes
        if referenced_nodes
        else None,
        "referenced_nodes": referenced_nodes,
        "unresolved_placeholder_workflow_rate": unresolved_workflows / len(rows),
        "avg_executed_nodes": executed_nodes / len(rows),
        "avg_retrieval_calls": retrieval_calls / len(rows),
        "pooled_em": statistics.fmean(float(row["em"]) for row in rows),
    }


def load_analyzer_runtime_quality(root: Path, expected_samples: int) -> dict[str, Any]:
    path = root / "analysis" / "trajectory_quality.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"Expected one trajectory-quality row in {path}")
    row = payload[0]
    if int(row["num_samples"]) != expected_samples:
        raise RuntimeError(
            f"Unexpected trajectory-quality sample count in {path}: "
            f"{row['num_samples']}/{expected_samples}"
        )
    return {
        "analyzer_strict_valid_plan_rate": float(row["valid_plan"]),
        "analyzer_dependency_resolution_rate": float(row["dep_resolved"]),
        "analyzer_unresolved_id_rate": float(row["unresolved_id"]),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    labels = report["labels"]
    lines = [
        "# DAGENT Final Supplemental Analysis",
        "",
        "## Official 2Wiki Type Slices",
        "",
        f"| Type | n | {labels['reference']} EM | {labels['candidate']} EM | "
        "Delta EM | 95% CI | McNemar p | Bonferroni p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for question_type, row in report["two_wiki_types"].items():
        em = row["em"]
        mc = row["mcnemar"]
        lines.append(
            f"| {question_type} | {row['samples']} | {em['reference']:.6f} | "
            f"{em['candidate']:.6f} | {em['delta']:+.6f} | "
            f"[{em['bootstrap_95ci'][0]:+.6f}, {em['bootstrap_95ci'][1]:+.6f}] | "
            f"{mc['pvalue']:.6g} | {mc['bonferroni_pvalue']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Lexical Gold-String Coverage",
            "",
            "This is a retrieval-context containment proxy, not supporting-fact recall.",
            "",
            "| Dataset | Model | n (non-binary) | Coverage | EM if present | EM if absent |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset, models in report["coverage"].items():
        for label, row in models.items():
            lines.append(
                f"| {dataset} | {label} | {row['non_binary_samples']} | "
                f"{row['gold_string_coverage']:.6f} | {row['em_when_present']:.6f} | "
                f"{row['em_when_absent']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Runtime Reliability and Cost",
            "",
            "| Model | Samples | Strict valid | Warning-free | Dependency resolved | "
            "Avg nodes | Avg retrievals | Pooled EM |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in report["runtime"].items():
        lines.append(
            f"| {label} | {row['samples']} | {row['analyzer_strict_valid_plan_rate']:.6f} | "
            f"{row['warning_free_plan_rate']:.6f} | {row['analyzer_dependency_resolution_rate']:.6f} | "
            f"{row['avg_executed_nodes']:.6f} | "
            f"{row['avg_retrieval_calls']:.6f} | {row['pooled_em']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-label", default="PureEM")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-label", default="PSC")
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--two-wiki-raw", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alignment-key", choices=ALIGNMENT_KEYS, default="sample_index")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = load_run(args.reference_run, DEFAULT_DATASETS)
    candidate = load_run(args.candidate_run, DEFAULT_DATASETS)
    aligned_reference, aligned_candidate = validate_alignment(
        reference, candidate, args.alignment_key
    )
    report = {
        "schema_version": 1,
        "alignment_key": args.alignment_key,
        "labels": {
            "reference": args.reference_label,
            "candidate": args.candidate_label,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "two_wiki_types": paired_type_analysis(
            aligned_reference["2wikimultihopqa"],
            aligned_candidate["2wikimultihopqa"],
            load_2wiki_types(args.two_wiki_raw),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "coverage": {
            dataset: {
                args.reference_label: coverage_analysis(reference[dataset]),
                args.candidate_label: coverage_analysis(candidate[dataset]),
            }
            for dataset in DEFAULT_DATASETS
        },
        "runtime": {},
    }
    for label, root, rows in (
        (args.reference_label, args.reference_run, reference),
        (args.candidate_label, args.candidate_run, candidate),
    ):
        report["runtime"][label] = {
            **runtime_analysis(rows),
            **load_analyzer_runtime_quality(root, sum(DEFAULT_DATASETS.values())),
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "supplemental_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "supplemental_analysis.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] Supplemental analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
