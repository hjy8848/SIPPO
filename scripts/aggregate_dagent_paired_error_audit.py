#!/usr/bin/env python3
"""Validate and aggregate a manually labelled paired error audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


TAXONOMY = {
    "correct",
    "plan_decomposition_omission",
    "wrong_relation_or_shortcut_query",
    "retrieval_miss",
    "query_rewrite_or_entity_transfer",
    "evidence_present_but_node_conclusion_wrong",
    "reasoning_comparison_or_composition",
    "final_synthesis",
    "answer_normalization_or_alias",
    "invalid_plan_or_execution_failure",
    "evidence_or_corpus_insufficiency",
    "gold_annotation_or_question_defect",
    "other",
}

STAGE_BY_CATEGORY = {
    "plan_decomposition_omission": "planning_or_grounding",
    "wrong_relation_or_shortcut_query": "planning_or_grounding",
    "retrieval_miss": "planning_or_grounding",
    "query_rewrite_or_entity_transfer": "planning_or_grounding",
    "evidence_present_but_node_conclusion_wrong": "evidence_interpretation_or_composition",
    "reasoning_comparison_or_composition": "evidence_interpretation_or_composition",
    "final_synthesis": "evidence_interpretation_or_composition",
    "answer_normalization_or_alias": "surface_evaluation",
    "invalid_plan_or_execution_failure": "execution_failure",
    "evidence_or_corpus_insufficiency": "data_or_corpus_issue",
    "gold_annotation_or_question_defect": "data_or_corpus_issue",
    "other": "other",
}


def percentage(count: float, total: float) -> float:
    return 100.0 * count / total if total else 0.0


def sorted_distribution(counter: Mapping[str, float], total: float) -> list[dict[str, Any]]:
    return [
        {"category": category, "count": count, "percent": percentage(count, total)}
        for category, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("Every label row must have a case_id")
        if case_id in labels:
            raise ValueError(f"Duplicate label row: {case_id}")
        labels[case_id] = {key: str(value or "").strip() for key, value in row.items()}
    return labels


def validate(cases: Iterable[Mapping[str, Any]], labels: Mapping[str, Mapping[str, str]]) -> None:
    case_list = list(cases)
    expected = {str(case["case_id"]) for case in case_list}
    actual = set(labels)
    if expected != actual:
        raise ValueError(
            f"Label coverage mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    for case in case_list:
        case_id = str(case["case_id"])
        row = labels[case_id]
        if not row.get("pair_difference"):
            raise ValueError(f"Missing pair_difference for {case_id}")
        for arm in ("pure", "psc"):
            primary = row.get(f"{arm}_primary", "")
            secondary = row.get(f"{arm}_secondary", "")
            if primary not in TAXONOMY:
                raise ValueError(f"Invalid {arm} primary for {case_id}: {primary!r}")
            if secondary and secondary not in TAXONOMY - {"correct"}:
                raise ValueError(f"Invalid {arm} secondary for {case_id}: {secondary!r}")
            is_correct = float(case[arm]["em"]) == 1.0
            if is_correct != (primary == "correct"):
                raise ValueError(
                    f"EM/label mismatch for {case_id}/{arm}: em={case[arm]['em']} primary={primary}"
                )


def aggregate(cases: list[Mapping[str, Any]], labels: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    raw: dict[str, Counter[str]] = {"pure": Counter(), "psc": Counter()}
    weighted: dict[str, Counter[str]] = {"pure": Counter(), "psc": Counter()}
    raw_totals = Counter()
    weighted_totals = Counter()
    raw_stages: dict[str, Counter[str]] = {"pure": Counter(), "psc": Counter()}
    weighted_stages: dict[str, Counter[str]] = {"pure": Counter(), "psc": Counter()}
    by_dataset: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"pure": Counter(), "psc": Counter()}
    )
    by_stratum: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"pure": Counter(), "psc": Counter()}
    )
    pair_differences: Counter[str] = Counter()
    for case in cases:
        row = labels[str(case["case_id"])]
        weight = float(case["selection_weight"])
        pair_differences[row["pair_difference"]] += 1
        for arm in ("pure", "psc"):
            category = row[f"{arm}_primary"]
            if category == "correct":
                continue
            raw[arm][category] += 1
            weighted[arm][category] += weight
            raw_totals[arm] += 1
            weighted_totals[arm] += weight
            raw_stages[arm][STAGE_BY_CATEGORY[category]] += 1
            weighted_stages[arm][STAGE_BY_CATEGORY[category]] += weight
            by_dataset[str(case["dataset"])][arm][category] += 1
            by_stratum[str(case["stratum"])][arm][category] += 1

    return {
        "sample_size": len(cases),
        "eligible_population_questions": sum(
            float(case["selection_weight"]) for case in cases
        ),
        "arm_error_totals": {
            arm: {"sample": raw_totals[arm], "population_estimate": weighted_totals[arm]}
            for arm in ("pure", "psc")
        },
        "sample_error_distribution": {
            arm: sorted_distribution(raw[arm], raw_totals[arm]) for arm in ("pure", "psc")
        },
        "design_weighted_error_distribution": {
            arm: sorted_distribution(weighted[arm], weighted_totals[arm])
            for arm in ("pure", "psc")
        },
        "sample_stage_distribution": {
            arm: sorted_distribution(raw_stages[arm], raw_totals[arm])
            for arm in ("pure", "psc")
        },
        "design_weighted_stage_distribution": {
            arm: sorted_distribution(weighted_stages[arm], weighted_totals[arm])
            for arm in ("pure", "psc")
        },
        "pair_difference_counts": dict(pair_differences.most_common()),
        "by_dataset_sample_counts": {
            dataset: {arm: dict(counts[arm].most_common()) for arm in ("pure", "psc")}
            for dataset, counts in sorted(by_dataset.items())
        },
        "by_stratum_sample_counts": {
            stratum: {arm: dict(counts[arm].most_common()) for arm in ("pure", "psc")}
            for stratum, counts in sorted(by_stratum.items())
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen2.5-3B paired error audit (100 questions)",
        "",
        "The sample is stratified by dataset and EM outcome quadrant. Raw percentages describe the reviewed sample; design-weighted percentages estimate the frozen population of questions where at least one arm is wrong.",
        "",
        f"- Reviewed questions: {report['sample_size']}",
        f"- Eligible population questions: {report['eligible_population_questions']:.0f}",
        "",
    ]
    for title, key in (
        ("Raw sample stage distribution", "sample_stage_distribution"),
        ("Design-weighted population stage distribution", "design_weighted_stage_distribution"),
        ("Raw sample error distribution", "sample_error_distribution"),
        ("Design-weighted population error distribution", "design_weighted_error_distribution"),
    ):
        lines.extend([f"## {title}", "", "| Arm | Category | Count | Percent |", "|---|---|---:|---:|"])
        for arm in ("pure", "psc"):
            for row in report[key][arm]:
                count = row["count"]
                rendered_count = f"{count:.1f}" if isinstance(count, float) else str(count)
                lines.append(f"| {arm} | {row['category']} | {rendered_count} | {row['percent']:.1f}% |")
        lines.append("")
    lines.extend(["## Pair-level comparison", "", "| Difference | Questions |", "|---|---:|"])
    for label, count in report["pair_difference_counts"].items():
        lines.append(f"| {label} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    labels = load_labels(args.labels)
    validate(cases, labels)
    report = aggregate(cases, labels)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "error_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "error_audit_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
