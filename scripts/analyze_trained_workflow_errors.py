#!/usr/bin/env python3
"""Attribute errors in completed DAGENT evaluations to observable workflow stages."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REF_RE = re.compile(r"#(\d+)#")
NODE_RE = re.compile(
    r"<(search|reason)\b[^>]*\bid\s*=\s*['\"](\d+)['\"][^>]*>(.*?)</\1\s*>",
    re.I | re.S,
)


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def answer_list(row: Mapping[str, Any]) -> list[str]:
    answers = row.get("gold_answer") or []
    if isinstance(answers, str):
        answers = [answers]
    return [normalize(answer) for answer in answers if normalize(answer)]


def contains_answer(text: Any, answers: Sequence[str]) -> bool:
    normalized = normalize(text)
    return any(answer in normalized for answer in answers)


def is_error(value: Any) -> bool:
    return value not in (None, "", False, [], {})


def trace_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row.get("raw_trace") or row.get("trace") or {}


def nodes_of(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(trace_of(row).get("nodes_execution") or [])


def execution_error(row: Mapping[str, Any]) -> str:
    trace = trace_of(row)
    return str(trace.get("execution_error") or row.get("error") or "").strip()


def parse_plan_types(row: Mapping[str, Any]) -> list[str]:
    trace = trace_of(row)
    plan = str(trace.get("plan_output") or trace.get("raw_plan_output") or "")
    return [kind.casefold() for kind, _, _ in NODE_RE.findall(plan)]


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list in {path}")
    return payload


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = normalize(row.get("question"))
            if not key or key in rows:
                raise ValueError(f"invalid or duplicate metadata row {line_number}")
            rows[key] = row
    return rows


def _joined_node_text(nodes: Iterable[Mapping[str, Any]], field: str) -> str:
    return " ".join(str(node.get(field) or "") for node in nodes)


def classify_train_row(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    nodes = nodes_of(row)
    answers = answer_list(row)
    blueprint = metadata.get("blueprint") or {}
    comparison = metadata.get("comparison_evidence") or {}
    winner = normalize(blueprint.get("selected_candidate") or metadata.get("winner"))
    counterfactual = normalize(metadata.get("counterfactual_entity"))
    def comparison_variants(candidate: str) -> set[str]:
        evidence = comparison.get(candidate) or {}
        answer = normalize(evidence.get("answer"))
        variants = {answer} if answer else set()
        key = evidence.get("comparison_key")
        if key not in (None, ""):
            key_text = str(int(float(key)))
            if len(key_text) >= 4:
                variants.add(key_text[:4])
        variants.update(re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", str(evidence.get("answer") or "")))
        return variants

    variants_a = comparison_variants("candidate_a")
    variants_b = comparison_variants("candidate_b")

    first_reason_position = next(
        (index for index, node in enumerate(nodes) if str(node.get("type", "")).casefold() == "reason"),
        len(nodes),
    )
    pre_reason_searches = [
        node
        for node in nodes[:first_reason_position]
        if str(node.get("type", "")).casefold() == "search"
    ]
    # Reason receives parent conclusions, not the full retrieved contexts.
    pre_reason_conclusions = normalize(_joined_node_text(pre_reason_searches, "conclusion"))
    comparison_evidence_count = sum(
        any(value and value in pre_reason_conclusions for value in variants)
        for variants in (variants_a, variants_b)
    )
    comparison_evidence_complete = comparison_evidence_count == 2

    reason_positions = [
        (index, node)
        for index, node in enumerate(nodes)
        if str(node.get("type", "")).casefold() == "reason"
    ]
    entity_correct_reasons = [
        (index, node)
        for index, node in reason_positions
        if winner and winner in normalize(node.get("conclusion") or node.get("raw_output"))
    ]
    source_correct_reasons = []
    source_wrong_reasons = []
    if reason_positions:
        previous: dict[str, Mapping[str, Any]] = {}
        for index, node in enumerate(nodes):
            if str(node.get("type", "")).casefold() != "reason":
                previous[str(node.get("id") or "")] = node
                continue
            conclusion = normalize(node.get("conclusion") or node.get("raw_output"))
            raw_reason = str(node.get("query") or node.get("task") or "")
            matched_parents = [
                previous[reference]
                for reference in REF_RE.findall(raw_reason)
                if reference in previous
                and normalize(previous[reference].get("conclusion")) == conclusion
                and conclusion
            ]
            if len(matched_parents) == 1:
                parent_query = normalize(
                    matched_parents[0].get("query") or matched_parents[0].get("task")
                )
                if winner and winner in parent_query:
                    source_correct_reasons.append((index, node))
                elif counterfactual and counterfactual in parent_query:
                    source_wrong_reasons.append((index, node))
            previous[str(node.get("id") or "")] = node
    correct_reason_keys = {
        (index, str(node.get("id") or ""))
        for index, node in entity_correct_reasons + source_correct_reasons
    }
    correct_reasons = [
        (index, node)
        for index, node in reason_positions
        if (index, str(node.get("id") or "")) in correct_reason_keys
    ]
    wrong_candidate_reasons = [
        (index, node)
        for index, node in reason_positions
        if (
            counterfactual
            and counterfactual in normalize(node.get("conclusion") or node.get("raw_output"))
        )
        or (index, str(node.get("id") or ""))
        in {(position, str(reason.get("id") or "")) for position, reason in source_wrong_reasons}
    ]
    reason_outputs_gold = any(
        contains_answer(node.get("conclusion") or node.get("raw_output"), answers)
        for _, node in reason_positions
    )
    counterfactual_answers = [
        normalize(metadata.get("counterfactual_answer")),
        normalize(((metadata.get("downstream_evidence") or {}).get("counterfactual") or {}).get("answer")),
    ]
    counterfactual_answers = [value for value in counterfactual_answers if value]
    reason_outputs_counterfactual_answer = any(
        contains_answer(
            node.get("conclusion") or node.get("raw_output"), counterfactual_answers
        )
        for _, node in reason_positions
    )
    pre_reason_has_gold = contains_answer(pre_reason_conclusions, answers)
    pre_reason_has_counterfactual_answer = contains_answer(
        pre_reason_conclusions, counterfactual_answers
    )
    semantic_shortcut = bool(
        reason_positions
        and pre_reason_has_gold
        and pre_reason_has_counterfactual_answer
        and not comparison_evidence_complete
    )

    consumed_searches: list[Mapping[str, Any]] = []
    bound_searches: list[Mapping[str, Any]] = []
    for reason_position, reason in correct_reasons:
        reason_id = str(reason.get("id") or "")
        for node in nodes[reason_position + 1 :]:
            if str(node.get("type", "")).casefold() != "search":
                continue
            raw_search = str(node.get("query") or node.get("task") or "")
            if reason_id and reason_id in REF_RE.findall(raw_search):
                consumed_searches.append(node)
                executed_query = str(
                    node.get("executed_query")
                    or node.get("resolved_query")
                    or node.get("query")
                    or ""
                )
                normalized_query = normalize(executed_query)
                if (
                    winner
                    and winner in normalized_query
                    and not (counterfactual and counterfactual in normalized_query)
                ):
                    bound_searches.append(node)

    downstream_context_has_gold = contains_answer(
        _joined_node_text(bound_searches, "search_context"), answers
    )
    downstream_conclusion_has_gold = contains_answer(
        _joined_node_text(bound_searches, "conclusion"), answers
    )
    last_conclusion_has_gold = bool(
        nodes and contains_answer(nodes[-1].get("conclusion"), answers)
    )
    error = execution_error(row)
    em = float(row.get("em", 0.0) or 0.0)

    if em == 1.0:
        category = "correct"
    elif error:
        category = "execution_contract_failure"
    elif last_conclusion_has_gold:
        category = "final_aggregation_failure"
    elif not reason_positions:
        category = "planner_missing_reason"
    elif semantic_shortcut:
        category = "semantic_plan_compares_target_values"
    elif comparison_evidence_complete and not correct_reasons:
        category = "reason_failure_with_complete_evidence"
    elif not comparison_evidence_complete and not correct_reasons:
        category = "comparison_evidence_or_plan_failure"
    elif not correct_reasons:
        category = "reason_failure_with_incomplete_evidence"
    elif not consumed_searches:
        category = "reason_not_consumed_by_search"
    elif not bound_searches:
        category = "reason_to_search_identity_transfer_failure"
    elif not downstream_context_has_gold:
        category = "downstream_retrieval_or_relation_failure"
    elif not downstream_conclusion_has_gold:
        category = "answer_extraction_failure"
    else:
        category = "final_aggregation_failure"

    plan_types = parse_plan_types(row)
    return {
        "question": str(row.get("question") or ""),
        "operation": str((metadata.get("reasoning") or {}).get("operation") or "unknown"),
        "em": em,
        "f1": float(row.get("f1", 0.0) or 0.0),
        "category": category,
        "execution_error": error,
        "plan_topology": "-".join(plan_types) or "none",
        "canonical_plan": plan_types == ["search", "search", "reason", "search"],
        "comparison_evidence_count": comparison_evidence_count,
        "comparison_evidence_complete": comparison_evidence_complete,
        "reason_count": len(reason_positions),
        "reason_selects_winner": bool(correct_reasons),
        "reason_selects_winner_entity": bool(entity_correct_reasons),
        "reason_selects_winner_value": bool(source_correct_reasons),
        "reason_selects_counterfactual": bool(wrong_candidate_reasons),
        "reason_outputs_gold": reason_outputs_gold,
        "reason_outputs_counterfactual_answer": reason_outputs_counterfactual_answer,
        "semantic_shortcut": semantic_shortcut,
        "correct_reason_consumed": bool(consumed_searches),
        "identity_binding_correct": bool(bound_searches),
        "downstream_context_has_gold": downstream_context_has_gold,
        "downstream_conclusion_has_gold": downstream_conclusion_has_gold,
        "last_conclusion_has_gold": last_conclusion_has_gold,
        "pred_answer": str(row.get("pred_answer") or ""),
    }


def summarize_train(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    operations = sorted({str(row["operation"]) for row in rows})
    return {
        "samples": len(rows),
        "em": sum(float(row["em"]) for row in rows) / len(rows),
        "categories": dict(Counter(str(row["category"]) for row in rows)),
        "error_messages": dict(
            Counter(str(row["execution_error"]) for row in rows if row["execution_error"])
        ),
        "stage_coverage": {
            "canonical_plan": sum(bool(row["canonical_plan"]) for row in rows),
            "comparison_evidence_complete": sum(
                bool(row["comparison_evidence_complete"]) for row in rows
            ),
            "reason_selects_winner": sum(bool(row["reason_selects_winner"]) for row in rows),
            "reason_selects_winner_entity": sum(
                bool(row["reason_selects_winner_entity"]) for row in rows
            ),
            "reason_selects_winner_value": sum(
                bool(row["reason_selects_winner_value"]) for row in rows
            ),
            "reason_outputs_gold": sum(bool(row["reason_outputs_gold"]) for row in rows),
            "semantic_shortcut": sum(bool(row["semantic_shortcut"]) for row in rows),
            "semantic_shortcut_em": sum(
                bool(row["semantic_shortcut"]) and float(row["em"]) == 1.0 for row in rows
            ),
            "correct_reason_consumed": sum(bool(row["correct_reason_consumed"]) for row in rows),
            "identity_binding_correct": sum(bool(row["identity_binding_correct"]) for row in rows),
            "downstream_context_has_gold": sum(
                bool(row["downstream_context_has_gold"]) for row in rows
            ),
            "downstream_conclusion_has_gold": sum(
                bool(row["downstream_conclusion_has_gold"]) for row in rows
            ),
        },
        "success_paths": {
            "correct_total": sum(float(row["em"]) == 1.0 for row in rows),
            "correct_with_canonical_plan": sum(
                float(row["em"]) == 1.0 and bool(row["canonical_plan"]) for row in rows
            ),
            "correct_with_semantic_shortcut": sum(
                float(row["em"]) == 1.0 and bool(row["semantic_shortcut"]) for row in rows
            ),
            "correct_with_reason_consumed": sum(
                float(row["em"]) == 1.0 and bool(row["correct_reason_consumed"])
                for row in rows
            ),
            "correct_with_unique_identity_binding": sum(
                float(row["em"]) == 1.0 and bool(row["identity_binding_correct"])
                for row in rows
            ),
        },
        "by_operation": {
            operation: {
                "samples": len(group := [row for row in rows if row["operation"] == operation]),
                "categories": dict(Counter(str(row["category"]) for row in group)),
            }
            for operation in operations
        },
    }


def classify_formal_row(row: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    nodes = nodes_of(row)
    answers = answer_list(row)
    binary = any(answer in {"yes", "no"} for answer in answers)
    search_nodes = [
        node for node in nodes if str(node.get("type", "")).casefold() == "search"
    ]
    all_context = _joined_node_text(search_nodes, "search_context")
    all_conclusions = _joined_node_text(nodes, "conclusion")
    context_has_gold = contains_answer(all_context, answers)
    conclusion_has_gold = contains_answer(all_conclusions, answers)
    last_conclusion_has_gold = bool(
        nodes and contains_answer(nodes[-1].get("conclusion"), answers)
    )
    error = execution_error(row)
    em = float(row.get("em", 0.0) or 0.0)

    if em == 1.0:
        category = "correct"
    elif error:
        category = "execution_contract_failure"
    elif binary:
        category = "binary_reasoning_or_aggregation_failure"
    elif not context_has_gold:
        category = "retrieval_context_missing_gold"
    elif not conclusion_has_gold:
        category = "answer_extraction_failure"
    elif last_conclusion_has_gold:
        category = "final_sink_aggregation_failure"
    else:
        category = "evidence_selection_or_aggregation_failure"

    plan_types = parse_plan_types(row)
    reason_ids: set[str] = set()
    reason_to_search = False
    for node in nodes:
        node_type = str(node.get("type", "")).casefold()
        node_id = str(node.get("id") or "")
        if node_type == "reason":
            reason_ids.add(node_id)
        elif node_type == "search":
            reason_to_search |= any(
                reference in reason_ids for reference in REF_RE.findall(str(node.get("query") or ""))
            )
    return {
        "dataset": dataset,
        "question": str(row.get("question") or ""),
        "em": em,
        "f1": float(row.get("f1", 0.0) or 0.0),
        "category": category,
        "binary": binary,
        "execution_error": error,
        "plan_topology": "-".join(plan_types) or "none",
        "executed_reason": any(
            str(node.get("type", "")).casefold() == "reason" for node in nodes
        ),
        "executed_reason_to_search": reason_to_search,
        "context_has_gold": context_has_gold,
        "conclusion_has_gold": conclusion_has_gold,
        "last_conclusion_has_gold": last_conclusion_has_gold,
        "pred_answer": str(row.get("pred_answer") or ""),
    }


def summarize_formal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in rows})

    def group_summary(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(group),
            "em": sum(float(row["em"]) for row in group) / len(group),
            "categories": dict(Counter(str(row["category"]) for row in group)),
            "executed_reason": sum(bool(row["executed_reason"]) for row in group),
            "executed_reason_to_search": sum(
                bool(row["executed_reason_to_search"]) for row in group
            ),
        }

    return {
        "pooled": group_summary(rows),
        "by_dataset": {
            dataset: group_summary([row for row in rows if row["dataset"] == dataset])
            for dataset in datasets
        },
        "interpretation_guard": (
            "Formal stage labels use lexical gold containment. They separate observable "
            "pipeline stages but are not supporting-fact recall or causal interventions."
        ),
    }


def write_cases(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_formal_specs(values: Sequence[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError("--formal must use DATASET=PATH")
        dataset, path = value.split("=", 1)
        specs.append((dataset.strip(), Path(path)))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--formal", action="append", default=[], help="DATASET=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")

    metadata = load_metadata(args.train_metadata)
    train_rows = load_json_rows(args.train_trajectories)
    classified_train = []
    for row in train_rows:
        key = normalize(row.get("question"))
        if key not in metadata:
            raise ValueError(f"train question missing from metadata: {row.get('question')}")
        classified_train.append(classify_train_row(row, metadata[key]))
    if len(classified_train) != len(metadata):
        raise ValueError(f"train/metadata count mismatch: {len(classified_train)}/{len(metadata)}")

    classified_formal = []
    formal_inputs = {}
    for dataset, path in parse_formal_specs(args.formal):
        formal_inputs[dataset] = str(path.resolve())
        classified_formal.extend(
            classify_formal_row(row, dataset) for row in load_json_rows(path)
        )

    report = {
        "method": "trained_workflow_error_attribution_v1",
        "inputs": {
            "train_trajectories": str(args.train_trajectories.resolve()),
            "train_metadata": str(args.train_metadata.resolve()),
            "formal": formal_inputs,
        },
        "train1000": summarize_train(classified_train),
        "formal": summarize_formal(classified_formal) if classified_formal else None,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_cases(args.output_dir / "train_cases.csv", classified_train)
    write_cases(args.output_dir / "formal_cases.csv", classified_formal)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
