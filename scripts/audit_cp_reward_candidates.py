#!/usr/bin/env python3
"""Audit outcome-preserving reward candidates for CP-DAGENT.

This audit deliberately separates safety gates from trainable reward signals.
It never treats protocol validity, topology, or role usage as correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import string
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: Any, references: Any) -> int:
    if isinstance(references, str):
        candidates = [references]
    elif isinstance(references, Sequence):
        candidates = list(references)
    else:
        candidates = [references]
    prediction = normalize_answer(prediction)
    return int(bool(prediction) and prediction in {normalize_answer(item) for item in candidates})


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_process_valid_psc(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if float(row.get("em", 0.0) or 0.0) == 1.0]
    shortcuts = [row for row in positives if as_bool(row.get("semantic_shortcut"))]
    grounded = [
        row
        for row in positives
        if as_bool(row.get("correct_reason_consumed"))
        and as_bool(row.get("identity_binding_correct"))
        and as_bool(row.get("downstream_conclusion_has_gold"))
    ]
    canonical_but_wrong_reason = [
        row
        for row in positives
        if as_bool(row.get("canonical_plan"))
        and not as_bool(row.get("reason_selects_winner"))
    ]
    positive_count = len(positives)
    grounded_count = len(grounded)
    retention = grounded_count / max(1, positive_count)
    return {
        "candidate": "process_valid_psc_chosen_gate",
        "status": "offline_retrospective_proxy_only",
        "samples": len(rows),
        "em_positive_workflows": positive_count,
        "em_positive_semantic_shortcuts": len(shortcuts),
        "em_positive_semantic_shortcut_rate": len(shortcuts) / max(1, positive_count),
        "em_positive_grounded_reason_to_search": grounded_count,
        "strict_proxy_retention_rate": retention,
        "canonical_but_wrong_reason_em_positive": len(canonical_but_wrong_reason),
        "gate_semantics": (
            "The strict proxy requires an EM-positive workflow to consume the correct "
            "Reason, bind the selected identity, and retrieve a gold-bearing downstream conclusion."
        ),
        "not_equivalent_to_runtime_contract_gate": True,
        "safety_gate_is_not_a_reward": True,
        "offline_gate": {
            "minimum_grounded_positive_cases": 30,
            "minimum_positive_retention_rate": 0.10,
        },
        "passed_as_trainable_reward": bool(grounded_count >= 30 and retention >= 0.10),
        "interpretation": (
            "Useful as a chosen-eligibility safety condition, but too sparse and label-dependent "
            "to become an independent scalar reward on these legacy trajectories."
        ),
    }


def _selection_cases(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        trace = row.get("raw_trace") or {}
        nodes = list(trace.get("nodes_execution") or [])
        sink_id = str(trace.get("answer_sink_id") or "").strip("#")
        for node_index, node in enumerate(nodes):
            if node.get("reason_execution_method") != "model_typed_selection":
                continue
            handoff = node.get("typed_handoff") or {}
            selected = str(handoff.get("selected_candidate") or node.get("conclusion") or "")
            alternatives = [
                str(source.get("candidate") or "")
                for source in handoff.get("rejected_sources") or []
                if str(source.get("candidate") or "").strip()
            ]
            factual_em = exact_match(selected, row.get("gold_answer") or [])
            alternative_em = max(
                (exact_match(candidate, row.get("gold_answer") or []) for candidate in alternatives),
                default=0,
            )
            final_answer = str(trace.get("final_answer") or row.get("pred_answer") or "")
            terminal = sink_id == str(node.get("id") or "") and node_index == len(nodes) - 1
            valid_selection = bool(
                selected.strip()
                and alternatives
                and not str(node.get("handoff_validation_error") or "").strip()
                and not str(node.get("handoff_dependency_error") or "").strip()
            )
            identity_exact = bool(
                valid_selection
                and terminal
                and normalize_answer(final_answer) == normalize_answer(node.get("conclusion"))
                and int(float(row.get("em", 0.0) or 0.0)) == factual_em
            )
            cases.append(
                {
                    "row_index": row_index,
                    "question": str(row.get("question") or ""),
                    "node_id": str(node.get("id") or ""),
                    "operator": str((node.get("contract_node") or {}).get("op") or ""),
                    "selected_candidate": selected,
                    "alternative_candidates": alternatives,
                    "gold_answer": row.get("gold_answer") or [],
                    "factual_em": factual_em,
                    "alternative_em": alternative_em,
                    "terminal_sink": terminal,
                    "valid_selection": valid_selection,
                    "identity_exact": identity_exact,
                    "decisive": factual_em != alternative_em,
                    "factual_wins": factual_em > alternative_em,
                    "alternative_wins": alternative_em > factual_em,
                }
            )
    return cases


def audit_typed_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_calls = _selection_cases(rows)
    cases = [case for case in all_calls if case["valid_selection"]]
    exact_cases = [case for case in cases if case["identity_exact"]]
    decisive = [case for case in exact_cases if case["decisive"]]
    factual_wins = sum(case["factual_wins"] for case in decisive)
    alternative_wins = sum(case["alternative_wins"] for case in decisive)
    decisive_rate = len(decisive) / max(1, len(exact_cases))
    interval = wilson_interval(len(decisive), len(exact_cases))
    identity_passed = len(exact_cases) == len(cases)
    passed = bool(
        identity_passed
        and len(cases) >= 16
        and interval[0] > 0.25
        and factual_wins > 0
        and alternative_wins > 0
    )
    return {
        "candidate": "typed_selection_counterfactual_decisiveness_weight",
        "status": "offline_exact_terminal_counterfactual",
        "evaluation_samples": len(rows),
        "model_typed_selection_calls": len(all_calls),
        "ineligible_invalid_selection_calls": len(all_calls) - len(cases),
        "eligible_typed_selection_cases": len(cases),
        "identity_exact_cases": len(exact_cases),
        "identity_mismatches": len(cases) - len(exact_cases),
        "decisive_cases": len(decisive),
        "decisive_rate": decisive_rate,
        "decisive_rate_wilson_95ci": interval,
        "factual_wins": factual_wins,
        "alternative_wins": alternative_wins,
        "both_wrong": sum(
            case["factual_em"] == case["alternative_em"] == 0 for case in exact_cases
        ),
        "both_correct": sum(
            case["factual_em"] == case["alternative_em"] == 1 for case in exact_cases
        ),
        "overall_coverage_rate": len(cases) / max(1, len(rows)),
        "overall_decisive_coverage_rate": len(decisive) / max(1, len(rows)),
        "proposed_weight": "A_role = A_EM * (1 + alpha * I[factual_EM != legal_sibling_EM]), 0 < alpha <= 0.2",
        "advantage_sign_preserved_by_construction": True,
        "missing_marker_intervention": False,
        "model_or_retriever_replay_required": False,
        "offline_gate": {
            "minimum_cases": 16,
            "decisive_rate_ci_low_must_exceed": 0.25,
            "identity_mismatches_must_equal": 0,
            "must_observe_both_action_directions": True,
        },
        "passed": passed,
        "cases": all_calls,
    }


def audit_legacy_onpolicy(report: Mapping[str, Any]) -> dict[str, Any]:
    statistics = report.get("replacement_statistics") or {}
    overall = report.get("overall") or {}
    invariants = report.get("safety_invariants") or {}
    identity_changes = int(statistics.get("identity_response_terminal_em_changed_samples", 0))
    identity_exact = bool(invariants.get("identity_replacement_is_exact_negative_control", False))
    informative = int(overall.get("informative_estimate_cases", 0))
    passed = bool(identity_exact and identity_changes == 0 and informative >= 30)
    return {
        "candidate": "generic_onpolicy_search_reason_replacement_weight",
        "status": "invalid_estimator" if not identity_exact else "offline_audit",
        "replacement_samples": int(statistics.get("replacement_samples", 0)),
        "distinct_response_rate": float(statistics.get("distinct_response_rate", 0.0)),
        "terminal_em_changed_rate": float(statistics.get("terminal_em_changed_rate", 0.0)),
        "informative_estimate_cases": informative,
        "identity_response_samples": int(statistics.get("identity_response_samples", 0)),
        "identity_terminal_changes": identity_changes,
        "identity_control_exact": identity_exact,
        "passed": passed,
        "interpretation": (
            "Reject the generic replay reward: byte-identical interventions changed terminal EM, "
            "and non-identity replacements supplied essentially no terminal signal."
        ),
    }


def audit_lifecycle(report: Mapping[str, Any]) -> dict[str, Any]:
    train = report.get("train1000") or {}
    defects = train.get("defects") or {}
    return {
        "em": train.get("em") or {},
        "psc_vs_pure": {
            name: {
                "introduced": int(values.get("psc_introduced_vs_pure", 0)),
                "resolved": int(values.get("psc_resolved_vs_pure", 0)),
            }
            for name, values in defects.items()
            if name
            in {
                "semantic_shortcut",
                "execution_contract_failure",
                "reason_failure_with_complete_evidence",
                "reason_not_consumed",
                "identity_transfer_failure",
                "final_aggregation_failure",
            }
        },
        "interpretation": (
            "PSC improves EM and protocol execution while introducing many semantic shortcuts and "
            "unconsumed/incorrect Reason paths; this motivates outcome-grounded local credit, not format bonuses."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-trajectories", type=Path, required=True)
    parser.add_argument("--psc-train-cases", type=Path, required=True)
    parser.add_argument("--legacy-onpolicy-report", type=Path, required=True)
    parser.add_argument("--lifecycle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "cp_trajectories": args.cp_trajectories,
        "psc_train_cases": args.psc_train_cases,
        "legacy_onpolicy_report": args.legacy_onpolicy_report,
        "lifecycle_report": args.lifecycle_report,
    }
    cp_rows = json.loads(args.cp_trajectories.read_text(encoding="utf-8"))
    legacy_report = json.loads(args.legacy_onpolicy_report.read_text(encoding="utf-8"))
    lifecycle_report = json.loads(args.lifecycle_report.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "method": "cp_dagent_reward_candidate_offline_audit_v1",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "evidence_scope": {
            "implemented_only": False,
            "offline_signal_audit": True,
            "one_batch_gradient_check": False,
            "short_training": False,
            "formal_evaluation": False,
        },
        "lifecycle_motivation": audit_lifecycle(lifecycle_report),
        "candidates": {
            "process_valid_psc": audit_process_valid_psc(load_csv(args.psc_train_cases)),
            "typed_selection": audit_typed_selection(cp_rows),
            "generic_onpolicy_replacement": audit_legacy_onpolicy(legacy_report),
        },
    }
    report["advance_to_one_batch"] = [
        name
        for name, result in report["candidates"].items()
        if result.get("passed") or result.get("passed_as_trainable_reward")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
