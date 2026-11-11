#!/usr/bin/env python3
"""Compare paired original and typed-handoff DAGENT pilot runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--typed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def _load_results(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {str(row["sample_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample_index in {path}")
    return result


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _paired_ci(deltas: list[float], samples: int, seed: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    means.sort()
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def _mcnemar_exact(original: list[dict], typed: list[dict]) -> dict:
    typed_only = sum(not bool(a["em"]) and bool(b["em"]) for a, b in zip(original, typed))
    original_only = sum(bool(a["em"]) and not bool(b["em"]) for a, b in zip(original, typed))
    discordant = typed_only + original_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(typed_only, original_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "typed_only_correct": typed_only,
        "original_only_correct": original_only,
        "discordant": discordant,
        "exact_p_value": p_value,
    }


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", str(text).casefold()))


def _contains_entity(query: str, entity: str) -> bool:
    query_norm = _normalize(query)
    entity_norm = _normalize(entity)
    if entity_norm and entity_norm in query_norm:
        return True
    tokens = [token for token in entity_norm.split() if len(token) >= 3]
    return bool(tokens) and all(token in query_norm.split() for token in tokens)


def _trace_audit(row: dict, metadata: dict) -> dict:
    trace = row.get("raw_trace") or {}
    nodes = trace.get("nodes_execution") or []
    reason_nodes = [node for node in nodes if node.get("type") == "reason"]
    typed_attempts = [
        node
        for node in reason_nodes
        if node.get("typed_handoff") or node.get("handoff_validation_error")
    ]
    handoff = next(
        (node.get("typed_handoff") for node in typed_attempts if node.get("typed_handoff")),
        None,
    )
    reason_ids = {str(node.get("id")) for node in reason_nodes}
    downstream = next(
        (
            node
            for node in nodes
            if node.get("type") == "search"
            and any(f"#{reason_id}#" in str(node.get("query", "")) for reason_id in reason_ids)
        ),
        None,
    )
    executed_query = str((downstream or {}).get("executed_query", ""))
    oracle_ref = str(metadata.get("oracle_selected_ref", ""))
    selected_entity = str(metadata.get(f"candidate_{oracle_ref}", ""))
    other_ref = "2" if oracle_ref == "1" else "1"
    other_entity = str(metadata.get(f"candidate_{other_ref}", ""))
    mentions_selected = _contains_entity(executed_query, selected_entity)
    mentions_other = _contains_entity(executed_query, other_entity)
    return {
        "plan_output": str(trace.get("plan_output", "")),
        "plan_override_used": bool(trace.get("plan_override_used")),
        "typed_attempted": bool(typed_attempts),
        "typed_valid": handoff is not None,
        "typed_selected_ref": str((handoff or {}).get("selected_ref", "")),
        "typed_selected_ref_correct": bool(handoff) and str(handoff["selected_ref"]) == oracle_ref,
        "executed_query": executed_query,
        "mentions_oracle_entity": mentions_selected,
        "mentions_counterfactual_entity": mentions_other,
        "identity_transfer_error": bool(downstream) and (not mentions_selected or mentions_other),
        "handoff_validation_errors": [
            node.get("handoff_validation_error") for node in typed_attempts if node.get("handoff_validation_error")
        ],
        "rewrite_validation_errors": [
            node.get("rewrite_validation_error") for node in nodes if node.get("rewrite_validation_error")
        ],
        "rewrite_fallback_used": any(node.get("rewrite_fallback_used") for node in nodes),
    }


def compare(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite comparison output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest = pd.read_parquet(args.manifest)
    original = _load_results(args.original)
    typed = _load_results(args.typed)
    expected = set(manifest["sample_index"].astype(str))
    if set(original) != expected or set(typed) != expected:
        raise ValueError("Result sample identities do not exactly match the frozen manifest")

    case_rows = []
    for metadata in manifest.to_dict(orient="records"):
        key = str(metadata["sample_index"])
        original_row, typed_row = original[key], typed[key]
        original_audit = _trace_audit(original_row, metadata)
        typed_audit = _trace_audit(typed_row, metadata)
        case_rows.append(
            {
                **metadata,
                "original_em": float(original_row["em"]),
                "typed_em": float(typed_row["em"]),
                "delta_em": float(typed_row["em"]) - float(original_row["em"]),
                "original_f1": float(original_row["f1"]),
                "typed_f1": float(typed_row["f1"]),
                "delta_f1": float(typed_row["f1"]) - float(original_row["f1"]),
                "plans_identical": original_audit["plan_output"] == typed_audit["plan_output"],
                "original_plan_override_used": original_audit["plan_override_used"],
                "typed_plan_override_used": typed_audit["plan_override_used"],
                "original_identity_transfer_error": original_audit["identity_transfer_error"],
                "typed_identity_transfer_error": typed_audit["identity_transfer_error"],
                "typed_attempted": typed_audit["typed_attempted"],
                "typed_valid": typed_audit["typed_valid"],
                "typed_selected_ref_correct": typed_audit["typed_selected_ref_correct"],
                "typed_selected_ref": typed_audit["typed_selected_ref"],
                "original_executed_query": original_audit["executed_query"],
                "typed_executed_query": typed_audit["executed_query"],
                "typed_rewrite_fallback_used": typed_audit["rewrite_fallback_used"],
                "typed_handoff_errors": json.dumps(typed_audit["handoff_validation_errors"]),
                "typed_rewrite_errors": json.dumps(typed_audit["rewrite_validation_errors"]),
                "original_pipeline_error": bool(original_row.get("error") or original_row.get("pipeline_error")),
                "typed_pipeline_error": bool(typed_row.get("error") or typed_row.get("pipeline_error")),
            }
        )
    cases = pd.DataFrame(case_rows)
    cases.to_csv(args.output_dir / "paired_cases.csv", index=False)

    slices = {}
    for slice_name, frame in cases.groupby("protocol_slice"):
        keys = frame["sample_index"].astype(str).tolist()
        old_rows = [original[key] for key in keys]
        new_rows = [typed[key] for key in keys]
        em_deltas = [float(b["em"]) - float(a["em"]) for a, b in zip(old_rows, new_rows)]
        f1_deltas = [float(b["f1"]) - float(a["f1"]) for a, b in zip(old_rows, new_rows)]
        slices[slice_name] = {
            "count": len(keys),
            "original_em": _mean(old_rows, "em"),
            "typed_em": _mean(new_rows, "em"),
            "delta_em": sum(em_deltas) / len(em_deltas),
            "delta_em_bootstrap_95ci": _paired_ci(em_deltas, args.bootstrap_samples, args.seed),
            "original_f1": _mean(old_rows, "f1"),
            "typed_f1": _mean(new_rows, "f1"),
            "delta_f1": sum(f1_deltas) / len(f1_deltas),
            "delta_f1_bootstrap_95ci": _paired_ci(f1_deltas, args.bootstrap_samples, args.seed + 1),
            "mcnemar": _mcnemar_exact(old_rows, new_rows),
        }

    target = cases[cases["protocol_slice"] == "reason_to_search"]
    controls = cases[cases["protocol_slice"] == "direct_control"]
    eligible = target[target["typed_attempted"]]
    decision_checks = {
        "target_plan_override_complete": bool(len(target)) and bool(
            target["original_plan_override_used"].all()
            and target["typed_plan_override_used"].all()
        ),
        "target_em_strictly_improves": slices["reason_to_search"]["delta_em"] > 0,
        "control_em_non_regression": slices["direct_control"]["delta_em"] >= 0,
        "control_f1_non_regression": slices["direct_control"]["delta_f1"] >= 0,
        "zero_pipeline_errors": not bool(
            cases["original_pipeline_error"].any() or cases["typed_pipeline_error"].any()
        ),
        "identity_transfer_errors_reduce": bool(
            len(eligible)
            and eligible["typed_identity_transfer_error"].mean()
            < eligible["original_identity_transfer_error"].mean()
        ),
    }
    report = {
        "schema_version": 1,
        "paired_count": len(cases),
        "plans_identical_count": int(cases["plans_identical"].sum()),
        "slices": slices,
        "mechanism": {
            "typed_attempted": int(target["typed_attempted"].sum()),
            "typed_valid": int(target["typed_valid"].sum()),
            "typed_selected_ref_correct": int(target["typed_selected_ref_correct"].sum()),
            "eligible_identity_audit_count": len(eligible),
            "original_identity_transfer_errors": int(eligible["original_identity_transfer_error"].sum()),
            "typed_identity_transfer_errors": int(eligible["typed_identity_transfer_error"].sum()),
            "rewrite_fallbacks": int(target["typed_rewrite_fallback_used"].sum()),
        },
        "decision_checks": decision_checks,
        "passed": all(decision_checks.values()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(compare(args), indent=2))


if __name__ == "__main__":
    main()
