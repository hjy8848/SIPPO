#!/usr/bin/env python3
"""Compare matched SFT, PureEM, and PSC workflow errors sample by sample."""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


Row = Mapping[str, str]
Predicate = Callable[[Row], bool]


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def as_bool(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def load_cases(path: Path, formal: bool) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            dataset = str(row.get("dataset") or "train1000") if formal else "train1000"
            key = (dataset, normalize(row.get("question")))
            if not key[1] or key in rows:
                raise ValueError(f"invalid or duplicate case at {path}:{line_number}")
            rows[key] = row
    return rows


TRAIN_DEFECTS: dict[str, Predicate] = {
    "semantic_shortcut": lambda row: as_bool(row.get("semantic_shortcut")),
    "execution_contract_failure": lambda row: row.get("category")
    == "execution_contract_failure",
    "missing_reason": lambda row: row.get("category") == "planner_missing_reason",
    "comparison_evidence_or_plan_failure": lambda row: row.get("category")
    == "comparison_evidence_or_plan_failure",
    "reason_failure_with_complete_evidence": lambda row: row.get("category")
    == "reason_failure_with_complete_evidence",
    "reason_not_consumed": lambda row: row.get("category")
    == "reason_not_consumed_by_search",
    "identity_transfer_failure": lambda row: row.get("category")
    == "reason_to_search_identity_transfer_failure",
    "final_aggregation_failure": lambda row: row.get("category")
    == "final_aggregation_failure",
    "noncanonical_plan": lambda row: not as_bool(row.get("canonical_plan")),
}


FORMAL_DEFECTS: dict[str, Predicate] = {
    "execution_contract_failure": lambda row: row.get("category")
    == "execution_contract_failure",
    "retrieval_context_missing_gold": lambda row: row.get("category")
    == "retrieval_context_missing_gold",
    "answer_extraction_failure": lambda row: row.get("category")
    == "answer_extraction_failure",
    "final_sink_aggregation_failure": lambda row: row.get("category")
    == "final_sink_aggregation_failure",
}


def _defect_lifecycle(
    keys: Sequence[tuple[str, str]],
    stages: Mapping[str, Mapping[tuple[str, str], Row]],
    predicate: Predicate,
) -> dict[str, int]:
    flags = {
        stage: {key: predicate(rows[key]) for key in keys} for stage, rows in stages.items()
    }
    return {
        "sft": sum(flags["sft"].values()),
        "pure": sum(flags["pure"].values()),
        "psc": sum(flags["psc"].values()),
        "present_in_all": sum(
            flags["sft"][key] and flags["pure"][key] and flags["psc"][key]
            for key in keys
        ),
        "pure_introduced_vs_sft": sum(
            not flags["sft"][key] and flags["pure"][key] for key in keys
        ),
        "pure_resolved_vs_sft": sum(
            flags["sft"][key] and not flags["pure"][key] for key in keys
        ),
        "psc_introduced_vs_pure": sum(
            not flags["pure"][key] and flags["psc"][key] for key in keys
        ),
        "psc_resolved_vs_pure": sum(
            flags["pure"][key] and not flags["psc"][key] for key in keys
        ),
        "psc_new_vs_sft": sum(
            not flags["sft"][key] and flags["psc"][key] for key in keys
        ),
        "psc_removed_vs_sft": sum(
            flags["sft"][key] and not flags["psc"][key] for key in keys
        ),
    }


def _em_transition(
    keys: Sequence[tuple[str, str]],
    left: Mapping[tuple[str, str], Row],
    right: Mapping[tuple[str, str], Row],
) -> dict[str, int]:
    pairs = [(float(left[key]["em"]), float(right[key]["em"])) for key in keys]
    return {
        "improved": sum(a < b for a, b in pairs),
        "regressed": sum(a > b for a, b in pairs),
        "both_correct": sum(a == 1.0 and b == 1.0 for a, b in pairs),
        "both_wrong": sum(a == 0.0 and b == 0.0 for a, b in pairs),
    }


def compare_split(
    sft: Mapping[tuple[str, str], Row],
    pure: Mapping[tuple[str, str], Row],
    psc: Mapping[tuple[str, str], Row],
    defects: Mapping[str, Predicate],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(sft) != set(pure) or set(sft) != set(psc):
        raise ValueError(
            f"case sets differ: sft={len(sft)}, pure={len(pure)}, psc={len(psc)}"
        )
    keys = sorted(sft)
    stages = {"sft": sft, "pure": pure, "psc": psc}
    report = {
        "samples": len(keys),
        "em": {
            stage: sum(float(rows[key]["em"]) for key in keys) / len(keys)
            for stage, rows in stages.items()
        },
        "em_transitions": {
            "sft_to_pure": _em_transition(keys, sft, pure),
            "pure_to_psc": _em_transition(keys, pure, psc),
            "sft_to_psc": _em_transition(keys, sft, psc),
        },
        "defects": {
            name: _defect_lifecycle(keys, stages, predicate)
            for name, predicate in defects.items()
        },
        "category_transitions": {
            "sft_to_pure": dict(
                Counter((sft[key]["category"], pure[key]["category"]) for key in keys)
            ),
            "pure_to_psc": dict(
                Counter((pure[key]["category"], psc[key]["category"]) for key in keys)
            ),
        },
    }
    # JSON cannot encode tuple keys, so retain the most frequent transitions as records.
    for transition, counts in list(report["category_transitions"].items()):
        report["category_transitions"][transition] = [
            {"from": source, "to": target, "count": count}
            for (source, target), count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    cases = []
    for key in keys:
        case: dict[str, Any] = {
            "dataset": key[0],
            "question": sft[key]["question"],
        }
        for stage, rows in stages.items():
            case[f"{stage}_em"] = rows[key]["em"]
            case[f"{stage}_category"] = rows[key]["category"]
            case[f"{stage}_plan_topology"] = rows[key].get("plan_topology", "")
        for name, predicate in defects.items():
            for stage, rows in stages.items():
                case[f"{name}_{stage}"] = predicate(rows[key])
        cases.append(case)
    return report, cases


def plan_node_count(row: Row) -> int:
    topology = str(row.get("plan_topology") or "")
    return 0 if topology in {"", "none"} else len(topology.split("-"))


def node_budget_stable_subset(
    stages: Mapping[str, Mapping[tuple[str, str], Row]], max_nodes: int
) -> dict[str, dict[tuple[str, str], Row]]:
    """Keep rows whose generated plans fit the legacy SFT runtime budget."""
    keys = set.intersection(*(set(rows) for rows in stages.values()))
    stable_keys = {
        key
        for key in keys
        if all(plan_node_count(rows[key]) <= max_nodes for rows in stages.values())
        and f"maximum is {max_nodes}" not in stages["sft"][key].get("execution_error", "")
    }
    return {
        stage: {key: row for key, row in rows.items() if key in stable_keys}
        for stage, rows in stages.items()
    }


def write_cases(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "formal"):
        for stage in ("sft", "pure", "psc"):
            parser.add_argument(f"--{stage}-{split}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")

    train_stages = {
        "sft": load_cases(args.sft_train, formal=False),
        "pure": load_cases(args.pure_train, formal=False),
        "psc": load_cases(args.psc_train, formal=False),
    }
    train_report, train_cases = compare_split(
        train_stages["sft"], train_stages["pure"], train_stages["psc"], TRAIN_DEFECTS
    )
    stable_train_stages = node_budget_stable_subset(train_stages, max_nodes=5)
    stable_train_report, _ = compare_split(
        stable_train_stages["sft"],
        stable_train_stages["pure"],
        stable_train_stages["psc"],
        TRAIN_DEFECTS,
    )
    stable_train_report["selection_guard"] = (
        "Post-hoc output-conditioned subset: every generated plan has at most five "
        "nodes and the SFT row did not fail the legacy max-five runtime check. This "
        "is a node-budget robustness audit, not a causal training comparison."
    )
    formal_report, formal_cases = compare_split(
        load_cases(args.sft_formal, formal=True),
        load_cases(args.pure_formal, formal=True),
        load_cases(args.psc_formal, formal=True),
        FORMAL_DEFECTS,
    )
    report = {
        "method": "matched_sft_pure_psc_workflow_lifecycle_v1",
        "train1000": train_report,
        "train1000_max5_output_subset": stable_train_report,
        "formal_v2": formal_report,
        "interpretation_guard": (
            "Lifecycle counts are paired observable error labels. Formal lexical-stage labels "
            "are not causal supporting-fact or retrieval-recall measurements."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_cases(args.output_dir / "train_lifecycle_cases.csv", train_cases)
    write_cases(args.output_dir / "formal_lifecycle_cases.csv", formal_cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
