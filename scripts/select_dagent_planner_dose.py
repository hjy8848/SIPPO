#!/usr/bin/env python3
"""Apply preregistered mechanism gates to DAGENT Planner-dose candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_assignment(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} must use LABEL=PATH: {value}")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"{option} must use LABEL=PATH: {value}")
    return label.strip(), Path(raw_path.strip())


def load_comparison(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        comparison
        for comparison in payload.get("comparisons", [])
        if comparison.get("candidate") == label
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one candidate={label} comparison in {path}, found {len(matches)}"
        )
    return matches[0]


def evaluate_candidate(
    label: str,
    reasoner: dict[str, Any],
    planner: dict[str, Any],
    *,
    planner_valid_baseline: float,
    matched_reason_search: float,
) -> dict[str, Any]:
    semantic = reasoner["metrics"]["semantic_chosen"]
    valid = planner["metrics"]["valid"]
    reason_search = planner["metrics"]["reason_search"]
    over_five = planner["metrics"]["over_five_nodes"]

    semantic_ci = [float(value) for value in semantic["bootstrap_95ci"]]
    gates = {
        "reasoner_not_significantly_worse": semantic_ci[1] >= 0.0,
        "planner_valid_above_unified_max5": float(valid["candidate"])
        > planner_valid_baseline,
        "reason_search_above_matched": float(reason_search["candidate"])
        > matched_reason_search,
    }
    return {
        "label": label,
        "passes_all_gates": all(gates.values()),
        "gates": gates,
        "reasoner_semantic": {
            "candidate": float(semantic["candidate"]),
            "delta": float(semantic["delta"]),
            "bootstrap_95ci": semantic_ci,
        },
        "planner": {
            "valid": float(valid["candidate"]),
            "valid_delta": float(valid["delta"]),
            "reason_search": float(reason_search["candidate"]),
            "reason_search_delta": float(reason_search["delta"]),
            "over_five_nodes": float(over_five["candidate"]),
            "over_five_nodes_delta": float(over_five["delta"]),
        },
    }


def pareto_frontier(candidates: list[dict[str, Any]]) -> list[str]:
    passing = [candidate for candidate in candidates if candidate["passes_all_gates"]]
    frontier = []
    for candidate in passing:
        valid = candidate["planner"]["valid"]
        reason_search = candidate["planner"]["reason_search"]
        dominated = any(
            other["label"] != candidate["label"]
            and other["planner"]["valid"] >= valid
            and other["planner"]["reason_search"] >= reason_search
            and (
                other["planner"]["valid"] > valid
                or other["planner"]["reason_search"] > reason_search
            )
            for other in passing
        )
        if not dominated:
            frontier.append(candidate["label"])
    return sorted(frontier)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DAGENT Planner-Dose Mechanism Selection",
        "",
        "| Candidate | Pass | Reasoner semantic delta (95% CI) | Planner valid | Reason->Search | Over-five |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        semantic = candidate["reasoner_semantic"]
        low, high = semantic["bootstrap_95ci"]
        planner = candidate["planner"]
        lines.append(
            f"| {candidate['label']} | {str(candidate['passes_all_gates']).lower()} | "
            f"{semantic['delta']:+.6f} [{low:+.6f}, {high:+.6f}] | "
            f"{planner['valid']:.6f} | {planner['reason_search']:.6f} | "
            f"{planner['over_five_nodes']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Pareto candidates: "
            + (", ".join(report["pareto_candidates"]) or "none"),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoner", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--planner", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--planner-valid-baseline", type=float, default=0.9513846153846154)
    parser.add_argument("--matched-reason-search", type=float, default=1.0 / 1625.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reasoner_paths = dict(parse_assignment(value, "--reasoner") for value in args.reasoner)
    planner_paths = dict(parse_assignment(value, "--planner") for value in args.planner)
    if set(reasoner_paths) != set(planner_paths):
        raise RuntimeError(
            "Reasoner/Planner candidate labels differ: "
            f"{sorted(reasoner_paths)} != {sorted(planner_paths)}"
        )

    candidates = []
    for label in sorted(reasoner_paths):
        candidates.append(
            evaluate_candidate(
                label,
                load_comparison(reasoner_paths[label], label),
                load_comparison(planner_paths[label], label),
                planner_valid_baseline=args.planner_valid_baseline,
                matched_reason_search=args.matched_reason_search,
            )
        )
    report = {
        "schema_version": 1,
        "selection_scope": "exploratory_mechanism_only",
        "thresholds": {
            "planner_valid_baseline": args.planner_valid_baseline,
            "matched_reason_search": args.matched_reason_search,
            "reasoner_rule": "upper endpoint of paired 95% CI must be >= 0",
        },
        "candidates": candidates,
        "pareto_candidates": pareto_frontier(candidates),
        "strict_replication_required": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "selection.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
