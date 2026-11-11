#!/usr/bin/env python3
"""Archive the completed causal-reward audit beside the follow-up summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report_snapshot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for name, candidate in report.get("candidates", {}).items():
        test = candidate.get("test", {})
        rows.append(
            {
                "candidate": name,
                "status": "passed" if candidate.get("passed") else "failed",
                "test_cases": test.get("cases"),
                "changed_pairs": test.get("changed_pairs"),
                "direction_accuracy": test.get("direction_accuracy"),
                "direction_ci_low": (test.get("direction_accuracy_cluster_bootstrap_95ci") or [None])[0],
                "selection_em_gain": test.get("selection_em_gain_over_random"),
                "selection_gain_ci_low": (test.get("selection_gain_bootstrap_95ci") or [None])[0],
            }
        )
    with (args.output_dir / "causal_reward_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "dagent-causal-reward-heuristic-summary-v1",
        "source_report": str(args.source_report),
        "candidates": len(rows),
        "note": "Frozen causal-replay audit; not a retraining comparison.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Causal reward versus heuristic summary\n\n"
        "This archive summarizes the completed v3_400 frozen causal-replay audit. "
        "It compares candidate signals by their ability to predict factual versus "
        "counterfactual terminal EM on held-out trajectories. It is mechanism evidence, "
        "not a retraining result. `query_rewrite_delta` failed its minimum changed-pair "
        "gate and is retained as a negative audit outcome.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
