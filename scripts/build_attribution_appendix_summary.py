"""Build appendix-ready summaries from the frozen attribution audit.

This script only re-aggregates the checked-in JSON report.  It does not run
inference, fit a new model, or change any threshold in the source audit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "exp" / "analysis_outputs_attribution_reward_audit_shared_v2" / "report.json"
OUT = ROOT / "exp" / "followup_experiments_20260916_r1" / "attribution_audit"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def svg_bar(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 760, 430
    left, right, top, bottom = 82, 24, 34, 84
    plot_w, plot_h = width - left - right, height - top - bottom
    max_auc = 1.0
    bar_w = plot_w / max(len(rows), 1) * 0.58
    colors = ["#1f4e79", "#8f3b35", "#6f7f8f", "#a66a2c", "#486b54"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" '
        f'y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#333" stroke-width="1"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1.0 - tick / max_auc)
        parts.append(
            f'<line x1="{left-4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" '
            f'stroke="#333"/><text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
            f'font-family="serif" font-size="14">{tick:.2f}</text>'
        )
    for i, row in enumerate(rows):
        auc = float(row["auc"])
        cx = left + plot_w * (i + 0.5) / len(rows)
        x = cx - bar_w / 2
        y = top + plot_h * (1.0 - auc)
        h = top + plot_h - y
        color = colors[i % len(colors)]
        label = str(row["candidate"]).replace("_", " ")
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{color}" fill-opacity="0.9"/>',
                f'<text x="{cx:.1f}" y="{y-7:.1f}" text-anchor="middle" '
                f'font-family="serif" font-size="13">{auc:.3f}</text>',
                f'<text x="{cx:.1f}" y="{top+plot_h+20}" text-anchor="middle" '
                f'font-family="serif" font-size="12" transform="rotate(-28 {cx:.1f} '
                f'{top+plot_h+20})">{label}</text>',
            ]
        )
    parts.append(
        f'<text x="{left + plot_w/2:.1f}" y="{height-10}" text-anchor="middle" '
        'font-family="serif" font-size="15">Attribution signal</text>'
    )
    parts.append(
        f'<text x="18" y="{top + plot_h/2:.1f}" text-anchor="middle" '
        f'font-family="serif" font-size="15" transform="rotate(-90 18 {top + plot_h/2:.1f})">'
        'ROC AUC</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    candidates = report["candidates"]
    OUT.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    for name, result in candidates.items():
        test = result.get("calibrated_test", {})
        ci = result.get("mean_gap_bootstrap_95ci", [None, None])
        summary_rows.append(
            {
                "candidate": name,
                "count": result.get("count"),
                "auc": result.get("auc"),
                "mean_gap": result.get("mean_gap"),
                "mean_gap_ci_low": ci[0],
                "mean_gap_ci_high": ci[1],
                "calibrated_balanced_accuracy": test.get("balanced_accuracy"),
                "calibrated_precision": test.get("positive_precision"),
                "calibrated_recall": test.get("positive_recall"),
            }
        )
        for dataset, ds in result.get("by_source", {}).items():
            dataset_rows.append(
                {
                    "candidate": name,
                    "dataset": dataset,
                    "count": ds.get("count"),
                    "auc": ds.get("auc"),
                }
            )

    for filename, rows in (("attribution_summary.csv", summary_rows), ("attribution_by_dataset.csv", dataset_rows)):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "source_report": str(REPORT.relative_to(ROOT)),
        "audit_mode": report.get("audit_mode"),
        "acceptance_gate": report.get("acceptance_gate"),
        "candidates": summary_rows,
        "by_dataset": dataset_rows,
        "interpretation": "Frozen offline attribution audit; no retraining or new inference was run by this script.",
    }
    write_json(OUT / "manifest.json", payload)
    write_json(OUT / "report_snapshot.json", report)
    svg_bar(summary_rows, OUT / "attribution_auc.svg")
    (OUT / "README.md").write_text(
        "# Attribution audit summary\n\n"
        "This directory re-aggregates the frozen shared-v2 attribution audit. "
        "The five candidate signals are reported with sample count, ROC AUC, "
        "bootstrap mean-gap interval, and dataset-specific AUCs. The SVG is an "
        "appendix-ready diagnostic, not a training result. It does not establish "
        "that any signal improves downstream policy performance.\n\n"
        f"Source: `{REPORT.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
