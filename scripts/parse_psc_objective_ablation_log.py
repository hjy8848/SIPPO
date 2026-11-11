"""Parse one frozen current-policy PSC objective-ablation audit log."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VARIANTS = (
    "real",
    "shuffled",
    "cross_question",
    "no_length_normalization",
    "pair_weighted",
    "one_random_pair",
)


def metric(line: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}:{NUMBER}", line)
    return float(match.group(0).split(":", 1)[1]) if match else None


def svg(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 860, 430
    left, right, top, bottom = 76, 24, 30, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    panels = [
        ("planner_em_cosine", "Cosine with Planner-EM"),
        ("planner_em_norm_ratio", "PSC / Planner-EM norm ratio"),
    ]
    colors = ["#8f3b35", "#7b8794", "#486b54", "#a66a2c", "#1f4e79", "#6c5b7b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    panel_w = plot_w / 2 - 20
    for panel_idx, (field, title) in enumerate(panels):
        x0 = left + panel_idx * (panel_w + 40)
        y0 = top
        y_min, y_max = (-1.0, 1.0) if field == "planner_em_cosine" else (0.0, 0.8)
        parts.append(
            f'<text x="{x0 + panel_w/2:.1f}" y="18" text-anchor="middle" '
            f'font-family="serif" font-size="15">{title}</text>'
        )
        parts.extend(
            [
                f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+plot_h}" stroke="#333"/>',
                f'<line x1="{x0}" y1="{y0+plot_h}" x2="{x0+panel_w}" y2="{y0+plot_h}" stroke="#333"/>',
            ]
        )
        for tick in ([-1.0, -0.5, 0.0, 0.5, 1.0] if field == "planner_em_cosine" else [0.0, 0.2, 0.4, 0.6, 0.8]):
            y = y0 + plot_h * (1 - (tick - y_min) / (y_max - y_min))
            parts.append(
                f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" '
                f'font-family="serif" font-size="12">{tick:.1f}</text>'
            )
        bar_w = panel_w / len(rows) * 0.62
        for i, row in enumerate(rows):
            value = row.get(field)
            if value is None:
                continue
            value = float(value)
            cx = x0 + panel_w * (i + 0.5) / len(rows)
            y = y0 + plot_h * (1 - (value - y_min) / (y_max - y_min))
            zero = y0 + plot_h * (1 - (0 - y_min) / (y_max - y_min))
            top_y, h = min(y, zero), abs(zero - y)
            parts.extend(
                [
                    f'<rect x="{cx-bar_w/2:.1f}" y="{top_y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[i]}"/>',
                    f'<text x="{cx:.1f}" y="{top_y-6:.1f}" text-anchor="middle" font-family="serif" font-size="11">{value:.3f}</text>',
                    f'<text x="{cx:.1f}" y="{y0+plot_h+18}" text-anchor="middle" font-family="serif" font-size="10" transform="rotate(-32 {cx:.1f} {y0+plot_h+18})">{row["variant"]}</text>',
                ]
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python parse_psc_objective_ablation_log.py LOG")
    log = Path(sys.argv[1]).resolve()
    text = log.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if "step:1 -" in line]
    if not lines:
        raise SystemExit("no completed step:1 metrics found")
    line = lines[-1]
    out = log.parent.parent.parent / "followup_experiments_20260916_r1" / "psc_objective_ablation"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        prefix = f"aux/planner_outcome_contrast/ablation/{variant}/"
        rows.append(
            {
                "variant": variant,
                "pair_count": metric(line, prefix + "pair_count"),
                "informative_group_count": metric(line, prefix + "informative_group_count"),
                "gradient_norm": metric(line, prefix + "gradient_norm"),
                "planner_em_norm_ratio": metric(line, prefix + "planner_em_norm_ratio"),
                "planner_em_cosine": metric(line, prefix + "planner_em_cosine"),
            }
        )
    if any(row["planner_em_cosine"] is None for row in rows):
        raise SystemExit("one or more objective-ablation metrics are missing")
    with (out / "objective_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "objective_ablation.json").write_text(
        json.dumps(
            {
                "source_log": str(log),
                "mode": "frozen current-policy coefficient replay",
                "optimizer_steps_expected": metric(line, "actor/optimizer_steps_expected"),
                "optimizer_steps_executed": metric(line, "actor/optimizer_steps_executed"),
                "workflows": metric(line, "train/workflow/completed"),
                "incomplete_question_groups": metric(line, "train/workflow/incomplete_question_groups"),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    svg(rows, out / "objective_ablation.svg")


if __name__ == "__main__":
    main()
