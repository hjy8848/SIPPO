"""Parse and plot the frozen PSC permutation-null artifact."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_svg(path: Path, values: list[float], real: float) -> None:
    width, height = 720, 320
    left, top, plot_w, plot_h = 70, 35, 610, 220
    lo, hi = -1.0, 1.0
    bins = 30
    counts = [0] * bins
    for value in values:
        index = int((value - lo) / (hi - lo) * bins)
        counts[min(max(index, 0), bins - 1)] += 1
    max_count = max(counts, default=1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14">PSC shuffled-label cosine null</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    bar_w = plot_w / bins
    for index, count in enumerate(counts):
        bar_h = (count / max_count) * plot_h
        x = left + index * bar_w
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w - 1:.2f}" height="{bar_h:.2f}" fill="#999999" opacity="0.72"/>'
        )
    real_x = left + (real - lo) / (hi - lo) * plot_w
    parts.append(f'<line x1="{real_x:.2f}" y1="{top}" x2="{real_x:.2f}" y2="{top + plot_h}" stroke="#8f2f2f" stroke-width="2"/>')
    parts.append(f'<text x="{min(real_x + 6, left + plot_w - 120):.2f}" y="{top + 16}" fill="#8f2f2f" font-size="11">real = {real:.3f}</text>')
    for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
        x = left + (tick - lo) / (hi - lo) * plot_w
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#222"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 19}" text-anchor="middle" font-size="10">{tick:g}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="305" text-anchor="middle" font-size="12">cosine with Planner-EM</text>')
    parts.append(f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" font-size="11" transform="rotate(-90 18,{top + plot_h / 2})">count</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: parse_psc_permutation_null.py ARTIFACT.json [OUTPUT_DIR]")
    source = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else source.parent
    payload = json.loads(source.read_text(encoding="utf-8"))
    values = [float(value) for value in payload["null_cosines"]]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("null cosine list must be non-empty and finite")
    expected = int(payload["permutation_count"])
    if len(values) != expected:
        raise ValueError(f"expected {expected} permutations, found {len(values)}")
    real = float(payload["real_planner_em_cosine"])
    exceedances = int(sum(value >= real for value in values))
    p_value = (1 + exceedances) / (expected + 1)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "permutation_null.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["permutation_index", "planner_em_cosine"])
        writer.writerows(enumerate(values, start=1))

    summary = {
        "source": str(source),
        "mode": payload.get("mode"),
        "permutation_count": expected,
        "real_planner_em_cosine": real,
        "null_mean_cosine": statistics.fmean(values),
        "null_std_cosine": statistics.pstdev(values),
        "null_q10_cosine": quantile(values, 0.10),
        "null_q50_cosine": quantile(values, 0.50),
        "null_q90_cosine": quantile(values, 0.90),
        "exceedances_ge_real": exceedances,
        "p_value": p_value,
        "optimizer_steps_expected": int(payload.get("optimizer_steps_expected", 0)),
        "optimizer_steps_executed": int(payload.get("optimizer_steps_executed", 0)),
    }
    (out / "permutation_null_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_svg(out / "permutation_null.svg", values, real)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "artifact": "PSC shuffled-label permutation null",
                "source": str(source),
                "outputs": ["permutation_null.csv", "permutation_null_summary.json", "permutation_null.svg"],
                "interpretation_boundary": "Frozen current-policy mechanism audit; no optimizer update or downstream training claim.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
