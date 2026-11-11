"""Build an offline PSC lambda sensitivity audit from frozen gradient geometry."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "exp" / "followup_experiments_20260916_r1" / "psc_objective_audit" / "current_policy_audit.json"
OUT = ROOT / "exp" / "followup_experiments_20260916_r1" / "psc_lambda_sensitivity"
LAMBDAS = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]


def write_fallback_svg(path: Path, rows: list[dict[str, float]]) -> None:
    """Keep the figure reproducible when the local plotting stack is absent."""
    width, height = 780, 300
    panel_lefts = [65, 420]
    plot_w, plot_h = 270, 190
    top = 35
    x_min, x_max = math.log2(LAMBDAS[0]), math.log2(LAMBDAS[-1])

    def xcoord(value: float, left: float) -> float:
        return left + (math.log2(value) - x_min) / (x_max - x_min) * plot_w

    def panel(left: float, values: list[float], ymin: float, ymax: float, ylabel: str) -> str:
        parts = [f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#222"/>']
        parts.append(f'<text x="{left + plot_w / 2}" y="285" text-anchor="middle" font-size="12">lambda_PSC</text>')
        parts.append(f'<text x="{left - 48}" y="{top + plot_h / 2}" text-anchor="middle" font-size="11" transform="rotate(-90 {left - 48},{top + plot_h / 2})">{ylabel}</text>')
        points = []
        for lam, value in zip(LAMBDAS, values):
            x = xcoord(lam, left)
            y = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
            points.append(f"{x:.2f},{y:.2f}")
            parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 4}" stroke="#222"/>')
            parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="8" transform="rotate(35 {x:.2f},{top + plot_h + 18})">{lam:g}</text>')
        base_x = xcoord(0.02, left)
        parts.append(f'<line x1="{base_x:.2f}" y1="{top}" x2="{base_x:.2f}" y2="{top + plot_h}" stroke="#666" stroke-dasharray="4,3"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#8f2f2f" stroke-width="2"/>')
        for point in points:
            px, py = point.split(",")
            parts.append(f'<circle cx="{px}" cy="{py}" r="3" fill="#8f2f2f"/>')
        return "".join(parts)

    ratio = [row["psc_to_planner_em_norm_ratio"] for row in rows]
    cosine = [row["combined_update_to_planner_em_cosine"] for row in rows]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append(panel(panel_lefts[0], ratio, 0.0, 1.25, "norm ratio"))
    svg.append(panel(panel_lefts[1], cosine, 0.78, 1.01, "combined cosine"))
    svg.append('<text x="390" y="18" text-anchor="middle" font-size="13">Offline PSC gradient-geometry sensitivity</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    real = source["real"]
    base_lambda = 0.02
    base_ratio = float(real["weighted_psc_to_planner_em_norm_ratio"])
    base_cosine = float(real["weighted_psc_planner_em_cosine"])

    rows = []
    for lam in LAMBDAS:
        ratio = base_ratio * (lam / base_lambda)
        denominator = math.sqrt(1.0 + ratio * ratio + 2.0 * ratio * base_cosine)
        combined_cosine = (1.0 + ratio * base_cosine) / denominator
        rows.append(
            {
                "lambda_psc": lam,
                "psc_to_planner_em_norm_ratio": ratio,
                "combined_update_to_planner_em_cosine": combined_cosine,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "lambda_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "source": str(SOURCE.relative_to(ROOT)),
        "mode": "offline_gradient_geometry",
        "not_retrained": True,
        "base_lambda": base_lambda,
        "base_psc_to_planner_em_norm_ratio": base_ratio,
        "base_psc_planner_em_cosine": base_cosine,
        "lambdas": rows,
        "formula": {
            "ratio": "r(lambda) = r(0.02) * lambda / 0.02",
            "combined_cosine": "(1 + r*c) / sqrt(1 + r^2 + 2*r*c)",
        },
    }
    (OUT / "lambda_sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        x = [row["lambda_psc"] for row in rows]
        y_ratio = [row["psc_to_planner_em_norm_ratio"] for row in rows]
        y_cos = [row["combined_update_to_planner_em_cosine"] for row in rows]
        plt.rcParams.update({"font.family": "serif", "font.size": 9})
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.45), constrained_layout=True)
        for ax, y, ylabel in [
            (axes[0], y_ratio, r"$||\lambda g_{PSC}|| / ||g_{PlannerEM}||$"),
            (axes[1], y_cos, r"$\cos(g_{EM}+\lambda g_{PSC}, g_{EM})$"),
        ]:
            ax.plot(x, y, color="#8f2f2f", marker="o", markersize=3.5, linewidth=1.6)
            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in x], rotation=35, ha="right")
            ax.set_xlabel(r"$\lambda_{PSC}$")
            ax.set_ylabel(ylabel)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].axhspan(0.10, 0.30, color="#777777", alpha=0.10, linewidth=0)
        axes[0].axvline(base_lambda, color="#444444", linestyle="--", linewidth=0.8)
        axes[1].axvline(base_lambda, color="#444444", linestyle="--", linewidth=0.8)
        fig.savefig(OUT / "lambda_sensitivity.svg", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        write_fallback_svg(OUT / "lambda_sensitivity.svg", rows)
        (OUT / "plot_error.txt").write_text(f"{type(exc).__name__}: {exc}; used standard-library SVG fallback.\n", encoding="utf-8")

    manifest = {
        "artifact": "PSC offline lambda sensitivity",
        "source": str(SOURCE.relative_to(ROOT)),
        "output_files": ["lambda_sensitivity.csv", "lambda_sensitivity.json", "lambda_sensitivity.svg"],
        "interpretation_boundary": "This is frozen gradient geometry, not a retrained hyperparameter sweep.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
