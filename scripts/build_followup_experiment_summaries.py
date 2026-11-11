#!/usr/bin/env python3
"""Build reproducible, no-training follow-up summaries for the DAGENT paper.

This script only re-aggregates frozen local evidence. It does not create new
model outputs and does not turn descriptive diagnostics into causal claims.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "exp" / "paper_experiment_archive_20260916_r1"
CURVES = ROOT / "exp" / "training_curves_3b_20260916_r1"
OUT = ROOT / "exp" / "followup_experiments_20260916_r1"


DATASET_NAMES = {
    "2wikimultihopqa": "2Wiki",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "bamboogle": "Bamboogle",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def f(value: str | float | int | None) -> float:
    return float(value) if value not in (None, "") else float("nan")


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else float("nan")


def dataset_name(value: str) -> str:
    return DATASET_NAMES.get(value, value)


def paired_summary() -> dict[str, object]:
    source = ARCHIVE / "3b_diagnostics" / "paired" / "paired_questions.csv"
    rows = read_csv(source)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[dataset_name(row["dataset"])].append(row)
    groups["All"] = rows

    summary_rows: list[dict[str, object]] = []
    summary_json: dict[str, object] = {
        "source": str(source),
        "reference": "PureEM-step468",
        "candidate": "OldGroupedPSC-step468",
        "interpretation": "Paired descriptive comparison on the frozen 1,625-question Formal-v2 alignment.",
        "datasets": {},
    }
    for name, subset in groups.items():
        pureem_only = sum(f(row["reference_em"]) == 1 and f(row["candidate_em"]) == 0 for row in subset)
        psc_only = sum(f(row["reference_em"]) == 0 and f(row["candidate_em"]) == 1 for row in subset)
        both_correct = sum(f(row["reference_em"]) == 1 and f(row["candidate_em"]) == 1 for row in subset)
        both_wrong = sum(f(row["reference_em"]) == 0 and f(row["candidate_em"]) == 0 for row in subset)
        n = len(subset)
        item = {
            "dataset": name,
            "n": n,
            "pureem_only": pureem_only,
            "psc_only": psc_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "pureem_em": finite_mean([f(row["reference_em"]) for row in subset]),
            "psc_em": finite_mean([f(row["candidate_em"]) for row in subset]),
            "em_gain_pp_psc_minus_pureem": 100.0 * finite_mean([f(row["delta_em"]) for row in subset]),
            "pureem_f1": finite_mean([f(row["reference_f1"]) for row in subset]),
            "psc_f1": finite_mean([f(row["candidate_f1"]) for row in subset]),
            "f1_gain_pp_psc_minus_pureem": 100.0 * finite_mean([f(row["delta_f1"]) for row in subset]),
        }
        summary_rows.append(item)
        summary_json["datasets"][name] = item

    write_csv(OUT / "paired_win_tie_loss.csv", summary_rows)
    write_json(OUT / "paired_win_tie_loss.json", summary_json)
    return summary_json


def read_grouped_csv(method: str, filename: str) -> list[dict[str, str]]:
    return read_csv(ARCHIVE / "3b_full_answerable" / method / filename)


def topology_complexity_summary() -> dict[str, object]:
    output: dict[str, object] = {
        "sources": {},
        "topology": [],
        "complexity": [],
        "interpretation": "Descriptive association with generated workflow structure; not a causal topology claim.",
    }
    for analysis_type, filename, key in (
        ("topology", "topology_analysis.csv", "topology"),
        ("complexity", "complexity_analysis.csv", "complexity"),
    ):
        pure_rows = read_grouped_csv("pureem", filename)
        psc_rows = read_grouped_csv("psc", filename)
        pure = {row[key]: row for row in pure_rows}
        psc = {row[key]: row for row in psc_rows}
        rows: list[dict[str, object]] = []
        for group in sorted(set(pure) | set(psc)):
            p = pure.get(group, {})
            c = psc.get(group, {})
            rows.append(
                {
                    "group": group,
                    "pureem_share_pct": 100.0 * f(p.get("percent_data")),
                    "psc_share_pct": 100.0 * f(c.get("percent_data")),
                    "share_delta_pp": 100.0 * (f(c.get("percent_data")) - f(p.get("percent_data"))),
                    "pureem_em": f(p.get("em")),
                    "psc_em": f(c.get("em")),
                    "em_delta_pp": 100.0 * (f(c.get("em")) - f(p.get("em"))),
                    "pureem_valid_plan": f(p.get("valid_plan")),
                    "psc_valid_plan": f(c.get("valid_plan")),
                    "pureem_retrieval_calls": f(p.get("avg_retrieval_calls")),
                    "psc_retrieval_calls": f(c.get("avg_retrieval_calls")),
                    "pureem_num_samples": int(f(p.get("num_samples"))) if p else 0,
                    "psc_num_samples": int(f(c.get("num_samples"))) if c else 0,
                }
            )
        output[analysis_type] = rows
        output["sources"][analysis_type] = {
            "pureem": str(ARCHIVE / "3b_full_answerable" / "pureem" / filename),
            "psc": str(ARCHIVE / "3b_full_answerable" / "psc" / filename),
        }
        write_csv(OUT / f"{analysis_type}_pureem_vs_psc.csv", rows)
    write_json(OUT / "topology_complexity_summary.json", output)
    return output


def signal_availability_summary() -> dict[str, object]:
    source = ARCHIVE / "3b_diagnostics" / "psc_training_lineage" / "psc_training_signal.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    run = payload["runs"]["OldPSC-3B-step468"]
    per_step = run["per_step"]
    groups_per_step = 32  # 128 workflows with four siblings in this run.
    rows: list[dict[str, object]] = []
    for item in per_step:
        p = f(item.get("rollout_em"))
        rows.append(
            {
                "step": int(item["step"]),
                "rollout_em": p,
                "best_of_n_em": f(item.get("best_of_n_em")),
                "valid_rate": f(item.get("valid_rate")),
                "informative_questions": f(item.get("estimated_informative_questions")),
                "informative_fraction": f(item.get("estimated_informative_questions")) / groups_per_step,
                "pairs": f(item.get("estimated_pairs")),
                "pairs_per_group": f(item.get("estimated_pairs")) / groups_per_step,
                "gradient_ratio": f(item.get("interpretable_gradient_ratio")),
                "gradient_cosine": f(item.get("gradient_cosine")),
                "theoretical_mixed_outcome_reference": 1.0 - p**4 - (1.0 - p) ** 4,
            }
        )
    totals = {
        "steps": len(rows),
        "groups_per_step": groups_per_step,
        "total_question_groups": len(rows) * groups_per_step,
        "total_informative_questions": sum(row["informative_questions"] for row in rows),
        "total_pairs": sum(row["pairs"] for row in rows),
        "informative_fraction": sum(row["informative_questions"] for row in rows) / (len(rows) * groups_per_step),
        "mean_pairs_per_group": sum(row["pairs"] for row in rows) / (len(rows) * groups_per_step),
        "signal_present_every_step": run["audit"]["signal_present_every_step"],
        "positive_alignment_rate": run["audit"]["positive_alignment_rate"],
        "interpretation": "Theoretical mixed-outcome rate assumes independent binary sibling outcomes; it is a reference, not an observed signal estimate.",
    }
    result = {
        "source": str(source),
        "run": "OldPSC-3B-step468",
        "totals": totals,
        "per_step": rows,
    }
    write_csv(OUT / "psc_signal_availability.csv", rows)
    write_json(OUT / "psc_signal_availability.json", result)
    return result


def recovery_summary() -> dict[str, object]:
    boundaries = (234, 382)
    window = 20
    metric_names = {
        "train/rollout/em_aggregate": "rollout_em",
        "train/rollout/best_of_n_em": "best_of_n_em",
        "train/workflow/valid_rate": "valid_rate",
        "train/workflow/retrieval_calls_mean": "retrieval_calls",
        "actor/grad_norm": "grad_norm",
        "actor/ppo_kl": "ppo_kl",
    }
    method_files = {
        "PUREEM": CURVES / "pureem_curve.csv",
        "PSC": CURVES / "psc_curve.csv",
    }
    all_rows: list[dict[str, object]] = []
    for method, path in method_files.items():
        rows = read_csv(path)
        by_step = {int(row["step"]): row for row in rows}
        for boundary in boundaries:
            pre = [by_step[s] for s in range(boundary - window + 1, boundary + 1) if s in by_step]
            post = [by_step[s] for s in range(boundary + 1, boundary + window + 1) if s in by_step]
            for raw_name, short_name in metric_names.items():
                pre_mean = finite_mean([f(row.get(raw_name)) for row in pre])
                post_mean = finite_mean([f(row.get(raw_name)) for row in post])
                all_rows.append(
                    {
                        "method": method,
                        "boundary_step": boundary,
                        "window": window,
                        "metric": short_name,
                        "pre_mean": pre_mean,
                        "post_mean": post_mean,
                        "delta_post_minus_pre": post_mean - pre_mean if math.isfinite(pre_mean) and math.isfinite(post_mean) else float("nan"),
                        "pre_count": len(pre),
                        "post_count": len(post),
                    }
                )
    result = {
        "sources": {method: str(path) for method, path in method_files.items()},
        "boundaries": list(boundaries),
        "window": window,
        "rows": all_rows,
        "interpretation": "Windowed pre/post descriptive audit around engineering recovery boundaries; it does not isolate a training-cause effect.",
    }
    write_csv(OUT / "recovery_boundary_audit.csv", all_rows)
    write_json(OUT / "recovery_boundary_audit.json", result)
    return result


def make_plots() -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        make_svg_plots()
        return

    boundaries = (234, 382)
    method_files = {
        "PUREEM": CURVES / "pureem_curve.csv",
        "PSC": CURVES / "psc_curve.csv",
    }

    # Paired win/tie/loss composition.
    paired = json.loads((OUT / "paired_win_tie_loss.json").read_text(encoding="utf-8"))["datasets"]
    names = [name for name in ("2Wiki", "HotpotQA", "MuSiQue", "Bamboogle") if name in paired]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for key, label, color in (
        ("psc_only", "PSC only", "#9a3f3f"),
        ("both_correct", "Both correct", "#b8b8b8"),
        ("both_wrong", "Both wrong", "#eeeeee"),
        ("pureem_only", "PUREEM only", "#234f7d"),
    ):
        values = np.array([paired[name][key] / paired[name]["n"] for name in names])
        ax.bar(x, values, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.4)
        bottom += values
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of paired questions")
    ax.legend(frameon=True, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "paired_win_tie_loss.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    signal = json.loads((OUT / "psc_signal_availability.json").read_text(encoding="utf-8"))["per_step"]
    steps = np.array([row["step"] for row in signal])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    axes[0].plot(steps, [row["informative_fraction"] for row in signal], color="#9a3f3f", label="Observed informative fraction")
    axes[0].plot(steps, [row["theoretical_mixed_outcome_reference"] for row in signal], color="#777777", linestyle="--", label="Independent mixed-outcome reference")
    axes[0].set_xlabel("Rollout iteration")
    axes[0].set_ylabel("Fraction")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(steps, [row["pairs"] for row in signal], color="#234f7d", label="PSC pairs / step")
    axes[1].set_xlabel("Rollout iteration")
    axes[1].set_ylabel("Pairs")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "psc_signal_availability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Recovery boundary audit, using the original per-iteration series.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for method, path, color in (("PUREEM", method_files["PUREEM"], "#234f7d"), ("PSC", method_files["PSC"], "#9a3f3f")):
        rows = read_csv(path)
        steps = np.array([int(row["step"]) for row in rows])
        axes[0].plot(steps, [f(row["train/rollout/em_aggregate"]) for row in rows], color=color, alpha=0.22)
        axes[0].plot(steps, [f(row["train/rollout/best_of_n_em"]) for row in rows], color=color, alpha=0.75, label=f"{method} best-of-n")
        axes[1].plot(steps, [f(row["train/workflow/retrieval_calls_mean"]) for row in rows], color=color, alpha=0.75, label=method)
    for boundary in boundaries:
        for ax in axes:
            ax.axvline(boundary, color="#aaaaaa", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Rollout iteration")
    axes[0].set_ylabel("EM")
    axes[1].set_xlabel("Rollout iteration")
    axes[1].set_ylabel("Retrieval calls / workflow")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "recovery_boundary_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    write_json(OUT / "plot_status.json", {"available": True, "backend": "matplotlib", "files": ["paired_win_tie_loss.png", "psc_signal_availability.png", "recovery_boundary_audit.png"]})


def svg_escape(value: object) -> str:
    return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def svg_text(x: float, y: float, value: object, size: int = 12, anchor: str = "start", color: str = "#222222") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Georgia, Times New Roman, serif" font-size="{size}px" text-anchor="{anchor}" fill="{color}">{svg_escape(value)}</text>'


def svg_polyline(values: list[float], x0: float, y0: float, width: float, height: float, lo: float, hi: float, color: str, opacity: float = 1.0, stroke_width: float = 1.8, dash: str = "") -> str:
    points: list[str] = []
    span = max(hi - lo, 1e-9)
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        x = x0 + width * index / max(len(values) - 1, 1)
        y = y0 + height * (hi - value) / span
        points.append(f"{x:.2f},{y:.2f}")
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-opacity="{opacity}" stroke-width="{stroke_width}"{dash_attr}/>'


def svg_frame(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def make_svg_plots() -> None:
    """Dependency-free SVG fallbacks for the paper-friendly diagnostics."""
    paired = json.loads((OUT / "paired_win_tie_loss.json").read_text(encoding="utf-8"))["datasets"]
    names = [name for name in ("2Wiki", "HotpotQA", "MuSiQue", "Bamboogle") if name in paired]
    width, height = 760, 390
    parts = svg_frame(width, height)
    x0, y0, plot_w, plot_h = 88, 48, 620, 260
    parts.extend([svg_text(width / 2, 27, "Paired outcome composition", 16, "middle"),
                  f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#333"/>',
                  f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#333"/>'])
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + plot_h * (1 - tick)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(svg_text(x0 - 10, y + 4, f"{tick:.2f}", 10, "end", "#555555"))
    colors = (("psc_only", "PSC only", "#9a3f3f"), ("both_correct", "Both correct", "#b8b8b8"), ("both_wrong", "Both wrong", "#eeeeee"), ("pureem_only", "PUREEM only", "#234f7d"))
    bar_w = 72
    for i, name in enumerate(names):
        x = x0 + (i + 0.5) * plot_w / len(names) - bar_w / 2
        bottom = y0 + plot_h
        for key, _, color in colors:
            fraction = paired[name][key] / paired[name]["n"]
            bar_h = fraction * plot_h
            parts.append(f'<rect x="{x:.1f}" y="{bottom - bar_h:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" stroke="white" stroke-width="0.6"/>')
            bottom -= bar_h
        parts.append(svg_text(x + bar_w / 2, y0 + plot_h + 23, name, 11, "middle"))
    legend_x, legend_y = 122, 350
    for i, (_, label, color) in enumerate(colors):
        lx = legend_x + i * 150
        parts.append(f'<rect x="{lx}" y="{legend_y - 11}" width="12" height="12" fill="{color}" stroke="#aaaaaa" stroke-width="0.4"/>')
        parts.append(svg_text(lx + 17, legend_y, label, 10))
    parts.append('</svg>')
    (OUT / "paired_win_tie_loss.svg").write_text("\n".join(parts), encoding="utf-8")

    signal = json.loads((OUT / "psc_signal_availability.json").read_text(encoding="utf-8"))["per_step"]
    steps = [row["step"] for row in signal]
    width, height = 900, 350
    parts = svg_frame(width, height)
    for panel, title, series, lo, hi, y_label in (
        (0, "Signal availability", [([row["informative_fraction"] for row in signal], "#9a3f3f", 1.0), ([row["theoretical_mixed_outcome_reference"] for row in signal], "#777777", 0.7)], 0.0, 1.0, "fraction"),
        (1, "PSC pairs per rollout iteration", [([row["pairs"] for row in signal], "#234f7d", 1.0)], 0.0, max(row["pairs"] for row in signal) * 1.05, "pairs"),
    ):
        x0, y0, plot_w, plot_h = 70 + panel * 430, 55, 360, 235
        parts.append(svg_text(x0 + plot_w / 2, 28, title, 15, "middle"))
        parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#333"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#333"/>')
        for values, color, opacity in series:
            dash = "6,4" if color == "#777777" else ""
            parts.append(svg_polyline(values, x0, y0, plot_w, plot_h, lo, hi, color, opacity, 1.6, dash))
        parts.append(svg_text(x0 - 12, y0 + plot_h / 2, y_label, 11, "middle", "#555555"))
        parts.append(svg_text(x0 + plot_w / 2, y0 + plot_h + 33, "Rollout iteration", 11, "middle"))
    parts.extend([svg_text(100, 330, "Observed informative fraction", 10, "start", "#9a3f3f"), svg_text(310, 330, "Independent mixed-outcome reference", 10, "start", "#777777"), svg_text(610, 330, "Pairs", 10, "start", "#234f7d"), '</svg>'])
    (OUT / "psc_signal_availability.svg").write_text("\n".join(parts), encoding="utf-8")

    boundaries = (234, 382)
    method_files = {"PUREEM": CURVES / "pureem_curve.csv", "PSC": CURVES / "psc_curve.csv"}
    series_data: dict[str, list[dict[str, str]]] = {method: read_csv(path) for method, path in method_files.items()}
    width, height = 900, 350
    parts = svg_frame(width, height)
    for panel, raw_key, title, y_label, colors in (
        (0, "train/rollout/best_of_n_em", "Best-of-n EM", "EM", {"PUREEM": "#234f7d", "PSC": "#9a3f3f"}),
        (1, "train/workflow/retrieval_calls_mean", "Retrieval calls", "calls / workflow", {"PUREEM": "#234f7d", "PSC": "#9a3f3f"}),
    ):
        x0, y0, plot_w, plot_h = 70 + panel * 430, 55, 360, 235
        vals = [f(row[raw_key]) for rows in series_data.values() for row in rows]
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.08, 0.02)
        lo -= pad
        hi += pad
        parts.append(svg_text(x0 + plot_w / 2, 28, title, 15, "middle"))
        parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#333"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#333"/>')
        for boundary in boundaries:
            x = x0 + plot_w * (boundary - 1) / 467
            parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + plot_h}" stroke="#aaaaaa" stroke-dasharray="5,4"/>')
        for method, rows in series_data.items():
            parts.append(svg_polyline([f(row[raw_key]) for row in rows], x0, y0, plot_w, plot_h, lo, hi, colors[method], 0.78, 1.35))
        parts.append(svg_text(x0 - 12, y0 + plot_h / 2, y_label, 11, "middle", "#555555"))
        parts.append(svg_text(x0 + plot_w / 2, y0 + plot_h + 33, "Rollout iteration", 11, "middle"))
    parts.extend([svg_text(340, 330, "PUREEM", 10, "start", "#234f7d"), svg_text(420, 330, "PSC", 10, "start", "#9a3f3f"), svg_text(490, 330, "dashed = recovery boundary", 10, "start", "#777777"), '</svg>'])
    (OUT / "recovery_boundary_audit.svg").write_text("\n".join(parts), encoding="utf-8")
    write_json(OUT / "plot_status.json", {"available": True, "backend": "dependency-free-svg", "files": ["paired_win_tie_loss.svg", "psc_signal_availability.svg", "recovery_boundary_audit.svg"]})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paired_summary()
    topology_complexity_summary()
    signal_availability_summary()
    recovery_summary()
    make_plots()
    write_json(
        OUT / "manifest.json",
        {
            "status": "complete",
            "kind": "frozen-evidence reaggregation",
            "source_archive": str(ARCHIVE),
            "source_curves": str(CURVES),
            "outputs": sorted(path.name for path in OUT.iterdir() if path.is_file()),
            "not_retrained": True,
        },
    )


if __name__ == "__main__":
    main()
