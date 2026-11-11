#!/usr/bin/env python3
"""Build the paper-ready PSC mechanism validation figure without plotting deps."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "exp" / "followup_experiments_20260916_r1"
OUT = ROOT / "drafts" / "dagent_canonical" / "figures"
W, H = 792, 260
BLUE = "#1f4a78"
RED = "#9b3735"
GRAY = "#8b8b8b"
LIGHT = "#d9dde3"
TEXT = "#202020"


def rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Figure:
    def __init__(self) -> None:
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        self.pdf: list[str] = []

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = TEXT, width: float = 0.8, dash: str | None = None) -> None:
        self.svg.append(
            f'<line x1="{x1:.2f}" y1="{H-y1:.2f}" x2="{x2:.2f}" y2="{H-y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>'
        )
        r, g, b = rgb(color)
        self.pdf.append(f"{r:.4f} {g:.4f} {b:.4f} RG {width:.2f} w" + (f" [{dash.replace(' ', ',')}] 0 d" if dash else " [] 0 d") + f" {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def rect(self, x: float, y: float, w: float, h: float, fill: str | None = None, stroke: str | None = None, width: float = 0.8) -> None:
        svg_fill = fill or "none"
        svg_stroke = stroke or "none"
        self.svg.append(f'<rect x="{x:.2f}" y="{H-y-h:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{svg_fill}" stroke="{svg_stroke}" stroke-width="{width:.2f}"/>')
        if fill:
            r, g, b = rgb(fill)
            self.pdf.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        if stroke:
            r, g, b = rgb(stroke)
            self.pdf.append(f"{r:.4f} {g:.4f} {b:.4f} RG {width:.2f} w")
        self.pdf.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re " + ("B" if fill and stroke else "f" if fill else "S"))

    def polyline(self, points: list[tuple[float, float]], color: str, width: float = 1.4, opacity: float = 1.0) -> None:
        if len(points) < 2:
            return
        svg_points = " ".join(f"{x:.2f},{H-y:.2f}" for x, y in points)
        self.svg.append(f'<polyline points="{svg_points}" fill="none" stroke="{color}" stroke-width="{width:.2f}" opacity="{opacity:.2f}"/>')
        base = rgb(color)
        r, g, b = tuple(channel * opacity + (1.0 - opacity) for channel in base)
        ops = [f"{r:.4f} {g:.4f} {b:.4f} RG {width:.2f} w"]
        ops.append(f"{points[0][0]:.2f} {points[0][1]:.2f} m")
        ops.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
        ops.append("S")
        self.pdf.append(" ".join(ops))

    def push_clip(self, x: float, y: float, w: float, h: float, clip_id: str) -> None:
        self.svg.append(f'<defs><clipPath id="{clip_id}"><rect x="{x:.2f}" y="{H-y-h:.2f}" width="{w:.2f}" height="{h:.2f}"/></clipPath></defs>')
        self.svg.append(f'<g clip-path="url(#{clip_id})">')
        self.pdf.append(f"q {x:.2f} {y:.2f} {w:.2f} {h:.2f} re W n")

    def pop_clip(self) -> None:
        self.svg.append("</g>")
        self.pdf.append("Q")

    def text(self, x: float, y: float, value: str, size: float = 8, color: str = TEXT, *, anchor: str = "start", bold: bool = False, rotate: int = 0) -> None:
        family = "Times New Roman"
        weight = "bold" if bold else "normal"
        svg_anchor = {"start": "start", "middle": "middle", "end": "end"}[anchor]
        transform = f' transform="rotate({-rotate} {x:.2f} {H-y:.2f})"' if rotate else ""
        self.svg.append(f'<text x="{x:.2f}" y="{H-y:.2f}" text-anchor="{svg_anchor}" font-family="{family}" font-size="{size:.2f}px" font-weight="{weight}" fill="{color}"{transform}>{value}</text>')
        r, g, b = rgb(color)
        font = "/F2" if bold else "/F1"
        if rotate == 90:
            matrix = f"0 1 -1 0 {x:.2f} {y:.2f} Tm"
        else:
            x_position = x
            if anchor == "middle":
                # Approximate centering for the standard Times font.
                x_position += -0.25 * size * len(value)
            elif anchor == "end":
                x_position += -0.5 * size * len(value)
            matrix = f"1 0 0 1 {x_position:.2f} {y:.2f} Tm"
        self.pdf.append(f"BT {font} {size:.2f} Tf {r:.4f} {g:.4f} {b:.4f} rg {matrix} ({esc(value)}) Tj ET")

    def finish(self) -> tuple[str, bytes]:
        self.svg.append("</svg>")
        stream = "\n".join(self.pdf).encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 260] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        ]
        pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, obj in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf.extend(f"{index} 0 obj\n".encode())
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")
        xref = len(pdf)
        pdf.extend(f"xref\n0 {len(objects)+1}\n".encode())
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode())
        pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        return "\n".join(self.svg), bytes(pdf)


def scale(value: float, lo: float, hi: float, a: float, b: float) -> float:
    return a + (value - lo) / (hi - lo) * (b - a)


def axes(fig: Figure, x: float, y: float, w: float, h: float, *, xmin: float, xmax: float, ymin: float, ymax: float, xticks: list[tuple[float, str]], yticks: list[tuple[float, str]], xlabel: str, ylabel: str, xtick_size: float = 6.8, ylabel_offset: float = -26) -> None:
    fig.line(x, y, x + w, y, TEXT, 0.8)
    fig.line(x, y, x, y + h, TEXT, 0.8)
    for value, label in xticks:
        xx = scale(value, xmin, xmax, x, x + w)
        fig.line(xx, y, xx, y - 3, TEXT, 0.7)
        fig.text(xx, y - 12, label, xtick_size, TEXT, anchor="middle")
    for value, label in yticks:
        yy = scale(value, ymin, ymax, y, y + h)
        fig.line(x, yy, x - 3, yy, TEXT, 0.7)
        fig.text(x - 7, yy - 2, label, 6.8, TEXT, anchor="end")
    fig.text(x + w / 2, y - 25, xlabel, 7.4, TEXT, anchor="middle")
    fig.text(x + ylabel_offset, y + h / 2, ylabel, 7.4, TEXT, anchor="middle", rotate=90)


def rolling(values: list[float], window: int = 11) -> list[float]:
    half = window // 2
    result = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    perm_summary = json.loads((DATA / "psc_permutation_null" / "permutation_null_summary.json").read_text(encoding="utf-8"))
    null_values = [float(row["planner_em_cosine"]) for row in csv.DictReader((DATA / "psc_permutation_null" / "permutation_null.csv").open(encoding="utf-8"))]
    objective_rows = list(csv.DictReader((DATA / "psc_objective_ablation" / "objective_ablation.csv").open(encoding="utf-8")))
    objective = {row["variant"]: row for row in objective_rows}
    signal_rows = list(csv.DictReader((DATA / "psc_signal_availability.csv").open(encoding="utf-8")))
    signal = [float(row["informative_fraction"]) for row in signal_rows]

    fig = Figure()
    panel_y, panel_w, panel_h = 49, 214, 164
    panel_xs = [42, 289, 536]
    for x in panel_xs:
        fig.rect(x, panel_y, panel_w, panel_h, None, LIGHT, 0.7)

    # (a) permutation null
    x, y, w, h = panel_xs[0] + 34, panel_y + 22, panel_w - 44, panel_h - 42
    xmin, xmax, ymin = -0.45, 1.02, 0
    max_count = max(null_values.count(v) for v in set(null_values))
    ymax = max(160, max_count * 1.12)
    axes(fig, x, y, w, h, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xticks=[(-0.4, "-0.4"), (0.0, "0"), (0.5, "0.5"), (1.0, "1.0")], yticks=[(0, "0"), (100, "100")], xlabel="Planner-EM cosine", ylabel="count")
    bins = 24
    step = (0.05 - xmin) / bins
    counts = [0] * bins
    for value in null_values:
        index = min(bins - 1, max(0, int((value - xmin) / step)))
        counts[index] += 1
    fig.push_clip(x, y, w, h, "permutation-plot-clip")
    for index, count in enumerate(counts):
        bx = scale(xmin + index * step, xmin, xmax, x, x + w)
        bw = step * w / (xmax - xmin) * 0.92
        bh = count / ymax * h
        fig.rect(bx, y, bw, bh, "#c9cfd7", None, 0)
    fig.pop_clip()
    # Redraw the plot boundaries after clipping so no histogram patch covers them.
    fig.line(x, y, x + w, y, TEXT, 0.8)
    fig.line(x, y, x, y + h, TEXT, 0.8)
    null_mean = float(perm_summary["null_mean_cosine"])
    real = float(perm_summary["real_planner_em_cosine"])
    xx = scale(null_mean, xmin, xmax, x, x + w)
    fig.line(xx, y, xx, y + h, "#b7b7b7", 0.65, "3 2")
    xr = scale(real, xmin, xmax, x, x + w)
    fig.line(xr, y, xr, y + h, RED, 2.1)
    fig.text(x + 5, y + h - 12, "real = 0.989", 7.5, RED, bold=True)
    fig.text(x + 5, y + h - 25, "p = 0.001996", 7.2, TEXT)
    fig.text(x + w - 3, y + h - 12, "null mean = -0.275", 6.6, GRAY, anchor="end")
    fig.text(panel_xs[0] + 8, panel_y + panel_h + 10, "(a) Outcome-label permutation", 8.6, TEXT, bold=True)

    # (b) direct objective-design gradient geometry
    x, y, w, h = panel_xs[1] + 34, panel_y + 22, panel_w - 44, panel_h - 42
    ymin, ymax = -0.25, 0.45
    axes(fig, x, y, w, h, xmin=-0.5, xmax=3.5, ymin=ymin, ymax=ymax, xticks=[(0, "Real"), (1, "Shuffled"), (2, "Cross-question"), (3, "No length norm.")], yticks=[(-0.2, "-0.2"), (0, "0"), (0.2, "0.2"), (0.4, "0.4")], xlabel="PSC design", ylabel="direct cosine", xtick_size=5.2)
    fig.line(x, scale(0, ymin, ymax, y, y + h), x + w, scale(0, ymin, ymax, y, y + h), LIGHT, 0.8)
    vals = [float(objective[name]["planner_em_cosine"]) for name in ["real", "shuffled", "cross_question", "no_length_normalization"]]
    colors = [RED, GRAY, BLUE, "#6b7e93"]
    zero = scale(0, ymin, ymax, y, y + h)
    for i, (value, color) in enumerate(zip(vals, colors)):
        center = scale(i, -0.5, 3.5, x, x + w)
        top = scale(max(value, 0), ymin, ymax, y, y + h)
        bottom = scale(min(value, 0), ymin, ymax, y, y + h)
        fig.rect(center - 13, bottom, 26, top - bottom, color, None, 0)
        fig.text(center, top + 9 if value >= 0 else bottom - 10, f"{value:+.3f}", 6.8, color, anchor="middle", bold=True)
    fig.text(x + w - 3, y + h - 12, "6.60x norm ratio", 6.7, RED, anchor="end", bold=True)
    fig.text(panel_xs[1] + 8, panel_y + panel_h + 10, "(b) Objective design", 8.6, TEXT, bold=True)

    # (c) signal availability
    x, y, w, h = panel_xs[2] + 34, panel_y + 22, panel_w - 44, panel_h - 42
    axes(fig, x, y, w, h, xmin=1, xmax=468, ymin=0, ymax=0.45, xticks=[(1, "1"), (117, "117"), (234, "234"), (351, "351"), (468, "468")], yticks=[(0, "0"), (0.2, "0.2"), (0.4, "0.4")], xlabel="rollout iteration", ylabel="Informative-group fraction", ylabel_offset=-48)
    raw = [(scale(i + 1, 1, 468, x, x + w), scale(value, 0, 0.45, y, y + h)) for i, value in enumerate(signal)]
    smooth_values = rolling(signal, 11)
    smooth = [(scale(i + 1, 1, 468, x, x + w), scale(value, 0, 0.45, y, y + h)) for i, value in enumerate(smooth_values)]
    fig.polyline(raw, RED, 0.55, 0.18)
    fig.polyline(smooth, RED, 1.6, 1.0)
    overall = 2906 / 14976
    yy = scale(overall, 0, 0.45, y, y + h)
    fig.line(x, yy, x + w, yy, "#a5a5a5", 0.7, "3 2")
    fig.text(x + 4, yy + 8, "mean = 19.4%", 6.6, GRAY)
    fig.text(x + w - 3, y + h - 12, "2,906 / 14,976 groups", 6.6, TEXT, anchor="end")
    fig.text(x + w - 3, y + h - 24, "signal in 468 / 468 steps", 6.6, TEXT, anchor="end")
    fig.text(panel_xs[2] + 8, panel_y + panel_h + 10, "(c) Signal availability", 8.6, TEXT, bold=True)

    svg, pdf = fig.finish()
    (OUT / "dagent_psc_mechanism_validation.svg").write_text(svg, encoding="utf-8")
    (OUT / "dagent_psc_mechanism_validation.pdf").write_bytes(pdf)
    print(OUT / "dagent_psc_mechanism_validation.svg")
    print(OUT / "dagent_psc_mechanism_validation.pdf")


if __name__ == "__main__":
    main()
