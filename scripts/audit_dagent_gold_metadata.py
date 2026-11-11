#!/usr/bin/env python3
"""Audit gold evidence and topology metadata available to DAGENT rewards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import pandas as pd


def to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return to_python(value.as_py())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        return to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    return value


def as_list(value: Any) -> list[str]:
    value = to_python(value)
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip().strip('"') for item in value if str(item).strip()]


def question(row: pd.Series) -> str:
    info = to_python(row.get("extra_info", {}))
    reward = to_python(row.get("reward_model", {}))
    if isinstance(info, dict) and info.get("question"):
        return str(info["question"])
    if isinstance(reward, dict) and reward.get("question"):
        return str(reward["question"])
    prompt = to_python(row.get("prompt", []))
    if isinstance(prompt, list) and prompt:
        return str(prompt[-1].get("content", ""))
    return ""


def find_titles(info: dict[str, Any], ground_truth: dict[str, Any]) -> tuple[list[str], str]:
    candidates = (
        ("extra_info.golden_titles", info.get("golden_titles")),
        ("extra_info.gold_titles", info.get("gold_titles")),
        ("ground_truth.golden_titles", ground_truth.get("golden_titles")),
        ("ground_truth.gold_titles", ground_truth.get("gold_titles")),
        ("ground_truth.supporting_titles", ground_truth.get("supporting_titles")),
    )
    for source, value in candidates:
        titles = as_list(value)
        if titles:
            return titles, source
    supporting = to_python(ground_truth.get("supporting_facts", []))
    if isinstance(supporting, list):
        titles = []
        for item in supporting:
            if isinstance(item, dict):
                title = item.get("title")
            elif isinstance(item, (list, tuple)) and item:
                title = item[0]
            else:
                title = None
            if title and str(title) not in titles:
                titles.append(str(title))
        if titles:
            return titles, "ground_truth.supporting_facts"
    return [], "missing"


def inspect_file(path: Path, export_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    by_source: dict[str, Counter] = defaultdict(Counter)
    title_sources = Counter()
    type_counts: dict[str, Counter] = defaultdict(Counter)
    key_counts: dict[str, Counter] = defaultdict(Counter)
    title_lengths: dict[str, list[int]] = defaultdict(list)

    for row_index, row in frame.iterrows():
        source = str(row.get("data_source", "unknown"))
        info = to_python(row.get("extra_info", {}))
        reward = to_python(row.get("reward_model", {}))
        info = info if isinstance(info, dict) else {}
        reward = reward if isinstance(reward, dict) else {}
        ground_truth = to_python(reward.get("ground_truth", {}))
        ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
        titles, title_source = find_titles(info, ground_truth)
        by_source[source]["rows"] += 1
        by_source[source]["with_gold_titles"] += bool(titles)
        by_source[source]["with_question_type"] += bool(ground_truth.get("type"))
        by_source[source]["with_decomposition"] += any(
            key in ground_truth or key in info
            for key in ("decomposition", "question_decomposition", "subquestions", "supporting_facts")
        )
        title_sources[title_source] += 1
        title_lengths[source].append(len(titles))
        type_counts[source][str(ground_truth.get("type", "missing"))] += 1
        for key in info:
            key_counts[source][f"extra_info.{key}"] += 1
        for key in ground_truth:
            key_counts[source][f"ground_truth.{key}"] += 1
        export_rows.append(
            {
                "file": path.name,
                "row_index": int(row_index),
                "data_source": source,
                "question": question(row),
                "gold_answers": as_list(ground_truth.get("target")),
                "gold_titles": titles,
                "gold_title_source": title_source,
                "question_type": str(ground_truth.get("type", "")),
            }
        )

    summary = {}
    for source, counts in by_source.items():
        lengths = title_lengths[source]
        summary[source] = {
            **dict(counts),
            "gold_title_coverage": counts["with_gold_titles"] / max(1, counts["rows"]),
            "mean_gold_titles": sum(lengths) / max(1, len(lengths)),
            "question_types": dict(type_counts[source]),
            "metadata_keys": dict(key_counts[source].most_common()),
        }
    return {
        "path": str(path),
        "columns": list(frame.columns),
        "rows": len(frame),
        "by_source": summary,
        "gold_title_sources": dict(title_sources),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    report = {"files": [inspect_file(path, exported) for path in args.input]}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "gold_metadata.json").write_text(
        json.dumps(exported, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
