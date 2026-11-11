#!/usr/bin/env python3
"""Fail-closed paper statistics for complete answerable evaluation arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence



DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "bamboogle")
METRICS = ("em", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dagent-arm",
        action="append",
        default=[],
        metavar="LABEL=RUN_ROOT",
        help="A completed DAGENT arm. Repeat for Base, SFT, PureEM, and PSC.",
    )
    parser.add_argument(
        "--search-r1",
        type=Path,
        help="Optional scored Search-R1 root using the same per-dataset JSON files.",
    )
    parser.add_argument(
        "--external",
        action="append",
        default=[],
        metavar="LABEL=RUN_ROOT",
        help="A completed external-baseline root produced by the batch adapter. Repeat as needed.",
    )
    return parser.parse_args()


def parse_arm_specs(specs: Sequence[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for spec in specs:
        label, separator, raw_root = spec.partition("=")
        if not separator or not label.strip() or not raw_root.strip():
            raise ValueError(f"Invalid --dagent-arm {spec!r}; expected LABEL=RUN_ROOT")
        if label in arms:
            raise ValueError(f"Duplicate arm label: {label}")
        arms[label] = Path(raw_root)
    return arms


def normalize_question(value: Any) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return re.sub(r"\s+", " ", text)


def answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def canonical_answers(value: Any) -> str:
    return json.dumps(answer_list(value), ensure_ascii=False, separators=(",", ":"))


def question_gold_key(question: Any, answers: Any) -> tuple[str, str]:
    return normalize_question(question), canonical_answers(answers)


def load_search_r1_arm(
    root: Path,
    manifest: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Load native Search-R1 rows after its separate runtime has scored them."""
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    result_shas: dict[str, str] = {}
    for dataset in DATASETS:
        path = root / "trajectories" / f"search_r1_{dataset}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing Search-R1 result file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != len(manifest[dataset]):
            raise RuntimeError(f"Unexpected Search-R1 rows for {dataset}: {path}")
        expected = {
            question_gold_key(row["question"], row["golden_answers"])
            for row in manifest[dataset]
        }
        observed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in payload:
            key = question_gold_key(row.get("question"), row.get("golden_answers"))
            if key not in expected or key in observed:
                raise RuntimeError(f"Search-R1 question/gold mismatch in {path}: {key}")
            if row.get("error") not in (None, "", False):
                raise RuntimeError(f"Search-R1 execution error in {path}: {key[0]}")
            for metric in METRICS:
                value = float(row.get(metric, math.nan))
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite Search-R1 {metric} in {path}: {key[0]}")
            observed[key] = row
        if set(observed) != expected:
            raise RuntimeError(f"Search-R1 question multiset mismatch for {dataset}")
        by_dataset[dataset] = [
            observed[question_gold_key(row["question"], row["golden_answers"])]
            for row in manifest[dataset]
        ]
        result_shas[dataset] = sha256(path)
    return by_dataset, result_shas


def load_external_arm(
    root: Path,
    manifest: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_summary_sha256: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Load rows emitted by evaluate_external_baseline_full_answerable.py."""
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing external run manifest: {manifest_path}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = str(run_manifest.get("baseline") or "").strip()
    if not baseline:
        raise RuntimeError(f"External run manifest has no baseline: {manifest_path}")
    if run_manifest.get("manifest_summary_sha256") != manifest_summary_sha256:
        raise RuntimeError(f"External run manifest has an invalid manifest hash: {manifest_path}")
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    result_shas: dict[str, str] = {}
    for dataset in DATASETS:
        trajectory_root = root / "trajectories"
        candidates = [
            trajectory_root / f"{baseline}_{dataset}.json",
            trajectory_root / f"external_{baseline}_{dataset}.json",
            trajectory_root / f"search_r1_{dataset}.json",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise RuntimeError(f"Missing external result file under {trajectory_root}: {candidates}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != len(manifest[dataset]):
            raise RuntimeError(f"Unexpected external rows for {dataset}: {path}")
        expected = {
            question_gold_key(row["question"], row["golden_answers"])
            for row in manifest[dataset]
        }
        observed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in payload:
            key = question_gold_key(row.get("question"), row.get("golden_answers"))
            if key not in expected or key in observed:
                raise RuntimeError(f"External question/gold mismatch in {path}: {key}")
            if row.get("error") not in (None, "", False) or row.get("pipeline_error") not in (None, "", False):
                raise RuntimeError(f"External execution error in {path}: {key[0]}")
            for metric in METRICS:
                value = float(row.get(metric, math.nan))
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite external {metric} in {path}: {key[0]}")
            observed[key] = row
        if set(observed) != expected:
            raise RuntimeError(f"External question multiset mismatch for {dataset}")
        by_dataset[dataset] = [
            observed[question_gold_key(row["question"], row["golden_answers"])]
            for row in manifest[dataset]
        ]
        result_shas[dataset] = sha256(path)
    return by_dataset, result_shas


def metric_summary(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    per_dataset: dict[str, dict[str, float]] = {}
    all_rows: list[Mapping[str, Any]] = []
    for dataset in DATASETS:
        dataset_rows = list(rows[dataset])
        if not dataset_rows:
            raise RuntimeError(f"Empty dataset arm: {dataset}")
        per_dataset[dataset] = {
            metric: sum(float(row[metric]) for row in dataset_rows) / len(dataset_rows)
            for metric in ("em", "f1")
        }
        all_rows.extend(dataset_rows)
    macro = {
        metric: sum(per_dataset[dataset][metric] for dataset in DATASETS) / len(DATASETS)
        for metric in ("em", "f1")
    }
    micro = {
        metric: sum(float(row[metric]) for row in all_rows) / len(all_rows)
        for metric in ("em", "f1")
    }
    execution_errors = sum(
        bool((row.get("raw_trace") or {}).get("execution_error"))
        or bool(row.get("pipeline_error"))
        or row.get("error") not in (None, "", False)
        for row in all_rows
    )
    rewrite_calls = sum(
        sum(
            (event or {}).get("agent_name") == "QueryRewriteAgent"
            for event in ((row.get("raw_trace") or {}).get("sft_events") or [])
        )
        for row in all_rows
    )
    search_nodes = sum(
        sum(node.get("type") == "search" for node in ((row.get("raw_trace") or {}).get("nodes_execution") or []))
        for row in all_rows
    )
    return {
        "samples": len(all_rows),
        "per_dataset": per_dataset,
        "macro": macro,
        "micro": micro,
        "execution_error_rows": execution_errors,
        "query_rewrite_events": rewrite_calls,
        "search_nodes": search_nodes,
    }


def latex_number(value: float) -> str:
    return f"{value:.3f}"


def latex_label(label: str) -> str:
    return re.sub(r"([&_#%{}])", r"\\\1", label)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_latex(path: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & 2Wiki & HotpotQA & MuSiQue & Bamboogle & Macro EM/F1 & Micro EM/F1 \\\\",
        "\\midrule",
    ]
    for label, summary in summaries.items():
        values = [
            f'{summary["per_dataset"][dataset]["em"]:.4f}/{summary["per_dataset"][dataset]["f1"]:.4f}'
            for dataset in DATASETS
        ]
        values.extend(
            [
                f'{summary["macro"]["em"]:.4f}/{summary["macro"]["f1"]:.4f}',
                f'{summary["micro"]["em"]:.4f}/{summary["micro"]["f1"]:.4f}',
            ]
        )
        lines.append(
            latex_label(label)
            + " & "
            + " & ".join(
                latex_number(value) if isinstance(value, float) else value
                for value in values
            )
            + " \\\\" 
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    # Keep --help and contract inspection usable on lightweight workstations;
    # the remote evaluation environment supplies the parquet/numpy stack.
    from analyze_dagent_full_3b_eval import load_arm, load_manifest

    arms = parse_arm_specs(args.dagent_arm)
    external = parse_arm_specs(args.external)
    if not arms and args.search_r1 is None and not external:
        raise SystemExit("Provide at least one --dagent-arm, --search-r1, or --external")
    manifest, manifest_summary = load_manifest(args.manifest_dir)
    summaries: dict[str, Any] = {}
    result_roots: dict[str, str] = {}
    loaded_arms: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label, root in arms.items():
        loaded, result_shas = load_arm(root, manifest)
        summary = metric_summary(loaded)
        if summary["samples"] != int(manifest_summary["total_selected_rows"]):
            raise RuntimeError(f"{label} has {summary['samples']} rows, expected full manifest")
        summary["result_sha256"] = result_shas
        summaries[label] = summary
        result_roots[label] = str(root)
        loaded_arms[label] = loaded

    # Search-R1 is accepted only through its native-runtime adapter; it is not
    # silently coerced into DAGENT traces.
    if args.search_r1 is not None:
        loaded, result_shas = load_search_r1_arm(args.search_r1, manifest)
        summary = metric_summary(loaded)
        if summary["samples"] != int(manifest_summary["total_selected_rows"]):
            raise RuntimeError("Search-R1 does not cover the complete manifest")
        summary["result_sha256"] = result_shas
        summaries["Qwen3-4B Search-R1"] = summary
        result_roots["Qwen3-4B Search-R1"] = str(args.search_r1)
        loaded_arms["Qwen3-4B Search-R1"] = loaded

    for label, root in external.items():
        if label in loaded_arms:
            raise RuntimeError(f"Duplicate evaluation label: {label}")
        loaded, result_shas = load_external_arm(
            root,
            manifest,
            sha256(args.manifest_dir / "manifest_summary.json"),
        )
        summary = metric_summary(loaded)
        if summary["samples"] != int(manifest_summary["total_selected_rows"]):
            raise RuntimeError(f"{label} does not cover the complete manifest")
        summary["result_sha256"] = result_shas
        summaries[label] = summary
        result_roots[label] = str(root)
        loaded_arms[label] = loaded

    output = {
        "schema_version": 1,
        "manifest_summary_sha256": sha256(args.manifest_dir / "manifest_summary.json"),
        "dataset_counts": {
            dataset: len(manifest[dataset]) for dataset in DATASETS
        },
        "arms": summaries,
        "roots": result_roots,
        "pairwise": {},
    }
    labels = list(loaded_arms)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            deltas: dict[str, dict[str, float]] = {}
            for dataset in DATASETS:
                left_rows = loaded_arms[left][dataset]
                right_rows = loaded_arms[right][dataset]
                deltas[dataset] = {
                    metric: sum(float(r[metric]) - float(l[metric]) for l, r in zip(left_rows, right_rows))
                    / len(left_rows)
                    for metric in ("em", "f1")
                }
            output["pairwise"][f"{right}_minus_{left}"] = deltas

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "full_suite_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_latex(args.output_dir / "paper_table_rows.tex", summaries)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
