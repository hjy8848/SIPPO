#!/usr/bin/env python3
"""Produce paper-ready statistics for matched full-split DAGENT 3B evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "bamboogle")
ARMS = ("sft", "pureem", "psc")
ARM_LABELS = {"sft": "SFT", "pureem": "PureEM", "psc": "PSC"}
PAIRS = (("sft", "pureem"), ("sft", "psc"), ("pureem", "psc"))
METRICS = ("em", "f1")
PLAN_NODE_RE = re.compile(
    r"<(search|reason)\s+id=[\"'](\d+)[\"']>(.*?)</\1>", re.IGNORECASE | re.DOTALL
)
DEPENDENCY_RE = re.compile(r"#(\d+)#")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--pureem-root", type=Path, required=True)
    parser.add_argument("--psc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-chunk", type=int, default=16)
    parser.add_argument("--audit-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def normalize_question(value: Any) -> str:
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


def result_answers(row: Mapping[str, Any]) -> list[str]:
    for key in ("gold_answer", "golden_answers", "gold_answers"):
        values = answer_list(row.get(key))
        if values:
            return values
    return []


def question_gold_key(question: Any, answers: Any) -> tuple[str, str]:
    return normalize_question(question), canonical_answers(answers)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_result_file(root: Path, dataset: str) -> Path:
    trajectory_root = root / "trajectories" if (root / "trajectories").is_dir() else root
    matches = sorted(trajectory_root.glob(f"full_logs_*_{dataset}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {dataset} result under {trajectory_root}, found {matches}")
    return matches[0]


def load_manifest(manifest_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    summary_path = manifest_dir / "manifest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("total_selected_rows", -1)) != 22_517:
        raise RuntimeError(f"Unexpected manifest total in {summary_path}")
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        path = manifest_dir / f"{dataset}.parquet"
        table = pq.read_table(path, use_threads=False)
        rows = table.to_pylist()
        expected = int(summary["datasets"][dataset]["selected_rows"])
        if len(rows) != expected:
            raise RuntimeError(f"Manifest row mismatch for {dataset}: {len(rows)}/{expected}")
        if file_sha256(path) != summary["datasets"][dataset]["parquet_sha256"]:
            raise RuntimeError(f"Manifest SHA mismatch for {path}")
        keys = [question_gold_key(row["question"], row["golden_answers"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"Duplicate question+gold keys in {path}")
        by_dataset[dataset] = rows
    return by_dataset, summary


def load_arm(
    root: Path,
    manifest: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    result_shas: dict[str, str] = {}
    for dataset in DATASETS:
        path = locate_result_file(root, dataset)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != len(manifest[dataset]):
            raise RuntimeError(
                f"Unexpected result rows for {dataset} in {path}: {len(payload)}/{len(manifest[dataset])}"
            )
        manifest_by_question = {
            normalize_question(row["question"]): question_gold_key(
                row["question"], row["golden_answers"]
            )
            for row in manifest[dataset]
        }
        expected_keys = set(manifest_by_question.values())
        observed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in payload:
            question = normalize_question(row.get("question"))
            gold = result_answers(row)
            if not gold:
                raise RuntimeError(f"Missing result gold answer in {path}: {row.get('question')}")
            key = question_gold_key(row.get("question"), gold)
            if question not in manifest_by_question or key != manifest_by_question[question]:
                raise RuntimeError(f"Result question+gold is outside manifest in {path}: {key}")
            if key in observed:
                raise RuntimeError(f"Duplicate result question+gold in {path}: {key}")
            for metric in METRICS:
                value = float(row.get(metric, math.nan))
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {metric} in {path}: {key[0]}")
            if row.get("error") not in (None, "", False) or row.get("pipeline_error") not in (
                None,
                "",
                False,
            ):
                raise RuntimeError(f"Pipeline error in {path}: {key[0]}")
            observed[key] = row
        if set(observed) != expected_keys:
            raise RuntimeError(f"Result multiset mismatch for {dataset} in {path}")
        by_dataset[dataset] = [
            observed[question_gold_key(row["question"], row["golden_answers"])]
            for row in manifest[dataset]
        ]
        result_shas[dataset] = file_sha256(path)
    return by_dataset, result_shas


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = min(left_only, right_only)
    logs = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)
        for k in range(tail + 1)
    ]
    maximum = max(logs)
    probability = math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in logs)
    return min(1.0, 2.0 * probability)


def bootstrap_means(
    values: np.ndarray,
    *,
    samples: int,
    chunk: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty((samples, values.shape[1]), dtype=np.float64)
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = rng.integers(0, values.shape[0], size=(stop - start, values.shape[0]))
        output[start:stop] = values[indices].mean(axis=1)
    return output


def quantile_ci(values: np.ndarray) -> list[float]:
    return [float(item) for item in np.quantile(values, (0.025, 0.975))]


def build_difference_matrix(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for left, right in PAIRS:
        for metric in METRICS:
            columns.append(
                np.asarray(
                    [float(row[metric]) for row in arms[right]], dtype=np.float64
                )
                - np.asarray([float(row[metric]) for row in arms[left]], dtype=np.float64)
            )
            labels.append(f"{right}_minus_{left}.{metric}")
    return np.column_stack(columns), labels


def paired_report(
    arms: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    samples: int,
    chunk: int,
    seed: int,
) -> dict[str, Any]:
    matrices: dict[str, np.ndarray] = {}
    labels: list[str] | None = None
    stratified_bootstrap: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for dataset in DATASETS:
        matrix, current_labels = build_difference_matrix(
            {arm: arms[arm][dataset] for arm in ARMS}
        )
        matrices[dataset] = matrix
        labels = labels or current_labels
        if current_labels != labels:
            raise RuntimeError("Internal bootstrap column mismatch")
        stratified_bootstrap[dataset] = bootstrap_means(
            matrix, samples=samples, chunk=chunk, rng=rng
        )
    assert labels is not None
    macro_bootstrap = sum(stratified_bootstrap.values()) / len(DATASETS)
    micro_matrix = np.concatenate([matrices[name] for name in DATASETS], axis=0)
    micro_bootstrap = bootstrap_means(
        micro_matrix,
        samples=samples,
        chunk=chunk,
        rng=np.random.default_rng(seed + 1),
    )

    output: dict[str, Any] = {}
    for pair_index, (left, right) in enumerate(PAIRS):
        pair_name = f"{right}_minus_{left}"
        pair: dict[str, Any] = {"left": left, "right": right, "per_dataset": {}}
        for metric_index, metric in enumerate(METRICS):
            column = pair_index * len(METRICS) + metric_index
            for dataset in DATASETS:
                pair["per_dataset"].setdefault(dataset, {})[metric] = {
                    "delta": float(matrices[dataset][:, column].mean()),
                    "paired_bootstrap_95ci": quantile_ci(
                        stratified_bootstrap[dataset][:, column]
                    ),
                }
            pair.setdefault("macro", {})[metric] = {
                "delta": float(
                    statistics.fmean(matrices[name][:, column].mean() for name in DATASETS)
                ),
                "stratified_paired_bootstrap_95ci": quantile_ci(macro_bootstrap[:, column]),
            }
            pair.setdefault("micro", {})[metric] = {
                "delta": float(micro_matrix[:, column].mean()),
                "paired_bootstrap_95ci": quantile_ci(micro_bootstrap[:, column]),
            }

        quadrants = Counter()
        for dataset in DATASETS:
            for left_row, right_row in zip(arms[left][dataset], arms[right][dataset]):
                left_correct = float(left_row["em"]) == 1.0
                right_correct = float(right_row["em"]) == 1.0
                quadrant = (
                    "both_correct"
                    if left_correct and right_correct
                    else "left_only"
                    if left_correct
                    else "right_only"
                    if right_correct
                    else "both_wrong"
                )
                quadrants[quadrant] += 1
        pair["em_quadrants"] = dict(quadrants)
        pair["mcnemar"] = {
            "left_only": quadrants["left_only"],
            "right_only": quadrants["right_only"],
            "exact_two_sided_p": exact_mcnemar_pvalue(
                quadrants["left_only"], quadrants["right_only"]
            ),
        }
        output[pair_name] = pair
    return output


def arm_metrics(arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"per_dataset": {}}
    all_rows: list[Mapping[str, Any]] = []
    for dataset in DATASETS:
        rows = list(arm[dataset])
        all_rows.extend(rows)
        report["per_dataset"][dataset] = {
            "samples": len(rows),
            **{metric: statistics.fmean(float(row[metric]) for row in rows) for metric in METRICS},
        }
    report["macro"] = {
        metric: statistics.fmean(
            report["per_dataset"][dataset][metric] for dataset in DATASETS
        )
        for metric in METRICS
    }
    report["micro"] = {
        "samples": len(all_rows),
        **{metric: statistics.fmean(float(row[metric]) for row in all_rows) for metric in METRICS},
    }
    return report


def token_counter(path: Path | None) -> tuple[Callable[[str], int] | None, str]:
    if path is None:
        return None, "not_requested"
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    return (
        lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
        str(path.resolve()),
    )


def parse_plan_nodes(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = str(trace.get("plan_output") or trace.get("raw_plan_output") or "")
    return [
        {
            "type": kind.casefold(),
            "id": int(node_id),
            "dependencies": [int(value) for value in DEPENDENCY_RE.findall(body)],
        }
        for kind, node_id, body in PLAN_NODE_RE.findall(text)
    ]


def graph_depth(nodes: Sequence[Mapping[str, Any]]) -> int:
    depths: dict[int, int] = {}
    for node in sorted(nodes, key=lambda item: int(item["id"])):
        parents = [depths.get(int(dep), 0) for dep in node["dependencies"]]
        depths[int(node["id"])] = 1 + (max(parents) if parents else 0)
    return max(depths.values(), default=0)


def row_compute(
    row: Mapping[str, Any], count_tokens: Callable[[str], int] | None
) -> dict[str, Any]:
    trace = row.get("raw_trace") or {}
    nodes = trace.get("nodes_execution") or []
    planned = parse_plan_nodes(trace)
    events = trace.get("sft_events") or []
    generated_tokens = 0
    planner_tokens = 0
    for event in events:
        response = str((event or {}).get("response") or "")
        current = count_tokens(response) if count_tokens is not None else 0
        generated_tokens += current
        if (event or {}).get("agent_name") == "PlanningAgent":
            planner_tokens += current
    execution_error = str(trace.get("execution_error") or "").strip()
    return {
        "model_calls": len(events),
        "generated_tokens": generated_tokens,
        "planner_generated_tokens": planner_tokens,
        "retrieval_calls": int(row.get("retrieval_calls") or 0),
        "executed_nodes": len(nodes),
        "executed_search_nodes": sum(
            str((node or {}).get("type", "")).casefold() == "search" for node in nodes
        ),
        "executed_reason_nodes": sum(
            str((node or {}).get("type", "")).casefold() == "reason" for node in nodes
        ),
        "planned_nodes": len(planned),
        "graph_depth": graph_depth(planned),
        "query_rewrite_calls": sum(
            (event or {}).get("agent_name") == "QueryRewriteAgent" for event in events
        ),
        "execution_error": int(bool(execution_error)),
        "execution_error_text": execution_error,
    }


def summarize_numbers(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "sum": math.fsum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def compute_report(
    arm: Mapping[str, Sequence[Mapping[str, Any]]],
    count_tokens: Callable[[str], int] | None,
) -> dict[str, Any]:
    fields = (
        "model_calls",
        "generated_tokens",
        "planner_generated_tokens",
        "retrieval_calls",
        "executed_nodes",
        "executed_search_nodes",
        "executed_reason_nodes",
        "planned_nodes",
        "graph_depth",
        "query_rewrite_calls",
        "execution_error",
    )
    per_dataset: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    for dataset in DATASETS:
        computed = [row_compute(row, count_tokens) for row in arm[dataset]]
        all_rows.extend(computed)
        errors.update(row["execution_error_text"] for row in computed if row["execution_error_text"])
        per_dataset[dataset] = {
            field: summarize_numbers(computed, field) for field in fields
        }
        per_dataset[dataset]["reason_use_rate"] = statistics.fmean(
            row["executed_reason_nodes"] > 0 for row in computed
        )
    return {
        "per_dataset": per_dataset,
        "overall": {field: summarize_numbers(all_rows, field) for field in fields},
        "reason_use_rate": statistics.fmean(row["executed_reason_nodes"] > 0 for row in all_rows),
        "execution_error_reasons": dict(errors.most_common()),
        "pipeline_errors": 0,
    }


def prediction(row: Mapping[str, Any]) -> str:
    for key in ("predicted_answer", "prediction", "answer", "final_answer"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    trace = row.get("raw_trace") or {}
    for key in ("final_answer", "answer"):
        if trace.get(key) not in (None, ""):
            return str(trace[key])
    return ""


def audit_manifest(
    manifest: Mapping[str, Sequence[Mapping[str, Any]]],
    arms: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    strata: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for dataset in DATASETS:
        for index, manifest_row in enumerate(manifest[dataset]):
            pure_correct = float(arms["pureem"][dataset][index]["em"]) == 1.0
            psc_correct = float(arms["psc"][dataset][index]["em"]) == 1.0
            quadrant = (
                "both_correct"
                if pure_correct and psc_correct
                else "pureem_only"
                if pure_correct
                else "psc_only"
                if psc_correct
                else "both_wrong"
            )
            record: dict[str, Any] = {
                "dataset": dataset,
                "source_index": manifest_row["sample_index"],
                "question": manifest_row["question"],
                "golden_answers": manifest_row["golden_answers"],
                "pureem_psc_quadrant": quadrant,
                "manual_primary_error_category": "",
                "manual_secondary_error_category": "",
                "manual_notes": "",
            }
            for arm in ARMS:
                row = arms[arm][dataset][index]
                record[f"{arm}_answer"] = prediction(row)
                record[f"{arm}_em"] = float(row["em"])
                record[f"{arm}_f1"] = float(row["f1"])
            strata[(dataset, quadrant)].append(record)
    nonempty = sorted(strata)
    if samples < len(nonempty):
        raise RuntimeError(f"Audit size {samples} is smaller than {len(nonempty)} nonempty strata")
    allocation = {key: 1 for key in nonempty}
    remaining = samples - len(nonempty)
    total = sum(len(strata[key]) for key in nonempty)
    quotas = {key: remaining * len(strata[key]) / total for key in nonempty}
    for key in nonempty:
        allocation[key] += int(math.floor(quotas[key]))
    leftover = samples - sum(allocation.values())
    for key in sorted(nonempty, key=lambda item: (quotas[item] % 1, item), reverse=True)[:leftover]:
        allocation[key] += 1
    selected: list[dict[str, Any]] = []
    for key in nonempty:
        ordered = sorted(
            strata[key],
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row['dataset']}\0{row['source_index']}\0{row['question']}".encode()
            ).digest(),
        )
        selected.extend(ordered[: allocation[key]])
    return sorted(selected, key=lambda row: (row["dataset"], row["source_index"]))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_latex(report: Mapping[str, Any]) -> str:
    lines = [
        "% Auto-generated by scripts/analyze_dagent_full_3b_eval.py",
        "% Main outcome table rows: dataset & SFT EM/F1 & PureEM EM/F1 & PSC EM/F1",
    ]
    for dataset in DATASETS:
        cells = [dataset]
        for arm in ARMS:
            row = report["metrics"][arm]["per_dataset"][dataset]
            cells.append(f"{100 * row['em']:.2f} / {100 * row['f1']:.2f}")
        lines.append(" & ".join(cells) + r" \\")
    for scope in ("macro", "micro"):
        cells = [scope.capitalize()]
        for arm in ARMS:
            row = report["metrics"][arm][scope]
            cells.append(f"{100 * row['em']:.2f} / {100 * row['f1']:.2f}")
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis evidence: {args.output_dir}")
    if args.bootstrap_samples <= 0 or args.bootstrap_chunk <= 0:
        raise ValueError("Bootstrap sample and chunk counts must be positive")
    manifest, manifest_summary = load_manifest(args.manifest_dir)
    roots = {
        "sft": args.sft_root,
        "pureem": args.pureem_root,
        "psc": args.psc_root,
    }
    arms: dict[str, dict[str, list[dict[str, Any]]]] = {}
    result_shas: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        arms[arm], result_shas[arm] = load_arm(roots[arm], manifest)
    count_tokens, tokenizer_source = token_counter(args.tokenizer_json)
    report = {
        "schema_version": 1,
        "contract": {
            "manifest_summary_sha256": file_sha256(
                args.manifest_dir / "manifest_summary.json"
            ),
            "dataset_sizes": {
                name: len(manifest[name]) for name in DATASETS
            },
            "total_samples": sum(len(manifest[name]) for name in DATASETS),
            "alignment": "exact normalized question + ordered JSON-canonical gold list",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "macro_ci": "paired stratified bootstrap within each dataset, then equal dataset mean",
            "micro_ci": "ordinary paired bootstrap over all questions",
            "tokenizer": tokenizer_source,
            "result_sha256": result_shas,
            "source_manifest": manifest_summary,
        },
        "metrics": {arm: arm_metrics(arms[arm]) for arm in ARMS},
        "paired": paired_report(
            arms,
            samples=args.bootstrap_samples,
            chunk=args.bootstrap_chunk,
            seed=args.seed,
        ),
        "compute": {arm: compute_report(arms[arm], count_tokens) for arm in ARMS},
    }
    audit_rows = audit_manifest(
        manifest, arms, samples=args.audit_samples, seed=args.seed + 2
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "full_3b_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "paper_table_rows.tex").write_text(
        render_latex(report), encoding="utf-8"
    )
    (args.output_dir / "error_audit_100.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "error_audit_100.csv", audit_rows)

    metric_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for scope in (*DATASETS, "macro", "micro"):
            row = (
                report["metrics"][arm]["per_dataset"][scope]
                if scope in DATASETS
                else report["metrics"][arm][scope]
            )
            metric_rows.append(
                {"arm": ARM_LABELS[arm], "scope": scope, "em": row["em"], "f1": row["f1"]}
            )
    write_csv(args.output_dir / "outcome_metrics.csv", metric_rows)
    replacement = {
        "main_result_table": metric_rows,
        "paired": report["paired"],
        "compute": report["compute"],
        "reason_use": {
            arm: {
                "overall": report["compute"][arm]["reason_use_rate"],
                "per_dataset": {
                    dataset: report["compute"][arm]["per_dataset"][dataset]["reason_use_rate"]
                    for dataset in DATASETS
                },
            }
            for arm in ARMS
        },
        "error_audit_status": "new 100-question manifest generated; manual labels pending",
    }
    (args.output_dir / "paper_replacement.json").write_text(
        json.dumps(replacement, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(replacement, ensure_ascii=False, indent=2))
    print(f"[DONE] Full 3B analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
