#!/usr/bin/env python3
"""Merge and independently audit Wiki18-grounded Reason-to-Search batches."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "scripts/generate_dagent_reason_search_dataset.py"
SPEC = importlib.util.spec_from_file_location("reason_search_generator", GENERATOR_PATH)
GEN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEN
SPEC.loader.exec_module(GEN)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_is_verbatim(fact: dict[str, Any], documents: list[dict[str, Any]]) -> bool:
    try:
        document = documents[int(fact.get("doc_index", 0)) - 1]
    except (IndexError, TypeError, ValueError):
        return False
    quote = str(fact.get("evidence_quote", "")).strip()
    answer = str(fact.get("answer", "")).strip()
    contents = str(document.get("contents", ""))
    return bool(
        len(quote) >= 8
        and GEN.normalize(quote) in GEN.normalize(contents)
        and GEN.answer_in_text(answer, quote)
    )


def audit_record(record: dict[str, Any]) -> str | None:
    if record.get("topology") != GEN.TOPOLOGY:
        return "wrong topology"
    nodes = record.get("nodes_execution") or []
    if [node.get("type") for node in nodes] != ["search", "search", "reason", "search"]:
        return "wrong node sequence"
    if [str(node.get("id")) for node in nodes] != ["1", "2", "3", "4"]:
        return "wrong node IDs"
    if not GEN.REFERENCE_RE.search(str(nodes[3].get("task", ""))):
        return "search 4 does not depend on reason"
    if "#" in str(nodes[3].get("resolved_task", "")):
        return "search 4 has an unresolved reference"
    if not GEN.same_answer(nodes[2].get("conclusion"), record.get("winner")):
        return "reason conclusion disagrees with winner"
    if not GEN.same_answer(nodes[3].get("conclusion"), record.get("answer")):
        return "search 4 conclusion disagrees with final answer"
    if GEN.answer_in_text(record.get("answer"), record.get("question")):
        return "question leaks final answer"
    if re.search(
        r"\b(?:who|which)\b[^?]{0,160}\band\s+(?:where|when|what|how)\b",
        str(record.get("question", "")),
        flags=re.IGNORECASE,
    ):
        return "question requests winner and downstream answer"

    comparison = record.get("comparison_evidence") or {}
    fact_a = comparison.get("candidate_a") or {}
    fact_b = comparison.get("candidate_b") or {}
    if not evidence_is_verbatim(fact_a, nodes[0].get("documents") or []):
        return "candidate A evidence is not verbatim"
    if not evidence_is_verbatim(fact_b, nodes[1].get("documents") or []):
        return "candidate B evidence is not verbatim"
    blueprint = record.get("blueprint") or {}
    try:
        winner = GEN.deterministic_winner(
            blueprint.get("operation"),
            GEN.derived_comparison_key(fact_a.get("answer", ""), blueprint.get("operation")),
            GEN.derived_comparison_key(fact_b.get("answer", ""), blueprint.get("operation")),
            blueprint.get("candidate_a"),
            blueprint.get("candidate_b"),
        )
    except Exception as exc:
        return f"comparison cannot be recomputed: {exc}"
    if not GEN.same_answer(winner, record.get("winner")):
        return "recomputed comparison winner disagrees"
    if not GEN.same_answer(winner, (record.get("source") or {}).get("source_answer")):
        return "winner disagrees with source answer"

    downstream = record.get("downstream_evidence") or {}
    if not evidence_is_verbatim(downstream.get("winner") or {}, nodes[3].get("documents") or []):
        return "winner downstream evidence is not verbatim"
    counterfactual = record.get("counterfactual") or {}
    if not evidence_is_verbatim(
        downstream.get("counterfactual") or {}, counterfactual.get("documents") or []
    ):
        return "counterfactual downstream evidence is not verbatim"
    if GEN.same_answer(record.get("answer"), record.get("counterfactual_answer")):
        return "counterfactual answer is unchanged"

    direct = record.get("direct_retrieval_audit") or {}
    direct_context = "\n".join(str(doc.get("contents", "")) for doc in direct.get("documents", []))
    if direct.get("answer_found") is not False or GEN.answer_in_text(record.get("answer"), direct_context):
        return "direct retrieval exposes final answer"
    required_checks = (
        "comparison_reproduces_seed_answer",
        "reason_is_required_by_search_4",
        "counterfactual_retrievable",
        "counterfactual_answer_differs",
        "direct_question_does_not_retrieve_answer",
        "pre_reason_documents_are_not_forwarded",
        "final_answer_grounded_in_search_4",
    )
    verification = record.get("verification") or {}
    if any(verification.get(key) is not True for key in required_checks):
        return "stored verification flags are incomplete"
    return None


def relation_signature(record: dict[str, Any]) -> str:
    downstream = record.get("downstream") or {}
    return GEN.normalize(downstream.get("query_template") or downstream.get("relation"))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(GEN.to_python(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--target-accepted", type=int, default=2400)
    parser.add_argument("--max-per-source", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=1 / 12)
    parser.add_argument("--test-ratio", type=float, default=1 / 12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    batch_paths = sorted(args.batch_root.glob("*/accepted_trajectories.jsonl"))
    candidates = [(path, record) for path in batch_paths for record in read_jsonl(path)]
    random.Random(args.seed).shuffle(candidates)
    excluded_records = [record for path in args.exclude_jsonl for record in read_jsonl(path)]
    excluded = GEN.exclusion_index(excluded_records)
    rejected: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_trajectories: set[str] = set()
    seen_relations: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()

    for batch_path, record in candidates:
        reason = audit_record(record) or GEN.exclusion_reason(record, excluded)
        source = GEN.source_key(record.get("source", {}))
        question = GEN.normalize(record.get("question"))
        trajectory = str(record.get("trajectory_id", ""))
        relation = (source, relation_signature(record))
        if reason is None and question in seen_questions:
            reason = "duplicate question"
        if reason is None and trajectory in seen_trajectories:
            reason = "duplicate trajectory ID"
        if reason is None and relation in seen_relations:
            reason = "duplicate source-relation pair"
        if reason is None and source_counts[source] >= args.max_per_source:
            reason = "source trajectory cap reached"
        if reason is not None:
            rejected[reason] += 1
            continue
        record["generation_batch"] = batch_path.parent.name
        accepted.append(record)
        seen_questions.add(question)
        seen_trajectories.add(trajectory)
        seen_relations.add(relation)
        source_counts[source] += 1
        if len(accepted) >= args.target_accepted:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "accepted_trajectories.jsonl", accepted)
    stats = GEN.rebuild_outputs(
        accepted, args.output_dir, args.val_ratio, args.seed, test_ratio=args.test_ratio
    )
    report = {
        **stats,
        "target_accepted": args.target_accepted,
        "target_reached": len(accepted) >= args.target_accepted,
        "batch_files": [str(path) for path in batch_paths],
        "raw_candidates": len(candidates),
        "unique_sources": len(source_counts),
        "source_dataset_counts": dict(Counter(
            record.get("source", {}).get("source_dataset", "unknown") for record in accepted
        )),
        "top_relations": Counter(
            GEN.normalize((record.get("downstream") or {}).get("relation")) for record in accepted
        ).most_common(30),
        "answer_absent_from_pre_reason_documents": sum(
            (record.get("verification") or {}).get("answer_absent_from_pre_reason_documents") is True
            for record in accepted
        ),
        "rejection_counts": rejected.most_common(),
    }
    (args.output_dir / "merge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["target_reached"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
