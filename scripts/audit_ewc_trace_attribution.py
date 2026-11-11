#!/usr/bin/env python3
"""Attribute EWC-QA outcomes from recorded execution traces.

This audit does not score a plan against a prescribed topology.  It only
records whether an execution reached the target relation, whether a node had
already produced the gold answer, and whether the final answer consumed it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_relation_terms(relation: str) -> tuple[str, ...]:
    terms = {
        "alma mater": ("alma mater", "college", "university", "attend"),
        "artist": ("artist", "performed", "performer", "recorded"),
        "city": ("city", "located"),
        "country": ("country", "located"),
        "date of death": ("die", "death"),
        "director": ("director", "directed"),
        "editor": ("editor", "edited"),
        "founder": ("founder", "founded"),
        "genre": ("genre",),
        "headquarters location": ("headquarters", "location"),
        "nationality": ("nationality",),
        "occupation": ("occupation", "profession"),
        "place of birth": ("born", "birthplace"),
        "publisher": ("publisher", "published"),
        "spouse": ("spouse", "married"),
        "starring": ("starred", "starring", "cast"),
    }
    return terms.get(relation.lower(), (relation.lower(),))


def has_target_relation(query: str, relation: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in target_relation_terms(relation))


def case_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row["variant"] == "base":
            metadata[str(row["pair_id"])] = row
    return metadata


def audit_row(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    trace = row["trace"]
    nodes = trace.get("nodes_execution", [])
    gold = str(metadata["gold_answer"][0])
    normalized_gold = normalize(gold)
    documents = metadata["world_overlay"].get(
        "retrieval_documents", metadata["world_overlay"].get("documents", [])
    )
    upstream_text = "\n".join(
        str(document.get("contents", ""))
        for document in documents
        if document.get("fact_role") == "upstream_comparison"
    )
    upstream_gold_leak = normalized_gold in normalize(upstream_text)
    queries = [str(node.get("query", "")) for node in nodes]
    conclusions = [str(node.get("conclusion", "")) for node in nodes]
    target_nodes = [
        node
        for node in nodes
        if has_target_relation(str(node.get("query", "")), str(metadata["downstream_relation"]))
    ]
    gold_node_ids = [
        str(node.get("id", ""))
        for node in nodes
        if normalize(node.get("conclusion", "")) == normalized_gold
    ]
    final_correct = bool(row["em"])
    final_matches_gold_node = final_correct and bool(gold_node_ids)
    if final_correct:
        outcome_class = "final_correct"
    elif not target_nodes:
        outcome_class = "did_not_execute_target_relation"
    elif gold_node_ids:
        outcome_class = "gold_reached_but_not_consumed"
    else:
        outcome_class = "target_relation_attempted_but_wrong"
    return {
        "pair_id": row["pair_id"],
        "question": row["question"],
        "selected_entity": metadata["selected_entity"],
        "downstream_relation": metadata["downstream_relation"],
        "gold_answer": gold,
        "upstream_gold_leak": upstream_gold_leak,
        "final_answer": row["final_answer"],
        "final_correct": final_correct,
        "pipeline_error": bool(row["pipeline_error"]),
        "node_count": len(nodes),
        "reason_node_count": sum(str(node.get("type", "")).lower() == "reason" for node in nodes),
        "symbolic_dependency_node_count": sum("#" in query for query in queries),
        "target_relation_node_count": len(target_nodes),
        "gold_node_ids": gold_node_ids,
        "final_matches_gold_node": final_matches_gold_node,
        "outcome_class": outcome_class,
        "queries": queries,
        "conclusions": conclusions,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    return {
        "cases": count,
        "final_correct": sum(record["final_correct"] for record in records),
        "pipeline_errors": sum(record["pipeline_error"] for record in records),
        "reason_node_cases": sum(record["reason_node_count"] > 0 for record in records),
        "symbolic_dependency_cases": sum(record["symbolic_dependency_node_count"] > 0 for record in records),
        "target_relation_cases": sum(record["target_relation_node_count"] > 0 for record in records),
        "gold_reached_cases": sum(bool(record["gold_node_ids"]) for record in records),
        "upstream_gold_leak_cases": sum(record["upstream_gold_leak"] for record in records),
        "outcome_classes": dict(Counter(record["outcome_class"] for record in records)),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EWC-QA trace attribution",
        "",
        "This is a descriptive audit of the 20 base worlds. It does not treat a prescribed operator sequence as a learning target.",
        "",
        "| Model | Correct | Upstream gold leak | Reason cases | Symbolic-dependency cases | Target-relation cases | Gold reached in a node | No target relation | Gold reached, final missed | Target attempted, wrong |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, entry in report["models"].items():
        summary = entry["summary"]
        classes = summary["outcome_classes"]
        lines.append(
            "| {label} | {correct}/{cases} | {leak}/{cases} | {reason}/{cases} | {deps}/{cases} | {target}/{cases} | "
            "{gold}/{cases} | {no_target} | {not_consumed} | {wrong} |".format(
                label=label,
                correct=summary["final_correct"],
                cases=summary["cases"],
                leak=summary["upstream_gold_leak_cases"],
                reason=summary["reason_node_cases"],
                deps=summary["symbolic_dependency_cases"],
                target=summary["target_relation_cases"],
                gold=summary["gold_reached_cases"],
                no_target=classes.get("did_not_execute_target_relation", 0),
                not_consumed=classes.get("gold_reached_but_not_consumed", 0),
                wrong=classes.get("target_relation_attempted_but_wrong", 0),
            )
        )
    lines.extend(["", "## PSC base-case records", ""])
    psc = report["models"].get("PSC")
    if psc:
        lines.extend(["| Pair | Outcome | Relation | Gold | Final |", "| --- | --- | --- | --- | --- |"])
        for record in psc["records"]:
            lines.append(
                "| {pair} | {outcome} | {relation} | {gold} | {final} |".format(
                    pair=record["pair_id"],
                    outcome=record["outcome_class"],
                    relation=record["downstream_relation"],
                    gold=record["gold_answer"].replace("|", "\\|"),
                    final=record["final_answer"].replace("|", "\\|"),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=RESULTS_JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    metadata = case_metadata(args.cases)
    models: dict[str, Any] = {}
    for item in args.input:
        label, separator, raw_path = item.partition("=")
        if not separator:
            raise SystemExit(f"invalid input: {item}")
        rows = [row for row in read_jsonl(Path(raw_path)) if row["variant"] == "base"]
        records = [audit_row(row, metadata[str(row["pair_id"])]) for row in rows]
        models[label] = {"summary": summarize(records), "records": records}
    report = {"benchmark": "EWC-QA validation v6", "scope": "base worlds only", "models": models}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "attribution.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "attribution.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({label: data["summary"] for label, data in models.items()}, indent=2))


if __name__ == "__main__":
    main()
