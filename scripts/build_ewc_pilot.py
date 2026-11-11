#!/usr/bin/env python3
"""Build a small execution/workflow-consistency pilot from grounded records.

The pilot is a manifest, not a claim that an overlay corpus is already served.
Each source record yields a base instance, an upstream-fact-changing instance,
and an answer-preserving control. The verifier checks the pair semantics and
the source grounding metadata before any model evaluation is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/dagent_reason_search_wiki18_v1/accepted_trajectories.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/ewc_qa_pilot_v1"


def stable_id(*values: object) -> str:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize(value: object) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def first_relation(record: dict[str, Any]) -> str:
    """Return the relation actually executed by downstream Search node 4."""
    node4 = next(
        (node for node in record.get("nodes_execution", []) if str(node.get("id")) == "4"),
        None,
    )
    downstream = record.get("blueprint", {}).get("downstream", [])
    executed = str((node4 or {}).get("executed_query", "")).strip().casefold()
    for proposal in downstream:
        if not isinstance(proposal, dict):
            continue
        template = str(proposal.get("query_template", ""))
        rendered = template.replace("{entity}", str(record.get("winner", ""))).strip().casefold()
        if executed and rendered == executed:
            relation = str(proposal.get("relation", "")).strip()
            if relation:
                return relation
    if not downstream or not isinstance(downstream[0], dict):
        raise ValueError(f"missing downstream relation: {record.get('trajectory_id')}")
    relation = str(downstream[0].get("relation", "")).strip()
    if not relation:
        raise ValueError(f"empty downstream relation: {record.get('trajectory_id')}")
    return relation


def comparison_overlay(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record["comparison_evidence"]
    left = evidence["candidate_a"]
    right = evidence["candidate_b"]
    return {
        "operation": record["blueprint"].get("operation"),
        "comparison_family": record["source"].get("source_question_type"),
        "documents": [
            {
                "entity": record["blueprint"]["candidate_a"],
                "document_id": left.get("supporting_document_id"),
                "original_fact": left.get("answer"),
                "counterfactual_fact": right.get("answer"),
                "original_quote": left.get("evidence_quote"),
            },
            {
                "entity": record["blueprint"]["candidate_b"],
                "document_id": right.get("supporting_document_id"),
                "original_fact": right.get("answer"),
                "counterfactual_fact": left.get("answer"),
                "original_quote": right.get("evidence_quote"),
            },
        ],
        "update_rule": "replace each candidate comparison fact with the other candidate's verified comparison value; keep downstream entity facts unchanged",
    }


def unique_queries(*queries: object) -> list[str]:
    """Keep a finite, normalized set of retriever query keys."""
    return sorted({str(query).strip() for query in queries if str(query).strip()})


def comparison_query_aliases(entity: str, canonical_query: str) -> list[str]:
    """Return controlled paraphrases for common temporal comparison facts."""
    query = canonical_query.strip()
    query_lower = query.casefold()
    aliases: list[str] = [query]
    if "date of birth" in query_lower or "birth date" in query_lower:
        aliases.extend(
            [
                f"{entity} birth date",
                f"When was {entity} born?",
                f"What is {entity}'s date of birth?",
            ]
        )
    elif "date of death" in query_lower or "death date" in query_lower:
        aliases.extend(
            [
                f"{entity} death date",
                f"When did {entity} die?",
                f"What is {entity}'s date of death?",
            ]
        )
    elif "release date" in query_lower or "release year" in query_lower:
        aliases.extend([f"When was {entity} released?", f"{entity} release year"])
    elif "founding year" in query_lower or "year established" in query_lower:
        aliases.extend([f"When was {entity} founded?", f"{entity} founding year"])
    return unique_queries(*aliases)


def downstream_query_aliases(entity: str, relation: str, canonical_query: str) -> list[str]:
    """Return relation-preserving atomic queries without admitting the full question."""
    relation_lower = relation.casefold()
    aliases: list[str] = [canonical_query, f"{entity} {relation}"]
    if relation_lower == "alma mater":
        aliases.extend(
            [
                f"{entity} college",
                f"{entity} university",
                f"Which college did {entity} attend?",
                f"What is {entity}'s alma mater?",
            ]
        )
    elif relation_lower == "place of birth":
        aliases.extend([f"Where was {entity} born?", f"{entity} birthplace"])
    elif relation_lower == "occupation":
        aliases.extend([f"What is {entity}'s occupation?", f"What does {entity} do?"])
    elif relation_lower in {"director", "artist", "founder", "editor", "spouse", "starring"}:
        aliases.append(f"Who is {entity}'s {relation}?")
    else:
        aliases.extend([f"What is {entity}'s {relation}?", f"What was {entity}'s {relation}?"])
    return unique_queries(*aliases)


def paired_comparison_query_aliases(record: dict[str, Any]) -> list[str]:
    """Return finite aliases for a two-entity comparison without an answer label."""
    source_question = str(record.get("source", {}).get("source_question", "")).strip()
    blueprint = record["blueprint"]
    left = str(blueprint["candidate_a"])
    right = str(blueprint["candidate_b"])
    question_lower = source_question.casefold()
    aliases: list[str] = [source_question]
    if "older" in question_lower:
        aliases.extend(
            [
                f"Who is older, {left} or {right}?",
                f"Which person is older, {left} or {right}?",
            ]
        )
    elif "younger" in question_lower:
        aliases.extend(
            [
                f"Who is younger, {left} or {right}?",
                f"Which person is younger, {left} or {right}?",
            ]
        )
    elif "born first" in question_lower:
        aliases.extend(
            [
                f"Who was born first, {left} or {right}?",
                f"Which person was born earlier, {left} or {right}?",
            ]
        )
    elif "born later" in question_lower:
        aliases.extend(
            [
                f"Who was born later, {left} or {right}?",
                f"Which person was born later, {left} or {right}?",
            ]
        )
    elif "more recently" in question_lower:
        aliases.extend(
            [
                f"Which film was released more recently, {left} or {right}?",
                f"Which film was released later, {left} or {right}?",
            ]
        )
    elif "film" in question_lower and ("released first" in question_lower or "came out" in question_lower):
        aliases.extend(
            [
                f"Which film was released earlier, {left} or {right}?",
                f"Which film came out earlier, {left} or {right}?",
            ]
        )
    elif "opened first" in question_lower:
        aliases.append(f"Which museum was opened earlier, {left} or {right}?")
    elif "established first" in question_lower:
        aliases.extend(
            [
                f"Which was established earlier, {left} or {right}?",
                f"Which entity was founded earlier, {left} or {right}?",
            ]
        )
    return unique_queries(*aliases)


def retrieval_documents(
    record: dict[str, Any],
    *,
    swap_comparison: bool,
    fact_snippets: bool,
    finite_query_aliases: bool,
    finite_comparison_query_aliases: bool,
) -> list[dict[str, Any]]:
    blueprint = record["blueprint"]
    evidence = record["comparison_evidence"]
    left = evidence["candidate_a"]
    right = evidence["candidate_b"]
    left_value = right["answer"] if swap_comparison else left["answer"]
    right_value = left["answer"] if swap_comparison else right["answer"]
    winner_node = next(
        node for node in record["nodes_execution"] if str(node.get("id")) == "4"
    )
    comparison_nodes = {
        str(node.get("id")): node
        for node in record.get("nodes_execution", [])
        if str(node.get("id")) in {"1", "2"}
    }
    documents: list[dict[str, Any]] = []
    for label, fact, replacement, entity, node_id, query in (
        (
            "comparison-a",
            left,
            right["answer"] if swap_comparison else left["answer"],
            blueprint["candidate_a"],
            "1",
            blueprint["search_query_a"],
        ),
        (
            "comparison-b",
            right,
            left["answer"] if swap_comparison else right["answer"],
            blueprint["candidate_b"],
            "2",
            blueprint["search_query_b"],
        ),
    ):
        source_doc = None
        for node in comparison_nodes.values():
            for candidate in node.get("documents", []):
                if str(candidate.get("document_id")) == str(fact.get("supporting_document_id")):
                    source_doc = candidate
                    break
            if source_doc is not None:
                break
        if source_doc is None:
            raise ValueError(
                f"missing comparison supporting document {fact.get('supporting_document_id')}"
            )
        if fact_snippets:
            contents = f"{entity}\n{query}: {replacement}."
        else:
            contents = str(source_doc.get("contents", ""))
        if swap_comparison and not fact_snippets:
            original_value = str(fact.get("answer", ""))
            if not original_value or original_value not in contents:
                raise ValueError(
                    f"comparison answer missing from document {fact.get('supporting_document_id')}"
                )
            contents = contents.replace(
                original_value,
                str(replacement),
                1,
            )
        source_queries = unique_queries(
            fact.get("query", ""),
            query,
            comparison_nodes[node_id].get("executed_query", ""),
        )
        documents.append(
            {
                "document_id": f"ewc-{record['trajectory_id']}-{label}",
                "title": source_doc.get("title", fact.get("supporting_title", "")),
                "queries": unique_queries(
                    *(
                        comparison_query_aliases(entity, str(query))
                        if finite_query_aliases
                        else source_queries
                    ),
                    *(
                        paired_comparison_query_aliases(record)
                        if finite_comparison_query_aliases
                        else []
                    ),
                ),
                "contents": contents,
                "fact_role": "upstream_comparison",
            }
        )
    source_answer = str(record["answer"])
    alternate_answer = str(record["counterfactual_answer"])
    source_entity = str(record["winner"])
    alternate_entity = str(record["counterfactual_entity"])
    relation = first_relation(record)
    documents.extend([
        {
            "document_id": f"ewc-{record['trajectory_id']}-downstream-source",
            "title": winner_node.get("documents", [{}])[0].get("title", record["winner"]),
            "queries": (
                downstream_query_aliases(source_entity, relation, str(winner_node["executed_query"]))
                if finite_query_aliases
                else [winner_node["executed_query"]]
            ),
            "contents": (
                f"{source_entity}\n{relation}: {source_answer}."
                if fact_snippets
                else "\n\n".join(
                    str(document.get("contents", ""))
                    for document in winner_node.get("documents", [])
                )
            ),
            "fact_role": "downstream_answer",
        },
        {
            "document_id": f"ewc-{record['trajectory_id']}-downstream-alternate",
            "title": record["counterfactual"].get("supporting_title", record["counterfactual_entity"]),
            "queries": (
                downstream_query_aliases(
                    alternate_entity,
                    relation,
                    str(record["counterfactual"]["executed_query"]),
                )
                if finite_query_aliases
                else [record["counterfactual"]["executed_query"]]
            ),
            "contents": (
                f"{alternate_entity}\n{relation}: {alternate_answer}."
                if fact_snippets
                else "\n\n".join(
                    str(document.get("contents", ""))
                    for document in record["counterfactual"].get("documents", [])
                )
            ),
            "fact_role": "downstream_answer",
        },
    ])
    return documents


def source_metadata(record: dict[str, Any]) -> dict[str, Any]:
    verification = record.get("verification", {})
    return {
        "source_trajectory_id": record.get("trajectory_id"),
        "source_dataset": record.get("source", {}).get("source_dataset"),
        "source_question": record.get("source", {}).get("source_question"),
        "supporting_titles": record.get("supporting_titles", []),
        "direct_question_does_not_retrieve_answer": verification.get(
            "direct_question_does_not_retrieve_answer"
        ),
        "comparison_reproduces_seed_answer": verification.get(
            "comparison_reproduces_seed_answer"
        ),
        "counterfactual_retrievable": verification.get("counterfactual_retrievable"),
        "counterfactual_answer_differs": verification.get("counterfactual_answer_differs"),
        "pre_reason_documents_are_not_forwarded": verification.get(
            "pre_reason_documents_are_not_forwarded"
        ),
    }


def strict_safe_source(record: dict[str, Any]) -> bool:
    """Reject cases whose question or upstream facts already name an answer."""
    blueprint = record["blueprint"]
    evidence = record["comparison_evidence"]
    visible_upstream = " ".join(
        [
            str(record["question"]),
            str(blueprint["candidate_a"]),
            str(blueprint["candidate_b"]),
            str(blueprint["search_query_a"]),
            str(blueprint["search_query_b"]),
            str(evidence["candidate_a"].get("answer", "")),
            str(evidence["candidate_b"].get("answer", "")),
        ]
    )
    visible_norm = normalize(visible_upstream)
    return all(
        answer and normalize(answer) not in visible_norm
        for answer in (record["answer"], record["counterfactual_answer"])
    )


def build_case(
    record: dict[str, Any],
    index: int,
    *,
    fact_snippets: bool,
    finite_query_aliases: bool,
    finite_comparison_query_aliases: bool,
) -> dict[str, Any]:
    relation = first_relation(record)
    pair_id = f"ewc-{index:03d}-{stable_id(record.get('trajectory_id'), relation)}"
    common = {
        "pair_id": pair_id,
        "question": record["question"],
        "candidate_a": record["blueprint"]["candidate_a"],
        "candidate_b": record["blueprint"]["candidate_b"],
        "comparison_query_a": record["blueprint"]["search_query_a"],
        "comparison_query_b": record["blueprint"]["search_query_b"],
        "downstream_relation": relation,
        "source": source_metadata(record),
        "answer_options": {
            "source": record["answer"],
            "alternate": record["counterfactual_answer"],
        },
    }
    base_documents = retrieval_documents(
        record,
        swap_comparison=False,
        fact_snippets=fact_snippets,
        finite_query_aliases=finite_query_aliases,
        finite_comparison_query_aliases=finite_comparison_query_aliases,
    )
    changed_documents = retrieval_documents(
        record,
        swap_comparison=True,
        fact_snippets=fact_snippets,
        finite_query_aliases=finite_query_aliases,
        finite_comparison_query_aliases=finite_comparison_query_aliases,
    )
    distractor = {
        "document_id": f"{pair_id}-distractor",
        "title": "Unrelated EWC control document",
        "queries": ["ewc unrelated distractor"],
        "contents": "This control document is unrelated to either candidate and contains no benchmark answer.",
        "fact_role": "irrelevant_distractor",
    }
    return {
        "pair_id": pair_id,
        "case_type": "selection_then_lookup",
        "instances": [
            {
                **common,
                "instance_id": f"{pair_id}-base",
                "world_id": f"{pair_id}-world-base",
                "variant": "base",
                "expected_answer_relation": "source",
                "gold_answer": [record["answer"]],
                "selected_entity": record["winner"],
                "world_overlay": {
                    "type": "strict_fact_snippet_closed_world" if fact_snippets else "explicit_closed_world",
                    "strict_query_matching": fact_snippets,
                    "query_alias_policy": "finite_relation_and_pair_comparison_v2" if finite_comparison_query_aliases else ("finite_relation_preserving_v1" if finite_query_aliases else "canonical_only"),
                    "snippet_contract": "upstream comparison snippets omit final-answer aliases" if fact_snippets else None,
                    "documents": base_documents,
                },
            },
            {
                **common,
                "instance_id": f"{pair_id}-changed",
                "world_id": f"{pair_id}-world-changed",
                "variant": "dependency_changed",
                "expected_answer_relation": "must_change",
                "gold_answer": [record["counterfactual_answer"]],
                "selected_entity": record["counterfactual_entity"],
                "world_overlay": {
                    "type": "strict_fact_snippet_upstream_fact_swap" if fact_snippets else "verified_upstream_fact_swap",
                    "strict_query_matching": fact_snippets,
                    "query_alias_policy": "finite_relation_and_pair_comparison_v2" if finite_comparison_query_aliases else ("finite_relation_preserving_v1" if finite_query_aliases else "canonical_only"),
                    "snippet_contract": "upstream comparison snippets omit final-answer aliases" if fact_snippets else None,
                    **comparison_overlay(record),
                    "retrieval_documents": changed_documents,
                },
            },
            {
                **common,
                "instance_id": f"{pair_id}-preserve",
                "world_id": f"{pair_id}-world-preserve",
                "variant": "answer_preserving",
                "expected_answer_relation": "must_stay",
                "gold_answer": [record["answer"]],
                "selected_entity": record["winner"],
                "world_overlay": {
                    "type": "strict_fact_snippet_answer_preserving_distractor" if fact_snippets else "answer_preserving_distractor",
                    "strict_query_matching": fact_snippets,
                    "query_alias_policy": "finite_relation_and_pair_comparison_v2" if finite_comparison_query_aliases else ("finite_relation_preserving_v1" if finite_query_aliases else "canonical_only"),
                    "snippet_contract": "upstream comparison snippets omit final-answer aliases" if fact_snippets else None,
                    "operations": [
                        "permute_non_supporting_document_order",
                        "insert_same_dataset_distractor_documents",
                    ],
                    "protected_document_ids": [
                        item.get("supporting_document_id")
                        for item in (
                            record["comparison_evidence"]["candidate_a"],
                            record["comparison_evidence"]["candidate_b"],
                        )
                    ],
                    "update_rule": "do not alter facts, titles, or documents supporting the selected entity and downstream answer",
                    "retrieval_documents": [distractor, *reversed(base_documents)],
                },
            },
        ],
    }


def validate_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    instances = {row["variant"]: row for row in case["instances"]}
    required = {"base", "dependency_changed", "answer_preserving"}
    if set(instances) != required:
        errors.append(f"variants={sorted(instances)}")
        return errors
    if instances["base"]["gold_answer"] != instances["answer_preserving"]["gold_answer"]:
        errors.append("answer-preserving control changes the gold answer")
    if instances["base"]["gold_answer"] == instances["dependency_changed"]["gold_answer"]:
        errors.append("dependency-changing variant does not change the gold answer")
    if instances["base"]["selected_entity"] == instances["dependency_changed"]["selected_entity"]:
        errors.append("dependency-changing variant does not change the selected entity")
    for row in case["instances"]:
        source = row["source"]
        for field in (
            "direct_question_does_not_retrieve_answer",
            "comparison_reproduces_seed_answer",
            "counterfactual_retrievable",
            "counterfactual_answer_differs",
        ):
            if source.get(field) is not True:
                errors.append(f"{row['instance_id']}: source audit {field} is not true")
        documents = row["world_overlay"].get(
            "retrieval_documents", row["world_overlay"].get("documents", [])
        )
        if len(documents) < 4:
            errors.append(f"{row['instance_id']}: fewer than four overlay documents")
        joined = " ".join(str(document.get("contents", "")) for document in documents)
        if not all(str(answer) in joined for answer in row["answer_options"].values()):
            errors.append(f"{row['instance_id']}: an answer option is absent from the overlay")
        if row["world_overlay"].get("strict_query_matching"):
            upstream = " ".join(
                str(document.get("contents", ""))
                for document in documents
                if document.get("fact_role") == "upstream_comparison"
            )
            if normalize(row["gold_answer"][0]) in normalize(upstream):
                errors.append(f"{row['instance_id']}: gold answer leaks into an upstream snippet")
            if any(normalize(row["question"]) == normalize(query) for document in documents for query in document.get("queries", [])):
                errors.append(f"{row['instance_id']}: direct question is an overlay retrieval key")
    return errors


def select_records(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Round-robin over operation/relation buckets for a deterministic pilot."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["blueprint"].get("operation", "")), first_relation(record))
        buckets[key].append(record)
    selected: list[dict[str, Any]] = []
    used_source_ids: set[str] = set()
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            for record in buckets[key]:
                source_id = str(record.get("trajectory_id", ""))
                if source_id in used_source_ids:
                    continue
                selected.append(record)
                used_source_ids.add(source_id)
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    if len(selected) < count:
        raise ValueError(f"only found {len(selected)} unique records for requested {count}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument(
        "--strict-fact-snippets",
        action="store_true",
        help="replace full pages with minimal facts and enable exact overlay query matching",
    )
    parser.add_argument(
        "--finite-query-aliases",
        action="store_true",
        help="permit a finite, predeclared set of relation-preserving atomic query aliases",
    )
    parser.add_argument(
        "--finite-comparison-query-aliases",
        action="store_true",
        help="permit finite aliases for pairwise comparison queries that return both atomic facts",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    records = read_jsonl(args.source)
    if args.finite_query_aliases and not args.strict_fact_snippets:
        raise ValueError("--finite-query-aliases requires --strict-fact-snippets")
    if args.finite_comparison_query_aliases and not args.finite_query_aliases:
        raise ValueError("--finite-comparison-query-aliases requires --finite-query-aliases")
    if args.strict_fact_snippets:
        records = [record for record in records if strict_safe_source(record)]
    selected = select_records(records, args.cases)
    cases = [
        build_case(
            record,
            index,
            fact_snippets=args.strict_fact_snippets,
            finite_query_aliases=args.finite_query_aliases,
            finite_comparison_query_aliases=args.finite_comparison_query_aliases,
        )
        for index, record in enumerate(selected, 1)
    ]
    errors = [error for case in cases for error in validate_case(case)]
    if errors:
        raise ValueError("pilot validation failed: " + "; ".join(errors))
    instances = [instance for case in cases for instance in case["instances"]]
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "cases.jsonl", cases)
    write_jsonl(args.output_dir / "instances.jsonl", instances)
    report = {
        "schema_version": 4 if args.finite_comparison_query_aliases else (3 if args.finite_query_aliases else (2 if args.strict_fact_snippets else 1)),
        "name": (
            "EWC-QA strict fact-snippet finite atomic-and-pair-alias pilot"
            if args.finite_comparison_query_aliases
            else "EWC-QA strict fact-snippet finite-alias pilot"
            if args.finite_query_aliases
            else ("EWC-QA strict fact-snippet pilot" if args.strict_fact_snippets else "EWC-QA pilot")
        ),
        "status": "constructed_not_model_evaluated",
        "source": str(args.source),
        "case_count": len(cases),
        "instance_count": len(instances),
        "variants_per_case": ["base", "dependency_changed", "answer_preserving"],
        "case_types": sorted({case["case_type"] for case in cases}),
        "validation": {"errors": [], "passed": True},
        "important_boundary": (
            "the explicit overlay corpus is synthetic; strict fact snippets and finite predeclared atomic "
            "query aliases prevent upstream final-answer leakage and direct-question shortcuts, but the "
            "manifest must still be validated before model evaluation"
            if args.finite_query_aliases
            else "the explicit overlay corpus is synthetic; strict fact snippets and exact query matching prevent "
            "upstream final-answer leakage, but the manifest must still be validated before model evaluation"
            if args.strict_fact_snippets
            else "the explicit overlay corpus is synthetic and must be served through the same retrieval interface for model evaluation"
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
