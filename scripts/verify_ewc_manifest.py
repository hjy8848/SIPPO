#!/usr/bin/env python3
"""Independently audit EWC manifest semantics and overlay completeness."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def normalize(value: object) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def documents(instance: dict[str, Any]) -> list[dict[str, Any]]:
    overlay = instance["world_overlay"]
    return list(overlay.get("retrieval_documents", overlay.get("documents", [])))


def audit(path: Path) -> dict[str, Any]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    errors: list[str] = []
    seen_ids: set[str] = set()
    variant_counts: Counter[str] = Counter()
    for case in cases:
        pair_id = str(case.get("pair_id", ""))
        instances = case.get("instances", [])
        if len(instances) != 3:
            errors.append(f"{pair_id}: expected 3 instances")
            continue
        by_variant = {str(row.get("variant")): row for row in instances}
        if set(by_variant) != {"base", "dependency_changed", "answer_preserving"}:
            errors.append(f"{pair_id}: invalid variant set")
            continue
        for row in instances:
            instance_id = str(row.get("instance_id", ""))
            if not instance_id or instance_id in seen_ids:
                errors.append(f"{pair_id}: duplicate or empty instance id")
            seen_ids.add(instance_id)
            variant_counts[str(row["variant"])] += 1
            if not documents(row):
                errors.append(f"{instance_id}: empty overlay")
            if not row.get("gold_answer"):
                errors.append(f"{instance_id}: empty gold answer")
        base = by_variant["base"]
        changed = by_variant["dependency_changed"]
        preserve = by_variant["answer_preserving"]
        if base["gold_answer"] != preserve["gold_answer"]:
            errors.append(f"{pair_id}: preserve answer changed")
        if base["gold_answer"] == changed["gold_answer"]:
            errors.append(f"{pair_id}: changed answer did not change")
        if base["selected_entity"] == changed["selected_entity"]:
            errors.append(f"{pair_id}: changed entity did not change")
        base_upstream = [d.get("contents") for d in documents(base) if d.get("fact_role") == "upstream_comparison"]
        changed_upstream = [d.get("contents") for d in documents(changed) if d.get("fact_role") == "upstream_comparison"]
        if base_upstream == changed_upstream:
            errors.append(f"{pair_id}: changed upstream documents are identical")
        for variant, row in by_variant.items():
            overlay = row["world_overlay"]
            if not overlay.get("strict_query_matching"):
                continue
            upstream = " ".join(
                str(document.get("contents", ""))
                for document in documents(row)
                if document.get("fact_role") == "upstream_comparison"
            )
            if normalize(row["gold_answer"][0]) in normalize(upstream):
                errors.append(f"{pair_id}/{variant}: gold answer leaks into upstream snippet")
            query_keys = [
                normalize(query)
                for document in documents(row)
                for query in document.get("queries", [])
            ]
            if normalize(row["question"]) in query_keys:
                errors.append(f"{pair_id}/{variant}: direct question is an overlay key")
    return {
        "manifest": str(path.resolve()),
        "case_count": len(cases),
        "instance_count": sum(len(case.get("instances", [])) for case in cases),
        "variant_counts": dict(sorted(variant_counts.items())),
        "unique_instance_ids": len(seen_ids),
        "errors": errors,
        "passed": not errors,
        "manifest_digest": digest(path.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.manifest)
    target = args.output or args.manifest.with_name("independent_audit.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
