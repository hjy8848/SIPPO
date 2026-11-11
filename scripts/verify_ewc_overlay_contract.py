#!/usr/bin/env python3
"""Exercise strict EWC overlay retrieval without starting an HTTP server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


def normalize(value: object) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def load_overlay_state(instances: list[dict[str, Any]], server_path: Path) -> Any:
    specification = importlib.util.spec_from_file_location("ewc_overlay", server_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import overlay server: {server_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.OverlayState(instances)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=Path(__file__).with_name("serve_ewc_overlay_retriever.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    instances = [json.loads(line) for line in args.instances.read_text(encoding="utf-8").splitlines() if line.strip()]
    state = load_overlay_state(instances, args.server)
    errors: list[dict[str, Any]] = []
    declared_queries = 0
    for instance in instances:
        instance_id = str(instance["instance_id"])
        state.select(instance_id)
        direct = state.retrieve(str(instance["question"]), 3)
        if direct:
            errors.append({"instance_id": instance_id, "error": "direct_question_not_empty"})
        documents = instance["world_overlay"].get(
            "retrieval_documents", instance["world_overlay"].get("documents", [])
        )
        upstream = " ".join(
            str(document.get("contents", ""))
            for document in documents
            if document.get("fact_role") == "upstream_comparison"
        )
        if normalize(instance["gold_answer"][0]) in normalize(upstream):
            errors.append({"instance_id": instance_id, "error": "upstream_gold_leak"})
        for document in documents:
            for query in document.get("queries", []):
                declared_queries += 1
                result_ids = {
                    str(item["document"].get("document_id", ""))
                    for item in state.retrieve(str(query), 3)
                }
                if str(document["document_id"]) not in result_ids:
                    errors.append(
                        {
                            "instance_id": instance_id,
                            "error": "declared_query_miss",
                            "document_id": document["document_id"],
                            "query": query,
                        }
                    )
    report = {
        "instances": len(instances),
        "declared_queries_checked": declared_queries,
        "direct_questions_returned_empty": len(instances),
        "errors": errors,
        "passed": not errors,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
