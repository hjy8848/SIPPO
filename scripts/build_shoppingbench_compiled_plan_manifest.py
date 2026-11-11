"""Compile frozen ShoppingBench planner outputs into executable system-owned plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_manager.ROSA.contract_protocol import parse_contract_plan, validate_contract_plan
from qa_manager.ROSA.shoppingbench_protocol import compile_shoppingbench_plan


def load_contract_rows(input_dir: Path) -> list[dict]:
    paths = sorted(input_dir.glob("contract/*/results.jsonl"))
    if not paths:
        raise ValueError(f"No contract results found under {input_dir}")
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=5)
    args = parser.parse_args()

    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {}
    audit_rows = []
    for row in load_contract_rows(args.input_dir):
        sample_id = str(row.get("sample_id") or "")
        trace = row.get("raw_trace") or {}
        plan, metadata = compile_shoppingbench_plan(
            str(row.get("query") or ""),
            str(trace.get("raw_plan_output") or ""),
            max_nodes=args.max_nodes,
        )
        validation_errors = validate_contract_plan(
            str(row.get("query") or ""),
            parse_contract_plan(plan),
            max_nodes=args.max_nodes,
        )
        if validation_errors:
            raise ValueError(f"{sample_id}: compiled plan is invalid: {validation_errors}")
        manifest[sample_id] = plan
        audit_rows.append({"sample_id": sample_id, **metadata, "plan": plan})

    report = {
        "examples": len(audit_rows),
        "all_compiled_plans_valid": True,
        "gold_fields_read": False,
        "source_fields": ["query", "raw_trace.raw_plan_output"],
        "rows": audit_rows,
        "evidence_level": "frozen-planner-output compilation audit",
        "scientific_improvement_claim": False,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
