#!/usr/bin/env python3
"""Evaluate frozen-model Planner legality on the formal-v2 questions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_manager.ROSA.protocol import (  # noqa: E402
    PLANNER_BASE_SYSTEM_MSG,
    validate_plan_semantics,
    validate_plan_structure,
)


DATASETS = {
    "2wikimultihopqa": 500,
    "hotpotqa": 500,
    "musique": 500,
    "bamboogle": 125,
}
NODE_RE = re.compile(
    r'<(search|reason)\s+id="(\d+)">(.*?)</\1>', re.IGNORECASE | re.DOTALL
)
REF_RE = re.compile(r"#(\d+)#")
PLAN_RE = re.compile(r"\s*<plan>(.*?)</plan>\s*", re.IGNORECASE | re.DOTALL)
CONSTRAINTS = {
    "max_nodes": "- Use at most 5 nodes.",
    "reason_references": (
        "- Every <reason> must explicitly reference at least one previous node with #ID#."
    ),
    "concrete_search": (
        "- A <search> must ask for a concrete fact; it must not consist only of a #ID# "
        "placeholder.\n"
        "- A root <search> without #ID# must mention a specific entity or constraint from "
        "the original question."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-variant",
        choices=("canonical", *CONSTRAINTS, "validator_aligned"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def planner_prompt(variant: str) -> str:
    if variant == "canonical":
        return PLANNER_BASE_SYSTEM_MSG
    if variant in CONSTRAINTS:
        constraints = CONSTRAINTS[variant]
    if variant == "validator_aligned":
        constraints = "\n".join(CONSTRAINTS.values())
    if variant not in (*CONSTRAINTS, "validator_aligned"):
        raise ValueError(f"unknown prompt variant: {variant}")
    return (
        PLANNER_BASE_SYSTEM_MSG.rstrip()
        + "\n\nAdditional execution constraints enforced by the runtime:\n"
        + constraints
    )


def plan_nodes(output: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    match = PLAN_RE.fullmatch(output)
    if not match:
        return [], ["invalid_plan_wrapper"]
    content = match.group(1)
    nodes = NODE_RE.findall(content)
    residual = NODE_RE.sub("", content).strip()
    errors = []
    if residual:
        errors.append("unparsed_plan_content")
    if not nodes:
        errors.append("no_valid_nodes")
    return nodes, errors


def classify_plan(question: str, output: str, max_nodes: int = 5) -> dict[str, Any]:
    nodes, errors = plan_nodes(output.strip())
    ids = [int(node_id) for _, node_id, _ in nodes]
    if ids and ids != list(range(1, len(ids) + 1)):
        errors.append("nonconsecutive_or_duplicate_ids")
    seen_ids: set[int] = set()
    for node_type, node_id, text in nodes:
        current_id = int(node_id)
        references = [int(value) for value in REF_RE.findall(text)]
        if any(reference >= current_id for reference in references):
            errors.append("future_or_self_reference")
        if any(reference not in seen_ids for reference in references):
            errors.append("missing_previous_reference")
        if node_type.lower() == "search" and re.fullmatch(r"\s*#\d+#\s*", text):
            errors.append("naked_placeholder_search")
        seen_ids.add(current_id)
    errors.extend(
        "structure:" + error for error in validate_plan_structure(nodes, max_nodes)
    )
    errors.extend(
        "semantics:" + error for error in validate_plan_semantics(question, nodes)
    )
    errors = list(dict.fromkeys(errors))
    topology = "-".join(node_type[0].upper() for node_type, _, _ in nodes)
    reason_search = any(
        nodes[index][0].lower() == "reason"
        and nodes[index + 1][0].lower() == "search"
        for index in range(len(nodes) - 1)
    )
    return {
        "valid": not errors,
        "errors": errors,
        "node_count": len(nodes),
        "topology": topology,
        "reason_search": reason_search,
    }


def load_questions(manifest_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset, expected in DATASETS.items():
        path = manifest_dir / f"{dataset}.parquet"
        frame = pd.read_parquet(path)
        if len(frame) != expected:
            raise ValueError(f"unexpected {dataset} count: {len(frame)}/{expected}")
        for sample_index, question in enumerate(frame["question"]):
            rows.append(
                {
                    "dataset": dataset,
                    "sample_index": sample_index,
                    "question": str(question).strip(),
                }
            )
    return rows


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    errors = Counter(error for row in rows for error in row["errors"])
    node_counts = Counter(str(row["node_count"]) for row in rows)
    topologies = Counter(row["topology"] for row in rows)
    return {
        "samples": len(rows),
        "valid_rate": sum(row["valid"] for row in rows) / len(rows),
        "reason_search_rate": sum(row["reason_search"] for row in rows) / len(rows),
        "mean_node_count": sum(row["node_count"] for row in rows) / len(rows),
        "error_counts": dict(sorted(errors.items())),
        "node_counts": dict(sorted(node_counts.items(), key=lambda item: int(item[0]))),
        "topologies": dict(topologies.most_common()),
    }


@torch.inference_mode()
def generate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    system_prompt = planner_prompt(args.prompt_variant)
    prepared = []
    for row in rows:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question: {row['question']}\nGenerate the execution plan.",
            },
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        prepared.append((row, list(input_ids)))
    prepared.sort(key=lambda item: len(item[1]))

    results = []
    for offset in range(0, len(prepared), args.batch_size):
        batch = prepared[offset : offset + args.batch_size]
        encoded = tokenizer.pad(
            {"input_ids": [input_ids for _, input_ids in batch]},
            padding=True,
            return_tensors="pt",
        ).to(args.device)
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_width = encoded["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )
        for (row, _), output in zip(batch, outputs, strict=True):
            results.append(
                {
                    **row,
                    "output": output.strip(),
                    **classify_plan(row["question"], output),
                }
            )
        print(
            json.dumps({"generated": len(results), "total": len(rows)}),
            flush=True,
        )
    return results


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory exists: {args.output_dir}")
    rows = load_questions(args.manifest_dir)
    args.output_dir.mkdir(parents=True)
    results = generate(args, rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["dataset"]].append(row)
    report = {
        "schema_version": 1,
        "model": args.model,
        "manifest_dir": str(args.manifest_dir),
        "prompt_variant": args.prompt_variant,
        "max_new_tokens": args.max_new_tokens,
        "overall": summarize(results),
        "datasets": {dataset: summarize(grouped[dataset]) for dataset in DATASETS},
    }
    with (args.output_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda item: (item["dataset"], item["sample_index"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
