#!/usr/bin/env python3
"""Evaluate frozen-model Reasoner generation on DAGENT preference prompts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.score_dagent_preference_baseline import read_jsonl, write_json
except ImportError:
    from score_dagent_preference_baseline import read_jsonl, write_json


CONCLUSION_RE = re.compile(r"^<conclusion>([^<>\n]+)</conclusion>$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--splits", nargs="+", default=("train", "validation", "test"))
    parser.add_argument(
        "--causal-sft-report",
        type=Path,
        help="Unified-SFT mix report whose selected trajectory IDs define seen examples.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def normalized_entity(text: str) -> str:
    value = text.strip()
    match = CONCLUSION_RE.fullmatch(value)
    if match:
        value = match.group(1)
    value = value.strip("{} ")
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def classify_output(
    row: dict[str, Any], output: str, seen_group_ids: set[str] | None = None
) -> dict[str, Any]:
    output_entity = normalized_entity(output)
    chosen_entity = normalized_entity(str(row["chosen"]))
    rejected_entity = normalized_entity(str(row["rejected"]))
    semantic_label = (
        "chosen" if output_entity == chosen_entity
        else "rejected" if output_entity == rejected_entity
        else "other"
    )
    result = {
        "pair_id": row["pair_id"],
        "group_id": row["group_id"],
        "split": row["split"],
        "variant": row["variant"],
        "operation": row["operation"],
        "output": output,
        "format_valid": bool(CONCLUSION_RE.fullmatch(output.strip())),
        "semantic_label": semantic_label,
        "chosen_exact": output.strip() == str(row["chosen"]).strip(),
    }
    if seen_group_ids is not None:
        result["causal_sft_exposure"] = (
            "seen" if str(row["group_id"]) in seen_group_ids else "held_out"
        )
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "format_valid_rate": sum(row["format_valid"] for row in rows) / len(rows),
        "semantic_chosen_rate": sum(row["semantic_label"] == "chosen" for row in rows) / len(rows),
        "semantic_rejected_rate": sum(row["semantic_label"] == "rejected" for row in rows) / len(rows),
        "semantic_other_rate": sum(row["semantic_label"] == "other" for row in rows) / len(rows),
        "chosen_exact_rate": sum(row["chosen_exact"] for row in rows) / len(rows),
    }


def grouped_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: summarize(grouped[key]) for key in sorted(grouped)}


def nested_grouped_summary(
    rows: list[dict[str, Any]], outer_field: str, inner_field: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[outer_field])].append(row)
    return {
        key: grouped_summary(grouped[key], inner_field) for key in sorted(grouped)
    }


@torch.inference_mode()
def generate(
    model, tokenizer, rows, batch_size, max_new_tokens, device, seen_group_ids=None
):
    tokenizer.padding_side = "left"
    prepared = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        prepared.append((row, list(ids)))
    prepared.sort(key=lambda item: len(item[1]))
    results = []
    for offset in range(0, len(prepared), batch_size):
        batch = prepared[offset : offset + batch_size]
        padded = tokenizer.pad(
            {"input_ids": [item[1] for item in batch]},
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **padded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_width = padded["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        results.extend(
            classify_output(row, output.strip(), seen_group_ids)
            for (row, _), output in zip(batch, outputs)
        )
        print(json.dumps({"generated": len(results), "total": len(rows)}, sort_keys=True), flush=True)
    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_seen_group_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("selected_new_trajectory_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"causal SFT report has no selected_new_trajectory_ids: {path}"
        )
    ids = {str(value) for value in values}
    if len(ids) != len(values):
        raise ValueError("causal SFT report contains duplicate selected trajectory IDs")
    return ids


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory exists: {args.output_dir}")
    rows = [
        row
        for row in read_jsonl(args.preferences)
        if row.get("agent_name") == "ReasonerAgent" and row.get("split") in set(args.splits)
    ]
    if not rows:
        raise ValueError("no Reasoner preference rows selected")
    seen_group_ids = read_seen_group_ids(args.causal_sft_report)
    args.output_dir.mkdir(parents=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    results = generate(
        model,
        tokenizer,
        rows,
        args.batch_size,
        args.max_new_tokens,
        device,
        seen_group_ids,
    )
    report = {
        "schema_version": 1,
        "model": args.model,
        "preferences": str(args.preferences),
        "splits": list(args.splits),
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "overall": summarize(results),
        "by_split": grouped_summary(results, "split"),
        "by_variant": grouped_summary(results, "variant"),
        "by_operation": grouped_summary(results, "operation"),
    }
    if seen_group_ids is not None:
        report["causal_sft_report"] = str(args.causal_sft_report)
        report["causal_sft_selected_group_count"] = len(seen_group_ids)
        report["by_causal_sft_exposure"] = grouped_summary(
            results, "causal_sft_exposure"
        )
        report["by_causal_sft_exposure_and_operation"] = nested_grouped_summary(
            results, "causal_sft_exposure", "operation"
        )
        report["by_causal_sft_exposure_and_variant"] = nested_grouped_summary(
            results, "causal_sft_exposure", "variant"
        )
    write_jsonl(args.output_dir / "generations.jsonl", results)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
