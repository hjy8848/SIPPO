#!/usr/bin/env python3
"""Score DAGENT chosen/rejected actions under a frozen causal LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


GROUP_FIELDS = (
    "split",
    "agent_name",
    "variant",
    "downstream_relation",
    "operation",
    "reasoning_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Frozen Hugging Face model or checkpoint")
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25, help="Log every N pair batches")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_token_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode_pair(tokenizer, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    prompt_ids = prompt_token_ids(tokenizer, row["prompt"])
    encoded = {"row": row, "prompt_ids": prompt_ids}
    for side in ("chosen", "rejected"):
        response_ids = tokenizer(str(row[side]), add_special_tokens=False)["input_ids"]
        if not response_ids:
            raise ValueError(f"{row['pair_id']}: {side} has no tokens")
        total_length = len(prompt_ids) + len(response_ids)
        if total_length > max_length:
            raise ValueError(
                f"{row['pair_id']}: {side} length {total_length} exceeds max_length={max_length}"
            )
        encoded[f"{side}_ids"] = list(response_ids)
        encoded[f"{side}_total_length"] = total_length
    encoded["sort_length"] = max(encoded["chosen_total_length"], encoded["rejected_total_length"])
    return encoded


def flatten_batch(encoded_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequences = []
    for pair in encoded_pairs:
        for side in ("chosen", "rejected"):
            response_ids = pair[f"{side}_ids"]
            sequences.append(
                {
                    "pair_id": pair["row"]["pair_id"],
                    "side": side,
                    "input_ids": pair["prompt_ids"] + response_ids,
                    "prompt_length": len(pair["prompt_ids"]),
                    "response_length": len(response_ids),
                }
            )
    return sequences


@torch.inference_mode()
def score_sequences(model, tokenizer, sequences: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    max_length = max(len(item["input_ids"]) for item in sequences)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
    labels = torch.full((len(sequences), max_length), -100, dtype=torch.long)
    for index, item in enumerate(sequences):
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        length = ids.numel()
        input_ids[index, :length] = ids
        attention_mask[index, :length] = 1
        start = item["prompt_length"]
        labels[index, start:length] = ids[start:length]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_nll = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, reduction="none", ignore_index=-100
    )
    mask = shifted_labels.ne(-100)
    token_logp_sum = -(token_nll * mask).sum(dim=1)
    token_count = mask.sum(dim=1)
    if not torch.equal(token_count.cpu(), torch.tensor([x["response_length"] for x in sequences])):
        raise RuntimeError("Response/logit alignment failed")

    results = []
    for index, item in enumerate(sequences):
        count = int(token_count[index].item())
        logp_sum = float(token_logp_sum[index].item())
        results.append(
            {
                "pair_id": item["pair_id"],
                "side": item["side"],
                "token_count": count,
                "logp_sum": logp_sum,
                "logp_mean": logp_sum / count,
            }
        )
    return results


def length_bucket(delta: int) -> str:
    if delta <= -33:
        return "chosen_33+_tokens_shorter"
    if delta <= -9:
        return "chosen_9-32_tokens_shorter"
    if delta <= -1:
        return "chosen_1-8_tokens_shorter"
    if delta == 0:
        return "equal_length"
    if delta <= 8:
        return "chosen_1-8_tokens_longer"
    if delta <= 32:
        return "chosen_9-32_tokens_longer"
    return "chosen_33+_tokens_longer"


def combine_pair(row: dict[str, Any], side_scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    chosen = side_scores["chosen"]
    rejected = side_scores["rejected"]
    token_delta = chosen["token_count"] - rejected["token_count"]
    result = {
        key: row.get(key)
        for key in (
            "pair_id",
            "group_id",
            "split",
            "agent_name",
            "variant",
            "downstream_relation",
            "operation",
            "reasoning_mode",
        )
    }
    result.update(
        {
            "chosen_token_count": chosen["token_count"],
            "rejected_token_count": rejected["token_count"],
            "token_count_delta": token_delta,
            "length_bucket": length_bucket(token_delta),
            "chosen_logp_sum": chosen["logp_sum"],
            "rejected_logp_sum": rejected["logp_sum"],
            "sum_margin": chosen["logp_sum"] - rejected["logp_sum"],
            "chosen_logp_mean": chosen["logp_mean"],
            "rejected_logp_mean": rejected["logp_mean"],
            "mean_margin": chosen["logp_mean"] - rejected["logp_mean"],
        }
    )
    return result


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else math.nan


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    sum_margins = [float(row["sum_margin"]) for row in rows]
    mean_margins = [float(row["mean_margin"]) for row in rows]
    chosen_lengths = [int(row["chosen_token_count"]) for row in rows]
    rejected_lengths = [int(row["rejected_token_count"]) for row in rows]
    return {
        "count": len(rows),
        "sum_pair_accuracy": sum(value > 0 for value in sum_margins) / len(rows),
        "sum_tie_rate": sum(value == 0 for value in sum_margins) / len(rows),
        "sum_margin_mean": sum(sum_margins) / len(rows),
        "sum_margin_median": median(sum_margins),
        "mean_pair_accuracy": sum(value > 0 for value in mean_margins) / len(rows),
        "mean_tie_rate": sum(value == 0 for value in mean_margins) / len(rows),
        "mean_margin_mean": sum(mean_margins) / len(rows),
        "mean_margin_median": median(mean_margins),
        "chosen_token_count_mean": sum(chosen_lengths) / len(rows),
        "chosen_token_count_median": median(chosen_lengths),
        "rejected_token_count_mean": sum(rejected_lengths) / len(rows),
        "rejected_token_count_median": median(rejected_lengths),
    }


def grouped_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    return {key: summarize(grouped[key]) for key in sorted(grouped)}


def build_report(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    by_field = {field: grouped_summary(rows, field) for field in GROUP_FIELDS}
    by_field["length_bucket"] = grouped_summary(rows, "length_bucket")
    reason_rows = [row for row in rows if row["agent_name"] == "ReasonerAgent"]
    return {
        "schema_version": 1,
        "model": args.model,
        "preferences": str(args.preferences),
        "preferences_sha256": sha256(args.preferences),
        "scored_pairs": len(rows),
        "elapsed_seconds": elapsed,
        "primary_dpo_metric": "sum_margin",
        "length_normalized_diagnostic": "mean_margin",
        "response_eos_appended": False,
        "overall": summarize(rows),
        "by": by_field,
        "reasoner_factual_vs_flipped": grouped_summary(reason_rows, "variant"),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    rows = read_jsonl(args.preferences)
    if not rows:
        raise ValueError("Preference file is empty")
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Preference file contains duplicate pair IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "scores.partial.jsonl"
    if partial_path.exists() and not args.resume:
        raise FileExistsError(f"{partial_path} exists; pass --resume or use a new output directory")
    completed: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        for result in read_jsonl(partial_path):
            completed[str(result["pair_id"])] = result

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    pending = [encode_pair(tokenizer, row, args.max_length) for row in rows if row["pair_id"] not in completed]
    pending.sort(key=lambda item: item["sort_length"])
    started = time.time()
    with partial_path.open("a", encoding="utf-8") as output:
        for batch_index, offset in enumerate(range(0, len(pending), args.batch_size), start=1):
            pair_batch = pending[offset : offset + args.batch_size]
            sequence_scores = score_sequences(model, tokenizer, flatten_batch(pair_batch), device)
            by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for score in sequence_scores:
                by_pair[score["pair_id"]][score["side"]] = score
            for pair in pair_batch:
                result = combine_pair(pair["row"], by_pair[pair["row"]["pair_id"]])
                output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                completed[result["pair_id"]] = result
            output.flush()
            if batch_index % args.log_every == 0 or offset + len(pair_batch) == len(pending):
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "scored": len(completed),
                            "total": len(rows),
                            "elapsed_seconds": round(elapsed, 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if set(completed) != set(pair_ids):
        missing = sorted(set(pair_ids) - set(completed))
        raise RuntimeError(f"Missing {len(missing)} pair scores: {missing[:5]}")
    ordered = [completed[pair_id] for pair_id in pair_ids]
    if not all(math.isfinite(row[key]) for row in ordered for key in ("sum_margin", "mean_margin")):
        raise RuntimeError("Non-finite preference margin detected")

    scores_path = args.output_dir / "scores.jsonl"
    temporary = scores_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(scores_path)
    report = build_report(ordered, args, time.time() - started)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report["overall"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
