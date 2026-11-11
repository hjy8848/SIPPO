#!/usr/bin/env python3
"""Run one balanced Reasoner-only DPO update and audit its causal effect."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from scripts.score_dagent_preference_baseline import (
        encode_pair,
        flatten_batch,
        read_jsonl,
        score_sequences,
        write_json,
    )
except ModuleNotFoundError:
    from score_dagent_preference_baseline import (
        encode_pair,
        flatten_batch,
        read_jsonl,
        score_sequences,
        write_json,
    )


VARIANTS = ("factual_condition", "flipped_condition")
OPERATIONS = ("closer_to_anchor", "earlier", "later")
CONCLUSION_RE = re.compile(r"^<conclusion>[^<>\n]+</conclusion>$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups-per-operation", type=int, default=8)
    parser.add_argument(
        "--update-operation-counts",
        type=int,
        nargs=3,
        metavar=("CLOSER", "EARLIER", "LATER"),
        help="Override update-group counts in closer/earlier/later order.",
    )
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--generation-groups-per-operation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--optimizer-offload",
        action="store_true",
        help="Move the model and gradients to CPU for the single AdamW step.",
    )
    return parser.parse_args()


def index_twins(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    twins: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("agent_name") != "ReasonerAgent" or row.get("variant") not in VARIANTS:
            continue
        twins[str(row["group_id"])][str(row["variant"])] = row
    incomplete = [group_id for group_id, pair in twins.items() if set(pair) != set(VARIANTS)]
    if incomplete:
        raise ValueError(f"Incomplete Reasoner twins: {incomplete[:5]}")
    return dict(twins)


def select_groups(
    twins: dict[str, dict[str, dict[str, Any]]],
    *,
    split: str | tuple[str, ...],
    groups_per_operation: int | dict[str, int],
    seed: int,
) -> list[str]:
    splits = {split} if isinstance(split, str) else set(split)
    by_operation: dict[str, list[str]] = defaultdict(list)
    for group_id, pair in twins.items():
        factual = pair["factual_condition"]
        if factual["split"] in splits:
            by_operation[str(factual["operation"])].append(group_id)
    selected = []
    for operation_index, operation in enumerate(OPERATIONS):
        requested = (
            groups_per_operation[operation]
            if isinstance(groups_per_operation, dict)
            else groups_per_operation
        )
        if requested < 0:
            raise ValueError(f"Negative group count for {operation}: {requested}")
        candidates = sorted(by_operation[operation])
        if len(candidates) < requested:
            raise ValueError(
                f"Need {requested} {operation} groups in {sorted(splits)}, found {len(candidates)}"
            )
        random.Random(seed + operation_index).shuffle(candidates)
        selected.extend(candidates[:requested])
    return selected


def rows_for_groups(
    twins: dict[str, dict[str, dict[str, Any]]], group_ids: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for group_id in group_ids:
        rows.extend(twins[group_id][variant] for variant in VARIANTS)
    return rows


def encode_rows(tokenizer, rows: list[dict[str, Any]], max_length: int) -> list[dict[str, Any]]:
    return [encode_pair(tokenizer, row, max_length) for row in rows]


def make_tensors(tokenizer, encoded_pairs: list[dict[str, Any]], device: torch.device):
    sequences = flatten_batch(encoded_pairs)
    max_length = max(len(sequence["input_ids"]) for sequence in sequences)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no padding or EOS token")
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, sequence in enumerate(sequences):
        ids = torch.tensor(sequence["input_ids"], dtype=torch.long)
        length = ids.numel()
        input_ids[index, :length] = ids
        attention_mask[index, :length] = 1
        labels[index, sequence["prompt_length"] : length] = ids[sequence["prompt_length"] :]
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def differentiable_margins(model, tokenizer, encoded_pairs, device, autocast_dtype):
    input_ids, attention_mask, labels = make_tensors(tokenizer, encoded_pairs, device)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_nll = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, reduction="none", ignore_index=-100
    )
    mask = shifted_labels.ne(-100)
    logp_sums = -(token_nll * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    expected = torch.tensor(
        [
            length
            for pair in encoded_pairs
            for length in (len(pair["chosen_ids"]), len(pair["rejected_ids"]))
        ],
        device=device,
    )
    if not torch.equal(counts, expected):
        raise RuntimeError("Response/logit alignment failed during DPO")
    margins = logp_sums[0::2] - logp_sums[1::2]
    return margins, logp_sums, counts


@torch.inference_mode()
def evaluate_pairs(model, tokenizer, encoded_pairs, batch_size, device, autocast_dtype):
    scored: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for offset in range(0, len(encoded_pairs), batch_size):
        batch = encoded_pairs[offset : offset + batch_size]
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            values = score_sequences(model, tokenizer, flatten_batch(batch), device)
        for value in values:
            scored[value["pair_id"]][value["side"]] = value
    results = []
    for pair in encoded_pairs:
        row = pair["row"]
        sides = scored[row["pair_id"]]
        results.append(
            {
                "pair_id": row["pair_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "variant": row["variant"],
                "operation": row["operation"],
                "chosen_token_count": sides["chosen"]["token_count"],
                "rejected_token_count": sides["rejected"]["token_count"],
                "sum_margin": sides["chosen"]["logp_sum"] - sides["rejected"]["logp_sum"],
                "mean_margin": sides["chosen"]["logp_mean"] - sides["rejected"]["logp_mean"],
            }
        )
    return results


def summarize(rows: list[dict[str, Any]], key: str = "sum_margin") -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "count": len(rows),
        "accuracy": sum(value > 0 for value in values) / len(values),
        "margin_mean": sum(values) / len(values),
        "margin_median": float(statistics.median(values)),
    }


def paired_summary(before, after, *, field=None, value=None):
    before_by_id = {row["pair_id"]: row for row in before}
    selected_after = [row for row in after if field is None or row.get(field) == value]
    selected_before = [before_by_id[row["pair_id"]] for row in selected_after]
    deltas = [
        new["sum_margin"] - old["sum_margin"] for old, new in zip(selected_before, selected_after)
    ]
    return {
        "before": summarize(selected_before),
        "after": summarize(selected_after),
        "margin_delta_mean": sum(deltas) / len(deltas),
        "margin_delta_median": float(statistics.median(deltas)),
        "fraction_margin_improved": sum(delta > 0 for delta in deltas) / len(deltas),
    }


def twin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    twins: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        twins[row["group_id"]][row["variant"]] = float(row["sum_margin"])
    if not all(set(pair) == set(VARIANTS) for pair in twins.values()):
        raise ValueError("Scored rows do not contain complete twins")
    both = sum(all(pair[variant] > 0 for variant in VARIANTS) for pair in twins.values())
    return {"groups": len(twins), "both_correct": both, "both_correct_rate": both / len(twins)}


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Pearson inputs must have equal nonzero length")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def length_audit(before, after) -> dict[str, Any]:
    before_by_id = {row["pair_id"]: row for row in before}
    token_deltas = [row["chosen_token_count"] - row["rejected_token_count"] for row in after]
    margin_deltas = [row["sum_margin"] - before_by_id[row["pair_id"]]["sum_margin"] for row in after]
    return {
        "chosen_minus_rejected_token_total": sum(token_deltas),
        "margin_delta_vs_token_delta_pearson": pearson(token_deltas, margin_deltas),
    }


@torch.inference_mode()
def generate_probe(model, tokenizer, rows, device, autocast_dtype):
    outputs = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        attention_mask = torch.ones_like(prompt_ids)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=48,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0, prompt_ids.shape[1] :], skip_special_tokens=True).strip()
        outputs.append(
            {
                "pair_id": row["pair_id"],
                "group_id": row["group_id"],
                "variant": row["variant"],
                "operation": row["operation"],
                "output": text,
                "format_valid": bool(CONCLUSION_RE.fullmatch(text)),
                "chosen_exact": text == str(row["chosen"]).strip(),
            }
        )
    return outputs


def generation_summary(rows):
    return {
        "count": len(rows),
        "format_valid_rate": sum(row["format_valid"] for row in rows) / len(rows),
        "chosen_exact_rate": sum(row["chosen_exact"] for row in rows) / len(rows),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sample_indices(numel: int, width: int, device: torch.device) -> torch.Tensor:
    if numel < 1 or width < 1:
        raise ValueError("Parameter probe dimensions must be positive")
    return torch.linspace(
        0,
        numel - 1,
        min(width, numel),
        dtype=torch.float64,
        device=device,
    ).long().clamp_(0, numel - 1)


def parameter_probe(model, width: int = 256) -> dict[str, torch.Tensor]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    indices = sorted({0, len(named) // 2, len(named) - 1})
    probes = {}
    for index in indices:
        name, parameter = named[index]
        flat = parameter.detach().view(-1)
        indices = sample_indices(flat.numel(), width, flat.device)
        probes[name] = flat[indices].float().cpu()
    return probes


def parameter_probe_delta(model, before: dict[str, torch.Tensor], width: int = 256):
    named = dict(model.named_parameters())
    squared = 0.0
    maximum = 0.0
    changed = 0
    for name, old in before.items():
        flat = named[name].detach().view(-1)
        indices = sample_indices(flat.numel(), width, flat.device)
        delta = flat[indices].float().cpu() - old
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        changed += int(bool(delta.ne(0).any()))
    return {"sampled_l2": math.sqrt(squared), "sampled_max_abs": maximum, "changed_tensors": changed}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    all_rows = read_jsonl(args.preferences)
    twins = index_twins(all_rows)
    update_counts = (
        dict(zip(OPERATIONS, args.update_operation_counts, strict=True))
        if args.update_operation_counts is not None
        else {operation: args.groups_per_operation for operation in OPERATIONS}
    )
    if sum(update_counts.values()) < 1:
        raise ValueError("At least one update group is required")
    update_group_ids = select_groups(
        twins,
        split="train",
        groups_per_operation=update_counts,
        seed=args.seed,
    )
    generation_group_ids = select_groups(
        twins,
        split=("validation", "test"),
        groups_per_operation=args.generation_groups_per_operation,
        seed=args.seed + 1000,
    )
    update_rows = rows_for_groups(twins, update_group_ids)
    generation_rows = rows_for_groups(twins, generation_group_ids)
    reasoner_rows = [
        row for row in all_rows if row.get("agent_name") == "ReasonerAgent" and row.get("variant") in VARIANTS
    ]
    write_jsonl(args.output_dir / "update_pairs.jsonl", update_rows)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    autocast_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()

    encoded_all = encode_rows(tokenizer, reasoner_rows, args.max_length)
    encoded_update = encode_rows(tokenizer, update_rows, args.max_length)
    started = time.time()
    before_scores = evaluate_pairs(
        model, tokenizer, encoded_all, args.eval_batch_size, device, autocast_dtype
    )
    before_generation = generate_probe(model, tokenizer, generation_rows, device, autocast_dtype)
    with torch.no_grad():
        reference_margins = []
        for offset in range(0, len(encoded_update), args.micro_batch_size):
            margins, _, _ = differentiable_margins(
                model,
                tokenizer,
                encoded_update[offset : offset + args.micro_batch_size],
                device,
                autocast_dtype,
            )
            reference_margins.append(margins.detach())
        reference_margins = torch.cat(reference_margins)

    probes_before = parameter_probe(model)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        fused=not args.optimizer_offload,
    )
    optimizer.zero_grad(set_to_none=True)
    micro_batches = math.ceil(len(encoded_update) / args.micro_batch_size)
    train_losses = []
    train_relative_margins = []
    for micro_index, offset in enumerate(range(0, len(encoded_update), args.micro_batch_size)):
        batch = encoded_update[offset : offset + args.micro_batch_size]
        policy_margin, _, _ = differentiable_margins(
            model, tokenizer, batch, device, autocast_dtype
        )
        reference_margin = reference_margins[offset : offset + len(batch)]
        relative_margin = policy_margin - reference_margin
        loss = -F.logsigmoid(args.beta * relative_margin).mean()
        (loss / micro_batches).backward()
        train_losses.append(float(loss.detach()))
        train_relative_margins.extend(relative_margin.detach().float().cpu().tolist())
        print(
            json.dumps(
                {
                    "micro_batch": micro_index + 1,
                    "micro_batches": micro_batches,
                    "loss": train_losses[-1],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(f"Invalid gradient norm: {grad_norm}")
    if args.optimizer_offload:
        model.to("cpu")
        torch.cuda.empty_cache()
    optimizer.step()
    parameter_delta = parameter_probe_delta(model, probes_before)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if args.optimizer_offload:
        model.to(device)

    model.eval()
    after_scores = evaluate_pairs(
        model, tokenizer, encoded_all, args.eval_batch_size, device, autocast_dtype
    )
    after_generation = generate_probe(model, tokenizer, generation_rows, device, autocast_dtype)
    after_by_id = {row["pair_id"]: row for row in after_scores}
    post_update_margins = torch.tensor(
        [after_by_id[row["pair_id"]]["sum_margin"] for row in update_rows], device=device
    )
    post_relative = post_update_margins - reference_margins
    post_loss = float((-F.logsigmoid(args.beta * post_relative)).mean())

    update_ids = {row["pair_id"] for row in update_rows}
    heldout_before = [row for row in before_scores if row["split"] in ("validation", "test")]
    heldout_after = [row for row in after_scores if row["split"] in ("validation", "test")]
    train_probe_before = [row for row in before_scores if row["split"] == "train" and row["pair_id"] not in update_ids]
    train_probe_after = [row for row in after_scores if row["split"] == "train" and row["pair_id"] not in update_ids]
    report = {
        "schema_version": 1,
        "model": args.model,
        "preferences": str(args.preferences),
        "method": "reference_relative_dpo",
        "optimizer_steps": 1,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "optimizer_offload": args.optimizer_offload,
        "update_groups": len(update_group_ids),
        "update_pairs": len(update_rows),
        "update_pairs_by_operation": {
            operation: sum(row["operation"] == operation for row in update_rows)
            for operation in OPERATIONS
        },
        "update_pairs_by_variant": {
            variant: sum(row["variant"] == variant for row in update_rows) for variant in VARIANTS
        },
        "pre_update_dpo_loss": math.log(2.0),
        "micro_batch_loss_mean": sum(train_losses) / len(train_losses),
        "post_update_dpo_loss": post_loss,
        "initial_relative_margin_abs_max": max(map(abs, train_relative_margins)),
        "grad_norm_before_clip": grad_norm,
        "parameter_probe_delta": parameter_delta,
        "all_reasoner": paired_summary(before_scores, after_scores),
        "heldout_reasoner": paired_summary(heldout_before, heldout_after),
        "train_nonupdate_reasoner": paired_summary(train_probe_before, train_probe_after),
        "heldout_by_variant": {
            variant: paired_summary(heldout_before, heldout_after, field="variant", value=variant)
            for variant in VARIANTS
        },
        "heldout_by_operation": {
            operation: paired_summary(heldout_before, heldout_after, field="operation", value=operation)
            for operation in OPERATIONS
        },
        "twins_before": twin_summary(before_scores),
        "twins_after": twin_summary(after_scores),
        "length_audit": length_audit(before_scores, after_scores),
        "generation_before": generation_summary(before_generation),
        "generation_after": generation_summary(after_generation),
        "elapsed_seconds": time.time() - started,
    }
    report["gate"] = {
        "loss_decreased": report["post_update_dpo_loss"] < report["pre_update_dpo_loss"],
        "parameters_changed": parameter_delta["sampled_max_abs"] > 0,
        "finite_nonzero_gradient": math.isfinite(grad_norm) and grad_norm > 0,
        "heldout_factual_margin_improved": report["heldout_by_variant"]["factual_condition"][
            "margin_delta_mean"
        ]
        > 0,
        "heldout_flipped_margin_improved": report["heldout_by_variant"]["flipped_condition"][
            "margin_delta_mean"
        ]
        > 0,
        "format_not_regressed": report["generation_after"]["format_valid_rate"]
        >= report["generation_before"]["format_valid_rate"],
        "balanced_twin_token_total": sum(
            row["chosen_token_count"] - row["rejected_token_count"]
            for row in before_scores
            if row["pair_id"] in update_ids
        )
        == 0,
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_jsonl(args.output_dir / "scores_before.jsonl", before_scores)
    write_jsonl(args.output_dir / "scores_after.jsonl", after_scores)
    write_jsonl(args.output_dir / "generation_before.jsonl", before_generation)
    write_jsonl(args.output_dir / "generation_after.jsonl", after_generation)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
