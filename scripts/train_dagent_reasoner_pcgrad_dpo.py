#!/usr/bin/env python3
"""Train canonical DAGENT Reasoner preferences with operation-aware LoRA DPO."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from scripts.run_dagent_cwpo_one_batch import (
        OPERATIONS,
        VARIANTS,
        differentiable_margins,
        encode_rows,
        evaluate_pairs,
        generate_probe,
        generation_summary,
        index_twins,
        paired_summary,
        rows_for_groups,
        select_groups,
        twin_summary,
        write_jsonl,
    )
    from scripts.score_dagent_preference_baseline import read_jsonl, write_json
except ModuleNotFoundError:
    from run_dagent_cwpo_one_batch import (
        OPERATIONS,
        VARIANTS,
        differentiable_margins,
        encode_rows,
        evaluate_pairs,
        generate_probe,
        generation_summary,
        index_twins,
        paired_summary,
        rows_for_groups,
        select_groups,
        twin_summary,
        write_jsonl,
    )
    from score_dagent_preference_baseline import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--groups-per-operation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--chosen-nll-weight",
        type=float,
        default=0.0,
        help="Weight for token-mean NLL on chosen responses; preserves shared protocol tokens that cancel in DPO.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--generation-groups-per-operation", type=int, default=4)
    parser.add_argument("--gradient-strategy", choices=("mean", "pcgrad"), default="pcgrad")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


class GroupBatcher:
    def __init__(
        self,
        twins: dict[str, dict[str, dict[str, Any]]],
        operation: str,
        groups_per_batch: int,
        seed: int,
    ) -> None:
        self.twins = twins
        self.operation = operation
        self.groups_per_batch = groups_per_batch
        self.rng = random.Random(seed)
        self.groups = sorted(
            group_id
            for group_id, pair in twins.items()
            if pair["factual_condition"]["split"] == "train"
            and pair["factual_condition"]["operation"] == operation
        )
        if len(self.groups) < groups_per_batch:
            raise ValueError(
                f"Need {groups_per_batch} train groups for {operation}, found {len(self.groups)}"
            )
        self.order: list[str] = []
        self.cursor = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(self.groups)
        self.rng.shuffle(self.order)
        self.cursor = 0

    def next_rows(self) -> list[dict[str, Any]]:
        selected: list[str] = []
        while len(selected) < self.groups_per_batch:
            if self.cursor == len(self.order):
                self._reshuffle()
            take = min(self.groups_per_batch - len(selected), len(self.order) - self.cursor)
            selected.extend(self.order[self.cursor : self.cursor + take])
            self.cursor += take
        return rows_for_groups(self.twins, selected)


def gradient_dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
    return sum((a * b).sum() for a, b in zip(left, right, strict=True))


def gradient_norm_squared(gradient: list[torch.Tensor]) -> torch.Tensor:
    return gradient_dot(gradient, gradient)


def gradient_cosine(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    denominator = gradient_norm_squared(left).sqrt() * gradient_norm_squared(right).sqrt()
    if float(denominator) == 0.0:
        return 0.0
    return float((gradient_dot(left, right) / denominator).detach().cpu())


def combine_operation_gradients(
    gradients: dict[str, list[torch.Tensor]], strategy: str, seed: int
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    operations = [operation for operation in OPERATIONS if operation in gradients]
    raw_cosines = {
        f"{left}__{right}": gradient_cosine(gradients[left], gradients[right])
        for index, left in enumerate(operations)
        for right in operations[index + 1 :]
    }
    projected = {
        operation: [value.clone() for value in gradients[operation]] for operation in operations
    }
    projections = 0
    if strategy == "pcgrad":
        rng = random.Random(seed)
        for operation in operations:
            others = [candidate for candidate in operations if candidate != operation]
            rng.shuffle(others)
            for other in others:
                dot = gradient_dot(projected[operation], gradients[other])
                norm = gradient_norm_squared(gradients[other])
                if float(dot) < 0.0 and float(norm) > 0.0:
                    scale = dot / norm
                    projected[operation] = [
                        current - scale * reference
                        for current, reference in zip(
                            projected[operation], gradients[other], strict=True
                        )
                    ]
                    projections += 1
    combined = [
        sum(projected[operation][index] for operation in operations) / len(operations)
        for index in range(len(projected[operations[0]]))
    ]
    projected_cosines = {
        f"{left}__{right}": gradient_cosine(projected[left], projected[right])
        for index, left in enumerate(operations)
        for right in operations[index + 1 :]
    }
    return combined, {
        "raw_cosines": raw_cosines,
        "projected_cosines": projected_cosines,
        "projection_count": projections,
    }


def chosen_response_nll(
    logp_sums: torch.Tensor, token_counts: torch.Tensor
) -> torch.Tensor:
    if logp_sums.ndim != 1 or token_counts.shape != logp_sums.shape:
        raise ValueError("log-prob sums and token counts must be aligned vectors")
    if logp_sums.numel() == 0 or logp_sums.numel() % 2:
        raise ValueError("chosen/rejected log-prob vectors must contain complete pairs")
    chosen_counts = token_counts[0::2]
    if bool(chosen_counts.le(0).any()):
        raise ValueError("chosen responses must contain at least one scored token")
    return -(logp_sums[0::2] / chosen_counts).mean()


def operation_loss(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    max_length: int,
    beta: float,
    chosen_nll_weight: float,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = encode_rows(tokenizer, rows, max_length)
    adapter_context = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    with torch.no_grad(), adapter_context:
        reference_margin, _, _ = differentiable_margins(
            model, tokenizer, encoded, device, autocast_dtype
        )
    policy_margin, policy_logp_sums, policy_token_counts = differentiable_margins(
        model, tokenizer, encoded, device, autocast_dtype
    )
    relative_margin = policy_margin - reference_margin
    dpo_loss = -F.logsigmoid(beta * relative_margin).mean()
    chosen_nll = chosen_response_nll(policy_logp_sums, policy_token_counts)
    loss = dpo_loss + chosen_nll_weight * chosen_nll
    return loss, {
        "loss": float(loss.detach().cpu()),
        "dpo_loss": float(dpo_loss.detach().cpu()),
        "chosen_nll": float(chosen_nll.detach().cpu()),
        "chosen_nll_weight": chosen_nll_weight,
        "policy_margin_mean": float(policy_margin.detach().float().mean().cpu()),
        "reference_margin_mean": float(reference_margin.detach().float().mean().cpu()),
        "relative_margin_mean": float(relative_margin.detach().float().mean().cpu()),
    }


def evaluate_model(model, tokenizer, encoded_reasoner, generation_rows, args, device):
    model.eval()
    scores = evaluate_pairs(
        model,
        tokenizer,
        encoded_reasoner,
        args.eval_batch_size,
        device,
        torch.bfloat16,
    )
    generations = generate_probe(
        model, tokenizer, generation_rows, device, torch.bfloat16
    )
    return scores, generations


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    if args.max_steps < 1 or args.groups_per_operation < 1:
        raise ValueError("max_steps and groups_per_operation must be positive")
    if args.chosen_nll_weight < 0:
        raise ValueError("chosen_nll_weight must be non-negative")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    rows = read_jsonl(args.preferences)
    twins = index_twins(rows)
    reasoner_rows = [
        row
        for row in rows
        if row.get("agent_name") == "ReasonerAgent" and row.get("variant") in VARIANTS
    ]
    generation_group_ids = select_groups(
        twins,
        split=("validation", "test"),
        groups_per_operation=args.generation_groups_per_operation,
        seed=args.seed + 1000,
    )
    generation_rows = rows_for_groups(twins, generation_group_ids)
    write_jsonl(args.output_dir / "generation_probe_pairs.jsonl", generation_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA created no trainable parameters")

    encoded_reasoner = encode_rows(tokenizer, reasoner_rows, args.max_length)
    before_scores, before_generations = evaluate_model(
        model, tokenizer, encoded_reasoner, generation_rows, args, device
    )
    write_jsonl(args.output_dir / "scores_before.jsonl", before_scores)
    write_jsonl(args.output_dir / "generation_before.jsonl", before_generations)

    batchers = {
        operation: GroupBatcher(
            twins,
            operation,
            args.groups_per_operation,
            args.seed + operation_index * 1000,
        )
        for operation_index, operation in enumerate(OPERATIONS)
    }
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    started = time.time()
    with metrics_path.open("w", encoding="utf-8") as metric_handle:
        for step in range(1, args.max_steps + 1):
            model.train()
            operation_metrics: dict[str, dict[str, float]] = {}
            operation_gradients: dict[str, list[torch.Tensor]] = {}
            for operation in OPERATIONS:
                loss, values = operation_loss(
                    model,
                    tokenizer,
                    batchers[operation].next_rows(),
                    args.max_length,
                    args.beta,
                    args.chosen_nll_weight,
                    device,
                    torch.bfloat16,
                )
                grads = torch.autograd.grad(loss, trainable, allow_unused=True)
                operation_gradients[operation] = [
                    torch.zeros_like(parameter) if gradient is None else gradient.detach()
                    for parameter, gradient in zip(trainable, grads, strict=True)
                ]
                operation_metrics[operation] = values
            combined, conflict_metrics = combine_operation_gradients(
                operation_gradients, args.gradient_strategy, args.seed + step
            )
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(trainable, combined, strict=True):
                parameter.grad = gradient
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm))
            if not math.isfinite(grad_norm) or grad_norm <= 0.0:
                raise RuntimeError(f"Invalid gradient norm at step {step}: {grad_norm}")
            optimizer.step()
            record = {
                "step": step,
                "elapsed_seconds": time.time() - started,
                "gradient_strategy": args.gradient_strategy,
                "grad_norm_before_clip": grad_norm,
                "operations": operation_metrics,
                **conflict_metrics,
            }
            metric_handle.write(json.dumps(record, sort_keys=True) + "\n")
            metric_handle.flush()
            print(json.dumps(record, sort_keys=True), flush=True)

    after_scores, after_generations = evaluate_model(
        model, tokenizer, encoded_reasoner, generation_rows, args, device
    )
    write_jsonl(args.output_dir / "scores_after.jsonl", after_scores)
    write_jsonl(args.output_dir / "generation_after.jsonl", after_generations)
    heldout_before = [row for row in before_scores if row["split"] in ("validation", "test")]
    heldout_after = [row for row in after_scores if row["split"] in ("validation", "test")]
    report = {
        "schema_version": 1,
        "model": args.model,
        "preferences": str(args.preferences),
        "method": "operation_aware_lora_dpo",
        "arguments": vars(args) | {"output_dir": str(args.output_dir), "preferences": str(args.preferences)},
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "heldout_reasoner": paired_summary(heldout_before, heldout_after),
        "heldout_by_variant": {
            variant: paired_summary(
                heldout_before, heldout_after, field="variant", value=variant
            )
            for variant in VARIANTS
        },
        "heldout_by_operation": {
            operation: paired_summary(
                heldout_before, heldout_after, field="operation", value=operation
            )
            for operation in OPERATIONS
        },
        "twins_before": twin_summary(before_scores),
        "twins_after": twin_summary(after_scores),
        "generation_before": generation_summary(before_generations),
        "generation_after": generation_summary(after_generations),
    }
    report["gate"] = {
        "heldout_margin_improved": report["heldout_reasoner"]["margin_delta_mean"] > 0,
        "all_variants_improved": all(
            value["margin_delta_mean"] > 0 for value in report["heldout_by_variant"].values()
        ),
        "all_operations_improved": all(
            value["margin_delta_mean"] > 0 for value in report["heldout_by_operation"].values()
        ),
        "format_not_regressed": report["generation_after"]["format_valid_rate"]
        >= report["generation_before"]["format_valid_rate"],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(args.output_dir / "report.json", report)

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    merged_dir = args.output_dir / "merged_model"
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(json.dumps(report, sort_keys=True), flush=True)
    del merged, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
