#!/usr/bin/env python3
"""LoRA hindsight policy distillation for DAGENT planner outputs.

The frozen base policy receives sibling plan outcomes as hindsight, while the
student LoRA policy receives the original planner prompt. Both score the same
on-policy planner response. Distillation is applied to full-vocabulary token
distributions, not to scalar advantages or chosen-token probabilities alone.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW


SEARCH_CONTENT_RE = re.compile(r"<search\s+id=[\"']\d+[\"']\s*>(.*?)</search>", re.IGNORECASE | re.DOTALL)
NODE_RE = re.compile(
    r"<(search|reason)\s+id=[\"'](\d+)[\"']\s*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-samples", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--divergence", choices=("jsd", "forward_kl"), default="jsd")
    parser.add_argument("--reason-repeat", type=int, default=2)
    parser.add_argument(
        "--reason-distill-response",
        choices=("failed", "positive_reference"),
        default="failed",
        help="Response prefix used for Reason examples; positive_reference exposes the successful structural branch.",
    )
    parser.add_argument("--reason-preference-weight", type=float, default=0.0)
    parser.add_argument("--preference-beta", type=float, default=2.0)
    parser.add_argument("--reason-structure-ce-weight", type=float, default=0.0)
    parser.add_argument("--reason-positive-plan-ce-weight", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
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


def response_token_mask(tokenizer, response: str, objective: str) -> tuple[list[int], torch.Tensor]:
    encoding = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids = list(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    if objective == "planner":
        return response_ids, torch.ones(len(response_ids), dtype=torch.bool)
    spans = [match.span(1) for match in SEARCH_CONTENT_RE.finditer(response)]
    mask = []
    for start, end in offsets:
        mask.append(any(start < span_end and end > span_start for span_start, span_end in spans))
    return response_ids, torch.tensor(mask, dtype=torch.bool)


def reason_structure_token_mask(tokenizer, response: str) -> tuple[list[int], torch.Tensor]:
    encoding = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    response_ids = list(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    nodes = [
        (match.group(1).lower(), int(match.group(2)), match.group(3), match.span())
        for match in NODE_RE.finditer(response)
    ]
    spans: list[tuple[int, int]] = []
    for index, (kind, reason_id, _, reason_span) in enumerate(nodes):
        if kind != "reason":
            continue
        downstream = [
            span
            for later_kind, _, body, span in nodes[index + 1 :]
            if later_kind == "search" and f"#{reason_id}#" in body
        ]
        if downstream:
            spans.append(reason_span)
            spans.extend(downstream)
    mask = [
        any(start < span_end and end > span_start for span_start, span_end in spans)
        for start, end in offsets
    ]
    return response_ids, torch.tensor(mask, dtype=torch.bool)


def prompt_token_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def prepare_input(
    tokenizer,
    messages: list[dict[str, str]],
    response_ids: list[int],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, int] | None:
    prompt_ids = prompt_token_ids(tokenizer, messages)
    input_ids = prompt_ids + response_ids
    if not prompt_ids or not response_ids or len(input_ids) > max_length:
        return None
    return torch.tensor([input_ids], dtype=torch.long, device=device), len(prompt_ids)


def selected_response_logits(model, input_ids: torch.Tensor, prompt_length: int, mask: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=input_ids, use_cache=False)
    start = prompt_length - 1
    end = start + mask.numel()
    logits = output.logits[0, start:end]
    if logits.shape[0] != mask.numel():
        raise RuntimeError(f"Response/logit alignment failed: {logits.shape[0]} != {mask.numel()}")
    return logits[mask.to(logits.device)]


def divergence_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, kind: str, temperature: float) -> torch.Tensor:
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if kind == "forward_kl":
        loss = torch.sum(teacher_logp.exp() * (teacher_logp - student_logp), dim=-1)
    else:
        midpoint_logp = torch.logaddexp(student_logp, teacher_logp) - math.log(2.0)
        student_kl = torch.sum(student_logp.exp() * (student_logp - midpoint_logp), dim=-1)
        teacher_kl = torch.sum(teacher_logp.exp() * (teacher_logp - midpoint_logp), dim=-1)
        loss = 0.5 * (student_kl + teacher_kl)
    return loss.mean() * (temperature**2)


def chosen_logprob(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    logp = F.log_softmax(logits.float(), dim=-1)
    return float(logp.gather(-1, target_ids[:, None]).mean().detach().cpu())


def sequence_mean_logprob(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    response: str,
    max_length: int,
    require_grad: bool,
    disable: bool,
) -> torch.Tensor | None:
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if not response_ids:
        return None
    device = next(model.parameters()).device
    prepared = prepare_input(tokenizer, messages, response_ids, max_length, device)
    if prepared is None:
        return None
    mask = torch.ones(len(response_ids), dtype=torch.bool, device=device)
    context = disabled_adapter(model) if disable else contextlib.nullcontext()
    grad_context = contextlib.nullcontext() if require_grad else torch.no_grad()
    with context, grad_context:
        logits = selected_response_logits(model, *prepared, mask)
        targets = torch.tensor(response_ids, dtype=torch.long, device=device)
        return F.log_softmax(logits.float(), dim=-1).gather(-1, targets[:, None]).mean()


def reason_preference_loss(
    model,
    tokenizer,
    row: dict[str, Any],
    args: argparse.Namespace,
    require_grad: bool,
) -> dict[str, torch.Tensor] | None:
    messages = row["student_messages"]
    positive = str(row["positive_reference_response"])
    negative = str(row["response"])
    student_positive = sequence_mean_logprob(
        model, tokenizer, messages, positive, args.max_length, require_grad, disable=False
    )
    student_negative = sequence_mean_logprob(
        model, tokenizer, messages, negative, args.max_length, require_grad, disable=False
    )
    reference_positive = sequence_mean_logprob(
        model, tokenizer, messages, positive, args.max_length, False, disable=True
    )
    reference_negative = sequence_mean_logprob(
        model, tokenizer, messages, negative, args.max_length, False, disable=True
    )
    values = (student_positive, student_negative, reference_positive, reference_negative)
    if any(value is None for value in values):
        return None
    student_margin = student_positive - student_negative
    reference_margin = reference_positive - reference_negative
    relative_margin = student_margin - reference_margin
    return {
        "loss": -F.logsigmoid(args.preference_beta * relative_margin),
        "student_margin": student_margin,
        "reference_margin": reference_margin,
        "relative_margin": relative_margin,
    }


def reason_structure_imitation_loss(
    model,
    tokenizer,
    row: dict[str, Any],
    args: argparse.Namespace,
    require_grad: bool,
) -> torch.Tensor | None:
    response = str(row["positive_reference_response"])
    response_ids, mask = reason_structure_token_mask(tokenizer, response)
    if not response_ids or not mask.any():
        return None
    device = next(model.parameters()).device
    prepared = prepare_input(
        tokenizer, row["student_messages"], response_ids, args.max_length, device
    )
    if prepared is None:
        return None
    mask = mask.to(device)
    grad_context = contextlib.nullcontext() if require_grad else torch.no_grad()
    with grad_context:
        logits = selected_response_logits(model, *prepared, mask)
        targets = torch.tensor(response_ids, dtype=torch.long, device=device)[mask]
        return F.cross_entropy(logits.float(), targets)


def positive_plan_imitation_loss(
    model,
    tokenizer,
    row: dict[str, Any],
    args: argparse.Namespace,
    require_grad: bool,
) -> torch.Tensor | None:
    mean_logprob = sequence_mean_logprob(
        model,
        tokenizer,
        row["student_messages"],
        str(row["positive_reference_response"]),
        args.max_length,
        require_grad,
        disable=False,
    )
    return None if mean_logprob is None else -mean_logprob


def branch_preference(
    model,
    tokenizer,
    row: dict[str, Any],
    teacher_key: str,
    reference_key: str,
    args: argparse.Namespace,
) -> dict[str, float] | None:
    failed_ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
    reference_ids = tokenizer(str(row[reference_key]), add_special_tokens=False)["input_ids"]
    common = 0
    limit = min(len(failed_ids), len(reference_ids))
    while common < limit and failed_ids[common] == reference_ids[common]:
        common += 1
    if common >= limit:
        return None
    failed_token = failed_ids[common]
    reference_token = reference_ids[common]
    device = next(model.parameters()).device

    def next_logits(messages: list[dict[str, str]], disable: bool) -> torch.Tensor | None:
        prompt_ids = prompt_token_ids(tokenizer, messages)
        input_ids = prompt_ids + failed_ids[:common]
        if not input_ids or len(input_ids) > args.max_length:
            return None
        tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        context = disabled_adapter(model) if disable else contextlib.nullcontext()
        with torch.no_grad(), context:
            return model(input_ids=tensor, use_cache=False).logits[0, -1].float()

    student_logits = next_logits(row["student_messages"], disable=False)
    teacher_logits = next_logits(row[teacher_key], disable=True)
    if student_logits is None or teacher_logits is None:
        return None
    student_odds = float((student_logits[reference_token] - student_logits[failed_token]).cpu())
    teacher_odds = float((teacher_logits[reference_token] - teacher_logits[failed_token]).cpu())
    return {
        "common_prefix_tokens": float(common),
        "student_reference_vs_failed_logit": student_odds,
        "teacher_reference_vs_failed_logit": teacher_odds,
        "teacher_branch_gain": teacher_odds - student_odds,
    }


def disabled_adapter(model):
    method = getattr(model, "disable_adapter", None)
    return method() if method is not None else contextlib.nullcontext()


def distillation_response(row: dict[str, Any], args: argparse.Namespace) -> str:
    if (
        row.get("source_kind") == "reason"
        and args.reason_distill_response == "positive_reference"
    ):
        return str(row["positive_reference_response"])
    return str(row["response"])


def score_row(
    model,
    tokenizer,
    row: dict[str, Any],
    teacher_key: str,
    args: argparse.Namespace,
    require_grad: bool,
) -> dict[str, Any] | None:
    response = distillation_response(row, args)
    response_ids, response_mask = response_token_mask(tokenizer, response, row["objective"])
    if not response_ids or not response_mask.any():
        return None
    device = next(model.parameters()).device
    student_input = prepare_input(
        tokenizer,
        row["student_messages"],
        response_ids,
        args.max_length,
        device,
    )
    teacher_input = prepare_input(
        tokenizer,
        row[teacher_key],
        response_ids,
        args.max_length,
        device,
    )
    if student_input is None or teacher_input is None:
        return None
    response_mask = response_mask.to(device)
    target_ids = torch.tensor(response_ids, dtype=torch.long, device=device)[response_mask]

    with torch.no_grad(), disabled_adapter(model):
        teacher_logits = selected_response_logits(model, *teacher_input, response_mask).detach()
    if require_grad:
        student_logits = selected_response_logits(model, *student_input, response_mask)
    else:
        with torch.no_grad():
            student_logits = selected_response_logits(model, *student_input, response_mask)
    distill_loss = divergence_loss(student_logits, teacher_logits, args.divergence, args.temperature)
    loss = distill_loss
    preference = None
    if (
        teacher_key == "teacher_messages"
        and row.get("source_kind") == "reason"
        and args.reason_preference_weight > 0
    ):
        preference = reason_preference_loss(model, tokenizer, row, args, require_grad)
        if preference is None:
            return None
        loss = loss + args.reason_preference_weight * preference["loss"]
    structure_ce = None
    if (
        teacher_key == "teacher_messages"
        and row.get("source_kind") == "reason"
        and args.reason_structure_ce_weight > 0
    ):
        structure_ce = reason_structure_imitation_loss(
            model, tokenizer, row, args, require_grad
        )
        if structure_ce is None:
            return None
        loss = loss + args.reason_structure_ce_weight * structure_ce
    positive_plan_ce = None
    if (
        teacher_key == "teacher_messages"
        and row.get("source_kind") == "reason"
        and args.reason_positive_plan_ce_weight > 0
    ):
        positive_plan_ce = positive_plan_imitation_loss(
            model, tokenizer, row, args, require_grad
        )
        if positive_plan_ce is None:
            return None
        loss = loss + args.reason_positive_plan_ce_weight * positive_plan_ce
    result = {
        "loss": loss,
        "distill_loss": distill_loss,
        "token_count": int(response_mask.sum().item()),
        "distillation_response": (
            "positive_reference"
            if response == str(row.get("positive_reference_response", ""))
            else "failed"
        ),
        "student_chosen_logp": chosen_logprob(student_logits.detach(), target_ids),
        "teacher_chosen_logp": chosen_logprob(teacher_logits, target_ids),
        "teacher_logits": teacher_logits,
    }
    if preference is not None:
        result.update(
            {
                "preference_loss": preference["loss"],
                "preference_relative_margin": preference["relative_margin"],
            }
        )
    if structure_ce is not None:
        result["structure_ce_loss"] = structure_ce
    if positive_plan_ce is not None:
        result["positive_plan_ce_loss"] = positive_plan_ce
    return result


def run_audit(model, tokenizer, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    selected = list(rows)
    random.Random(args.seed).shuffle(selected)
    selected = selected[: args.audit_samples]
    metrics: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    model.eval()
    for row in selected:
        true_score = score_row(model, tokenizer, row, "teacher_messages", args, require_grad=False)
        control_key = (
            "control_teacher_messages"
            if "control_teacher_messages" in row
            else "shuffled_teacher_messages"
        )
        shuffled_score = score_row(model, tokenizer, row, control_key, args, require_grad=False)
        if true_score is None or shuffled_score is None:
            counts["skipped"] += 1
            continue
        teacher_sensitivity = divergence_loss(
            true_score["teacher_logits"],
            shuffled_score["teacher_logits"],
            "jsd",
            1.0,
        )
        true_devaluation = true_score["student_chosen_logp"] - true_score["teacher_chosen_logp"]
        shuffled_devaluation = shuffled_score["student_chosen_logp"] - shuffled_score["teacher_chosen_logp"]
        true_branch = branch_preference(
            model,
            tokenizer,
            row,
            "teacher_messages",
            "positive_reference_response",
            args,
        )
        control_branch = branch_preference(
            model,
            tokenizer,
            row,
            control_key,
            "positive_reference_response",
            args,
        )
        metrics["true_divergence"].append(float(true_score["loss"].detach().cpu()))
        metrics["true_distill_divergence"].append(
            float(true_score["distill_loss"].detach().cpu())
        )
        metrics["shuffled_divergence"].append(float(shuffled_score["loss"].detach().cpu()))
        metrics["teacher_label_sensitivity_jsd"].append(float(teacher_sensitivity.detach().cpu()))
        metrics["true_teacher_devalues_failed_response"].append(true_devaluation)
        metrics["shuffled_teacher_devalues_failed_response"].append(shuffled_devaluation)
        metrics["devaluation_margin_over_shuffled"].append(true_devaluation - shuffled_devaluation)
        metrics["token_count"].append(float(true_score["token_count"]))
        if "preference_loss" in true_score:
            metrics["reason_preference_loss"].append(
                float(true_score["preference_loss"].detach().cpu())
            )
            metrics["reason_preference_relative_margin"].append(
                float(true_score["preference_relative_margin"].detach().cpu())
            )
        if "structure_ce_loss" in true_score:
            metrics["reason_structure_ce_loss"].append(
                float(true_score["structure_ce_loss"].detach().cpu())
            )
        if "positive_plan_ce_loss" in true_score:
            metrics["reason_positive_plan_ce_loss"].append(
                float(true_score["positive_plan_ce_loss"].detach().cpu())
            )
        if true_branch is not None and control_branch is not None:
            metrics["true_success_branch_gain"].append(true_branch["teacher_branch_gain"])
            metrics["control_success_branch_gain"].append(control_branch["teacher_branch_gain"])
            branch_margin = true_branch["teacher_branch_gain"] - control_branch["teacher_branch_gain"]
            metrics["success_branch_margin_over_control"].append(branch_margin)
            metrics["branch_common_prefix_tokens"].append(true_branch["common_prefix_tokens"])
            metrics[f"source/{row['source_kind']}/success_branch_gain"].append(
                true_branch["teacher_branch_gain"]
            )
            metrics[f"source/{row['source_kind']}/branch_margin"].append(branch_margin)
        counts[f"source/{row['source_kind']}"] += 1
        counts[f"objective/{row['objective']}"] += 1
        counts[f"distillation_response/{true_score['distillation_response']}"] += 1
        del true_score["teacher_logits"], shuffled_score["teacher_logits"]
        torch.cuda.empty_cache()
    summary = {
        "rows_requested": len(selected),
        "rows_scored": len(metrics.get("true_divergence", [])),
        "counts": counts,
        "means": {key: sum(values) / len(values) for key, values in metrics.items() if values},
        "interpretation": {
            "true_divergence": "Must be non-zero; otherwise hindsight does not change the teacher policy.",
            "teacher_label_sensitivity_jsd": "Measures the distribution gap between a successful reference and a failed-reference control.",
            "devaluation_margin_over_shuffled": "Positive means the successful reference suppresses the sampled failed response more than the failed-reference control.",
            "true_success_branch_gain": "Positive means hindsight increases the successful token's log-odds at the first plan branch point.",
            "success_branch_margin_over_control": "Positive means a successful reference steers the first branch more than a failed-reference control.",
        },
    }
    return summary


def save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> None:
    path = output_dir / f"adapter_step_{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train(model, tokenizer, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    from transformers import get_cosine_schedule_with_warmup

    expanded = []
    for row in rows:
        repeat = args.reason_repeat if row.get("source_kind") == "reason" else 1
        expanded.extend([row] * max(1, repeat))
    if not expanded:
        raise RuntimeError("Training data is empty")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    updates_per_epoch = math.ceil(len(expanded) / args.gradient_accumulation)
    total_steps = min(args.max_steps, max(1, updates_per_epoch * args.epochs))
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    metrics_file = args.output_dir / "train_metrics.jsonl"
    rng = random.Random(args.seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    skipped = Counter()
    running_loss = 0.0
    started = time.time()
    with metrics_file.open("w", encoding="utf-8") as metric_handle:
        for epoch in range(args.epochs):
            rng.shuffle(expanded)
            for row in expanded:
                score = score_row(model, tokenizer, row, "teacher_messages", args, require_grad=True)
                if score is None:
                    skipped["unusable_or_too_long"] += 1
                    continue
                (score["loss"] / args.gradient_accumulation).backward()
                running_loss += float(score["loss"].detach().cpu())
                micro_step += 1
                del score["teacher_logits"]
                if micro_step % args.gradient_accumulation:
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": running_loss / args.gradient_accumulation,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.time() - started,
                }
                metric_handle.write(json.dumps(record) + "\n")
                metric_handle.flush()
                if global_step == 1 or global_step % args.log_every == 0:
                    print(json.dumps(record), flush=True)
                running_loss = 0.0
                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(model, tokenizer, args.output_dir, global_step)
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break
    return {
        "global_step": global_step,
        "total_steps_planned": total_steps,
        "micro_steps": micro_step,
        "skipped": skipped,
        "elapsed_seconds": time.time() - started,
    }


def main() -> int:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --overwrite or a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        ).cuda()
    except (ImportError, ValueError) as error:
        print(f"[WARN] FlashAttention unavailable ({error}); falling back to SDPA.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="sdpa",
            **model_kwargs,
        ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
    source_counts = Counter(row.get("source_kind", "unknown") for row in train_rows)
    print(f"[DATA] train={len(train_rows)} val={len(val_rows)} source_counts={dict(source_counts)}")

    audit = run_audit(model, tokenizer, train_rows, args)
    with (args.output_dir / "pretrain_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print("[AUDIT] " + json.dumps(audit, ensure_ascii=False), flush=True)
    if audit["rows_scored"] == 0 or audit["means"].get("true_divergence", 0.0) <= 0.0:
        raise RuntimeError("Hindsight teacher produced no measurable distribution shift")
    if args.audit_only:
        return 0

    train_summary = train(model, tokenizer, train_rows, args)
    save_checkpoint(model, tokenizer, args.output_dir, train_summary["global_step"])
    post_rows = val_rows or train_rows
    post_audit = run_audit(model, tokenizer, post_rows, args)
    with (args.output_dir / "posttrain_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(post_audit, handle, ensure_ascii=False, indent=2)

    merged_dir = args.output_dir / "merged_model"
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    summary = {
        "model": args.model,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file) if args.val_file else None,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "pretrain_audit": audit,
        "posttrain_audit": post_audit,
        "training": train_summary,
        "merged_model": str(merged_dir),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print("[DONE] " + json.dumps(train_summary, ensure_ascii=False), flush=True)
    print(f"[DONE] merged_model={merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
