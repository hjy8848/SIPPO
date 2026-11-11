#!/usr/bin/env python3
"""Train one full-vocabulary OPSD round for DAGENT.

The frozen stage-entry checkpoint acts as the privileged teacher with its LoRA
adapter disabled. The LoRA student receives the inference-time prompt. Both
score the same on-policy action, and only the student's full-vocabulary
forward KL receives gradients. Optional teacher-signal gates make the job fail
before optimization when privileged context does not support the target action.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel


PRIVILEGED_PREFIX = """Training-only privileged context follows. It contains a successful sibling workflow sampled by the same policy for the same question. Use it to reassess the current agent action, but respond only to the original prompt and never mention this reference.

"""

POLICY_ROLES = (
    "PlanningAgent",
    "QueryRewriteAgent",
    "AnswerGenerationAgent",
    "ReasonerAgent",
    "AnswerSummarizationAgent",
)


def distributed_info() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Number of complete sibling-pair workflows per optimizer update.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--pointwise-kl-clip",
        type=float,
        default=10.0,
        help="Search-E1 per-vocabulary forward-KL contribution cap; <=0 disables clipping.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--audit-samples", type=int, default=24)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run data, KL, and teacher-signal preflights without any optimizer step.",
    )
    parser.add_argument(
        "--teacher-gap-mode",
        choices=("none", "row", "token"),
        default="none",
        help=(
            "Restrict KL to positions where the privileged teacher assigns the "
            "target action more probability than the inference-time student."
        ),
    )
    parser.add_argument(
        "--min-teacher-chosen-gap",
        type=float,
        default=0.0,
        help="Minimum teacher-minus-student target-token log-probability gap.",
    )
    parser.add_argument(
        "--require-positive-teacher-signal",
        action="store_true",
        help="Abort before the first optimizer step when the teacher audit fails.",
    )
    parser.add_argument(
        "--min-audit-supported-mean-gap",
        type=float,
        default=0.01,
        help="Minimum mean chosen-token gap over positions selected by the configured gate.",
    )
    parser.add_argument(
        "--min-audit-supported-row-fraction",
        type=float,
        default=0.90,
        help="Minimum fraction of audited actions containing at least one selected token.",
    )
    parser.add_argument(
        "--min-audit-supported-token-fraction",
        type=float,
        default=0.50,
        help="Minimum fraction of audited action tokens selected for KL.",
    )
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in (
        "max_steps",
        "epochs",
        "gradient_accumulation",
        "max_length",
        "audit_samples",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    for name in (
        "min_audit_supported_row_fraction",
        "min_audit_supported_token_fraction",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


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
                raise ValueError(f"Expected an object at {path}:{line_number}")
            required = {"group_id", "role", "student_messages", "student_response", "privileged_reference"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"Missing fields at {path}:{line_number}: {sorted(missing)}")
            rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    return [
        sorted(group, key=lambda row: (int(row.get("workflow_step", 0)), str(row["role"])))
        for _, group in sorted(grouped.items())
    ]


def prompt_token_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def add_privileged_reference(
    messages: list[dict[str, str]], reference: str
) -> list[dict[str, str]]:
    block = PRIVILEGED_PREFIX + reference
    output = [dict(message) for message in messages]
    for index, message in enumerate(output):
        if message.get("role") == "system":
            output[index] = {
                "role": "system",
                "content": str(message.get("content") or "") + "\n\n" + block,
            }
            return output
    return [{"role": "system", "content": block}, *output]


def truncate_reference_tokens(tokenizer, reference: str, budget: int) -> str:
    reference_ids = tokenizer(reference, add_special_tokens=False)["input_ids"]
    if len(reference_ids) <= budget:
        return reference
    if budget < 16:
        return ""
    marker = tokenizer("\n[REFERENCE_MIDDLE_TRUNCATED]\n", add_special_tokens=False)["input_ids"]
    content_budget = max(1, budget - len(marker))
    head_count = max(1, round(content_budget * 0.7))
    tail_count = max(0, content_budget - head_count)
    kept = reference_ids[:head_count] + marker + (reference_ids[-tail_count:] if tail_count else [])
    return tokenizer.decode(kept, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def fit_teacher_messages(
    tokenizer,
    messages: list[dict[str, str]],
    reference: str,
    response_length: int,
    max_length: int,
) -> list[dict[str, str]] | None:
    empty_messages = add_privileged_reference(messages, "")
    empty_length = len(prompt_token_ids(tokenizer, empty_messages))
    budget = max_length - response_length - empty_length
    if budget < 1:
        return None
    fitted_reference = truncate_reference_tokens(tokenizer, reference, budget)
    if not fitted_reference:
        return None
    teacher_messages = add_privileged_reference(messages, fitted_reference)
    while len(prompt_token_ids(tokenizer, teacher_messages)) + response_length > max_length:
        budget -= max(8, budget // 10)
        if budget < 16:
            return None
        fitted_reference = truncate_reference_tokens(tokenizer, reference, budget)
        teacher_messages = add_privileged_reference(messages, fitted_reference)
    return teacher_messages


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


def response_logits(model, prepared: tuple[torch.Tensor, int], response_length: int) -> torch.Tensor:
    input_ids, prompt_length = prepared
    try:
        output = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=response_length + 1,
        )
    except TypeError:
        output = model(input_ids=input_ids, use_cache=False)
    all_logits = output.logits[0]
    if all_logits.shape[0] <= response_length + 1:
        # Qwen computes only the final R+1 positions: P-1 ... P+R-1.
        logits = all_logits[:response_length]
    else:
        start = prompt_length - 1
        logits = all_logits[start : start + response_length]
    if logits.shape[0] != response_length:
        raise RuntimeError(f"Response/logit alignment failed: {logits.shape[0]} != {response_length}")
    return logits


def pointwise_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    pointwise_clip: float = 10.0,
) -> torch.Tensor:
    """Return one full-vocabulary forward-KL value per response token."""
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    contribution = teacher_logp.exp() * (teacher_logp - student_logp)
    if pointwise_clip > 0:
        contribution = contribution.clamp_max(pointwise_clip)
    return contribution.sum(dim=-1) * (temperature**2)


def disabled_adapter(model):
    method = getattr(unwrap_model(model), "disable_adapter", None)
    return method() if method is not None else contextlib.nullcontext()


def teacher_gap_mask(
    chosen_gaps: torch.Tensor,
    mode: str,
    minimum_gap: float,
) -> torch.Tensor:
    positive = chosen_gaps > minimum_gap
    if mode == "none":
        return torch.ones_like(positive)
    if mode == "row":
        return torch.full_like(positive, bool(chosen_gaps.mean() > minimum_gap))
    if mode == "token":
        return positive
    raise ValueError(f"Unknown teacher gap mode: {mode}")


def score_row(
    model,
    tokenizer,
    row: dict[str, Any],
    args: argparse.Namespace,
    require_grad: bool,
    apply_teacher_gap_gate: bool = True,
) -> dict[str, Any] | None:
    response_ids = tokenizer(str(row["student_response"]), add_special_tokens=False)["input_ids"]
    if not response_ids:
        return None
    messages = row["student_messages"]
    teacher_messages = fit_teacher_messages(
        tokenizer,
        messages,
        str(row["privileged_reference"]),
        len(response_ids),
        args.max_length,
    )
    if teacher_messages is None:
        return None
    device = next(model.parameters()).device
    student_input = prepare_input(tokenizer, messages, response_ids, args.max_length, device)
    teacher_input = prepare_input(tokenizer, teacher_messages, response_ids, args.max_length, device)
    if student_input is None or teacher_input is None:
        return None

    with torch.no_grad(), disabled_adapter(model):
        teacher_logits = response_logits(model, teacher_input, len(response_ids)).detach()
    if require_grad:
        student_logits = response_logits(model, student_input, len(response_ids))
    else:
        with torch.no_grad():
            student_logits = response_logits(model, student_input, len(response_ids))
    token_losses = pointwise_forward_kl(
        student_logits,
        teacher_logits,
        args.temperature,
        args.pointwise_kl_clip,
    )
    target_ids = torch.tensor(response_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        student_logp = F.log_softmax(student_logits.float(), dim=-1)
        teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
        chosen_gaps = (
            teacher_logp.gather(-1, target_ids[:, None])
            - student_logp.gather(-1, target_ids[:, None])
        ).squeeze(-1)
        positive_gap_mask = chosen_gaps > args.min_teacher_chosen_gap
    eligible_mask = teacher_gap_mask(
        chosen_gaps,
        args.teacher_gap_mode if apply_teacher_gap_gate else "none",
        args.min_teacher_chosen_gap,
    )
    eligible_token_count = int(eligible_mask.sum().item())
    teacher_supported_mask = teacher_gap_mask(
        chosen_gaps,
        args.teacher_gap_mode,
        args.min_teacher_chosen_gap,
    )
    teacher_supported_token_count = int(teacher_supported_mask.sum().item())
    raw_token_count = len(response_ids)
    gated_loss_sum = (
        token_losses[eligible_mask].sum()
        if eligible_token_count
        else token_losses.sum() * 0.0
    )
    return {
        "loss_sum": gated_loss_sum,
        "raw_loss_sum": token_losses.sum().detach(),
        "loss_mean": token_losses.mean().detach(),
        "token_count": eligible_token_count,
        "raw_token_count": raw_token_count,
        "chosen_logp_gap": float(chosen_gaps.mean().cpu()),
        "chosen_logp_gap_sum": float(chosen_gaps.sum().cpu()),
        "positive_gap_token_count": int(positive_gap_mask.sum().item()),
        "teacher_supported_token_count": teacher_supported_token_count,
        "teacher_supported_gap_sum": float(
            chosen_gaps[teacher_supported_mask].sum().cpu()
            if teacher_supported_token_count
            else 0.0
        ),
        "teacher_supported_row": int(bool(teacher_supported_token_count)),
        "eligible_token_fraction": eligible_token_count / raw_token_count,
        "reference_truncated": len(prompt_token_ids(tokenizer, teacher_messages))
        + len(response_ids)
        >= args.max_length,
    }


def audit_teacher_signal(
    model,
    tokenizer,
    groups: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = [row for group in groups for row in group]
    random.Random(args.seed).shuffle(candidates)
    candidates = candidates[: args.audit_samples]
    totals = {
        "raw_kl_sum": 0.0,
        "gap_sum": 0.0,
        "tokens": 0,
        "positive_tokens": 0,
        "supported_tokens": 0,
        "supported_gap_sum": 0.0,
        "rows": 0,
        "positive_rows": 0,
        "supported_rows": 0,
        "skipped_rows": 0,
        "truncated_rows": 0,
    }
    model.eval()
    for row in candidates:
        score = score_row(
            model,
            tokenizer,
            row,
            args,
            require_grad=False,
            apply_teacher_gap_gate=False,
        )
        if score is None:
            totals["skipped_rows"] += 1
            continue
        totals["raw_kl_sum"] += float(score["raw_loss_sum"].cpu())
        totals["gap_sum"] += score["chosen_logp_gap_sum"]
        totals["tokens"] += score["raw_token_count"]
        totals["positive_tokens"] += score["positive_gap_token_count"]
        totals["supported_tokens"] += score["teacher_supported_token_count"]
        totals["supported_gap_sum"] += score["teacher_supported_gap_sum"]
        totals["rows"] += 1
        totals["positive_rows"] += int(
            score["chosen_logp_gap"] > args.min_teacher_chosen_gap
        )
        totals["supported_rows"] += score["teacher_supported_row"]
        totals["truncated_rows"] += int(score["reference_truncated"])
    if dist.is_initialized():
        packed = torch.tensor(
            [
                totals["raw_kl_sum"],
                totals["gap_sum"],
                totals["tokens"],
                totals["positive_tokens"],
                totals["supported_tokens"],
                totals["supported_gap_sum"],
                totals["rows"],
                totals["positive_rows"],
                totals["supported_rows"],
                totals["skipped_rows"],
                totals["truncated_rows"],
            ],
            dtype=torch.float64,
            device=next(model.parameters()).device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        values = packed.cpu().tolist()
        for key, value in zip(totals, values):
            totals[key] = value
    model.train()
    rows = int(totals["rows"])
    tokens = int(totals["tokens"])
    supported_tokens = int(totals["supported_tokens"])
    report = {
        "audited_rows": rows,
        "audited_tokens": tokens,
        "skipped_rows": int(totals["skipped_rows"]),
        "reference_truncated_rows": int(totals["truncated_rows"]),
        "mean_forward_kl": (
            totals["raw_kl_sum"] / tokens if tokens else None
        ),
        "mean_chosen_logp_gap": totals["gap_sum"] / tokens if tokens else None,
        "positive_row_fraction": totals["positive_rows"] / rows if rows else 0.0,
        "positive_token_fraction": (
            totals["positive_tokens"] / tokens if tokens else 0.0
        ),
        "teacher_gap_mode": args.teacher_gap_mode,
        "supported_mean_chosen_logp_gap": (
            totals["supported_gap_sum"] / supported_tokens
            if supported_tokens
            else None
        ),
        "supported_row_fraction": (
            totals["supported_rows"] / rows if rows else 0.0
        ),
        "supported_token_fraction": (
            supported_tokens / tokens if tokens else 0.0
        ),
        "thresholds": {
            "chosen_gap": args.min_teacher_chosen_gap,
            "supported_mean_gap": args.min_audit_supported_mean_gap,
            "supported_row_fraction": args.min_audit_supported_row_fraction,
            "supported_token_fraction": args.min_audit_supported_token_fraction,
        },
    }
    report["passed"] = bool(
        rows
        and tokens
        and report["supported_mean_chosen_logp_gap"] is not None
        and report["supported_mean_chosen_logp_gap"]
        >= args.min_audit_supported_mean_gap
        and report["supported_row_fraction"]
        >= args.min_audit_supported_row_fraction
        and report["supported_token_fraction"]
        >= args.min_audit_supported_token_fraction
    )
    return report


def write_and_validate_teacher_audit(
    report: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if is_main_process():
        audit_path = output_dir / "preflight_teacher_signal.json"
        with audit_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("[PREFLIGHT] " + json.dumps(report, ensure_ascii=False), flush=True)
    if (
        not report["audited_tokens"]
        or report["mean_forward_kl"] is None
        or report["mean_forward_kl"] <= 0
    ):
        raise RuntimeError(
            "Privileged teacher produced no measurable full-vocabulary forward KL: "
            + json.dumps(report, ensure_ascii=False)
        )
    if args.require_positive_teacher_signal and not report["passed"]:
        raise RuntimeError(
            "Teacher-signal preflight failed before optimization: "
            + json.dumps(report, ensure_ascii=False)
        )


def evaluate(
    model,
    tokenizer,
    groups: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not groups and not dist.is_initialized():
        return {"loss": None, "tokens": 0, "rows": 0, "skipped": 0, "role_loss": {}}
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    rows = 0
    skipped = 0
    role_sums: dict[str, float] = defaultdict(float)
    role_tokens: Counter[str] = Counter()
    for group in groups:
        for row in group:
            score = score_row(
                model,
                tokenizer,
                row,
                args,
                require_grad=False,
                apply_teacher_gap_gate=False,
            )
            if score is None:
                skipped += 1
                continue
            token_count = score["raw_token_count"]
            loss_sum = float(score["raw_loss_sum"].detach().cpu())
            total_loss += loss_sum
            total_tokens += token_count
            rows += 1
            role = str(row["role"])
            role_sums[role] += loss_sum
            role_tokens[role] += token_count
    if dist.is_initialized():
        packed = torch.tensor(
            [
                total_loss,
                total_tokens,
                rows,
                skipped,
                *[role_sums[role] for role in POLICY_ROLES],
                *[role_tokens[role] for role in POLICY_ROLES],
            ],
            dtype=torch.float64,
            device=next(model.parameters()).device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        values = packed.cpu().tolist()
        total_loss, total_tokens, rows, skipped = values[:4]
        offset = 4
        role_sums = defaultdict(
            float,
            {
                role: values[offset + index]
                for index, role in enumerate(POLICY_ROLES)
            },
        )
        offset += len(POLICY_ROLES)
        role_tokens = Counter(
            {
                role: round(values[offset + index])
                for index, role in enumerate(POLICY_ROLES)
            }
        )
    model.train()
    return {
        "loss": total_loss / total_tokens if total_tokens else None,
        "tokens": int(total_tokens),
        "rows": int(rows),
        "skipped": int(skipped),
        "role_loss": {
            role: role_sums[role] / role_tokens[role]
            for role in sorted(role_sums)
            if role_tokens[role]
        },
    }


def save_adapter(model, tokenizer, output_dir: Path, step: int) -> Path:
    path = output_dir / f"adapter_step_{step}"
    path.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path


def init_wandb(args: argparse.Namespace):
    if not args.wandb_project or not is_main_process():
        return None
    import wandb

    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            config={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            settings=wandb.Settings(
                init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "90"))
            ),
        )
    except Exception as error:
        print(
            f"[WARN] W&B initialization failed ({error}); continuing without online logging.",
            flush=True,
        )
        # A failed network initialization can leave W&B's service process in a
        # state where teardown blocks indefinitely. Other DDP ranks would then
        # wait at the following barrier until NCCL times out.
        return None


def train(
    model,
    tokenizer,
    train_groups: list[list[dict[str, Any]]],
    val_groups: list[list[dict[str, Any]]],
    args: argparse.Namespace,
    wandb_run,
) -> dict[str, Any]:
    from transformers import get_cosine_schedule_with_warmup

    if not train_groups:
        raise RuntimeError("No training sibling pairs are available")
    teacher_audit = audit_teacher_signal(model, tokenizer, train_groups, args)
    write_and_validate_teacher_audit(
        teacher_audit,
        args.output_dir,
        args,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    effective_gradient_accumulation = min(
        args.gradient_accumulation, len(train_groups)
    )
    updates_per_epoch = len(train_groups) // effective_gradient_accumulation
    total_steps = min(args.max_steps, max(1, updates_per_epoch * args.epochs))
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    rng = random.Random(args.seed)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    best_val_loss = float("inf")
    best_val_step = None
    global_step = 0
    workflows_since_update = 0
    update_loss_sum = 0.0
    update_policy_tokens = 0
    update_raw_tokens = 0
    update_gap_sum = 0.0
    update_positive_gap_tokens = 0
    skipped = Counter()
    started = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    main_process = is_main_process()
    has_validation = bool(val_groups)
    if dist.is_initialized():
        validation_flag = torch.tensor(
            int(has_validation), device=next(model.parameters()).device
        )
        dist.all_reduce(validation_flag, op=dist.ReduceOp.MAX)
        has_validation = bool(validation_flag.item())
    metrics_context = (
        metrics_path.open("w", encoding="utf-8")
        if main_process
        else contextlib.nullcontext(None)
    )

    with metrics_context as metrics_handle:
        for epoch in range(args.epochs):
            rng.shuffle(train_groups)
            for group in train_groups:
                usable = []
                group_token_count = 0
                for row in group:
                    score = score_row(model, tokenizer, row, args, require_grad=True)
                    if score is None:
                        skipped["unusable_or_too_long"] += 1
                        continue
                    if not score["token_count"]:
                        skipped["no_positive_teacher_signal"] += 1
                        continue
                    usable.append((row, score))
                    group_token_count += score["token_count"]
                if not usable or not group_token_count:
                    skipped["empty_workflow"] += 1
                group_is_usable = int(bool(usable and group_token_count))
                if dist.is_initialized():
                    usable_flag = torch.tensor(
                        group_is_usable, device=next(model.parameters()).device
                    )
                    dist.all_reduce(usable_flag, op=dist.ReduceOp.MIN)
                    group_is_usable = int(usable_flag.item())
                if not group_is_usable:
                    continue
                workflow_loss = sum(score["loss_sum"] for _, score in usable)
                workflow_loss = (
                    workflow_loss
                    / group_token_count
                    / effective_gradient_accumulation
                )
                workflow_loss.backward()
                update_loss_sum += sum(
                    float(score["loss_sum"].detach().cpu())
                    for _, score in usable
                )
                update_policy_tokens += group_token_count
                update_raw_tokens += sum(
                    score["raw_token_count"] for _, score in usable
                )
                update_gap_sum += sum(
                    score["chosen_logp_gap_sum"] for _, score in usable
                )
                update_positive_gap_tokens += sum(
                    score["positive_gap_token_count"] for _, score in usable
                )
                workflows_since_update += 1
                if workflows_since_update < effective_gradient_accumulation:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                workflows_since_update = 0
                global_step += 1
                train_loss_sum = update_loss_sum
                reported_tokens = float(update_policy_tokens)
                reported_raw_tokens = float(update_raw_tokens)
                reported_gap_sum = update_gap_sum
                reported_positive_gap_tokens = float(update_positive_gap_tokens)
                if dist.is_initialized():
                    reported = torch.tensor(
                        [
                            train_loss_sum,
                            reported_tokens,
                            reported_raw_tokens,
                            reported_gap_sum,
                            reported_positive_gap_tokens,
                        ],
                        dtype=torch.float64,
                        device=next(model.parameters()).device,
                    )
                    dist.all_reduce(reported, op=dist.ReduceOp.SUM)
                    (
                        train_loss_sum,
                        reported_tokens,
                        reported_raw_tokens,
                        reported_gap_sum,
                        reported_positive_gap_tokens,
                    ) = reported.cpu().tolist()
                train_loss = train_loss_sum / max(1.0, reported_tokens)
                record: dict[str, Any] = {
                    "step": global_step,
                    "epoch": epoch,
                    "train/opsd_forward_kl": train_loss,
                    "train/workflow_policy_tokens": int(reported_tokens),
                    "train/raw_policy_tokens": int(reported_raw_tokens),
                    "train/eligible_token_fraction": (
                        reported_tokens / reported_raw_tokens
                        if reported_raw_tokens
                        else 0.0
                    ),
                    "train/teacher_chosen_logp_gap": (
                        reported_gap_sum / reported_raw_tokens
                        if reported_raw_tokens
                        else None
                    ),
                    "train/positive_teacher_token_fraction": (
                        reported_positive_gap_tokens / reported_raw_tokens
                        if reported_raw_tokens
                        else 0.0
                    ),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/elapsed_seconds": time.time() - started,
                }
                update_loss_sum = 0.0
                update_policy_tokens = 0
                update_raw_tokens = 0
                update_gap_sum = 0.0
                update_positive_gap_tokens = 0
                should_eval = has_validation and (
                    global_step == 1
                    or global_step == total_steps
                    or (args.eval_every > 0 and global_step % args.eval_every == 0)
                )
                if should_eval:
                    val = evaluate(model, tokenizer, val_groups, args)
                    record["val/opsd_forward_kl"] = val["loss"]
                    record["val/policy_tokens"] = val["tokens"]
                    record["val/skipped_rows"] = val["skipped"]
                    for role, value in val["role_loss"].items():
                        record[f"val/opsd_forward_kl/{role}"] = value
                    if val["loss"] is not None and val["loss"] < best_val_loss:
                        best_val_loss = float(val["loss"])
                        best_val_step = global_step
                        if main_process:
                            best_path = args.output_dir / f"best_adapter_by_distill_loss_step_{global_step}"
                            unwrap_model(model).save_pretrained(best_path, safe_serialization=True)
                            tokenizer.save_pretrained(best_path)
                if main_process:
                    metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_handle.flush()
                    if wandb_run is not None:
                        wandb_run.log(record, step=global_step)
                    if global_step == 1 or global_step % args.log_every == 0:
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                if main_process and args.save_every > 0 and global_step % args.save_every == 0:
                    save_adapter(model, tokenizer, args.output_dir, global_step)
                if dist.is_initialized() and (
                    should_eval
                    or (args.save_every > 0 and global_step % args.save_every == 0)
                ):
                    dist.barrier()
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break
    if dist.is_initialized():
        skipped_rows: list[dict[str, int] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(skipped_rows, dict(skipped))
        skipped = Counter()
        for item in skipped_rows:
            skipped.update(item or {})
    return {
        "global_step": global_step,
        "total_steps_planned": total_steps,
        "best_val_distill_loss": None if best_val_loss == float("inf") else best_val_loss,
        "best_val_distill_loss_step": best_val_step,
        "teacher_signal_preflight": teacher_audit,
        "skipped": dict(skipped),
        "elapsed_seconds": time.time() - started,
    }


def main() -> int:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    rank, world_size, local_rank = distributed_info()
    main_process = rank == 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}; use --overwrite or a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    device = torch.device("cuda", local_rank)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        ).to(device)
    except (ImportError, ValueError) as error:
        if main_process:
            print(f"[WARN] FlashAttention unavailable ({error}); falling back to SDPA.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="sdpa",
            **model_kwargs,
        ).to(device)
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
    if main_process:
        model.print_trainable_parameters()
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    all_train_groups = group_rows(read_jsonl(args.train_file))
    all_val_groups = group_rows(read_jsonl(args.val_file))
    usable_train_count = len(all_train_groups) // world_size * world_size
    if usable_train_count < world_size:
        raise RuntimeError(
            f"Need at least {world_size} complete training pairs, got {len(all_train_groups)}"
        )
    train_groups = all_train_groups[:usable_train_count][rank::world_size]
    val_groups = all_val_groups[rank::world_size]
    if main_process:
        print(
            f"[DATA] train_pairs={usable_train_count} val_pairs={len(all_val_groups)} "
            f"world_size={world_size} pairs_per_rank={len(train_groups)}"
        )
    if args.audit_only:
        teacher_audit = audit_teacher_signal(
            model,
            tokenizer,
            train_groups,
            args,
        )
        write_and_validate_teacher_audit(
            teacher_audit,
            args.output_dir,
            args,
        )
        summary = {
            "audit_only": True,
            "optimizer_steps": 0,
            "base_model": args.model,
            "teacher_signal_preflight": teacher_audit,
        }
        if main_process:
            with (args.output_dir / "audit_summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print("[AUDIT DONE] " + json.dumps(summary, ensure_ascii=False), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0

    wandb_run = init_wandb(args)
    if dist.is_initialized():
        dist.barrier()
    try:
        summary = train(model, tokenizer, train_groups, val_groups, args, wandb_run)
        final_adapter = args.output_dir / f"adapter_step_{summary['global_step']}"
        if main_process:
            save_adapter(model, tokenizer, args.output_dir, summary["global_step"])
        if dist.is_initialized():
            dist.barrier()
        summary.update(
            {
                "base_model": args.model,
                "final_adapter": str(final_adapter),
                "selection_note": (
                    "best_adapter_by_distill_loss is diagnostic only; select the deployable "
                    "adapter by validation EM before merging, as in Search-E1."
                ),
            }
        )
        if main_process:
            with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print("[DONE] " + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
