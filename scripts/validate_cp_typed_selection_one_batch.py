#!/usr/bin/env python3
"""Run a one-batch gradient audit for CP-DAGENT typed selection credit.

The model generates Reason selections from exact logged action prompts. The
audit then compares each selected parent with the other legal parent and checks
that bounded decisiveness weights preserve the terminal-EM gradient direction.
No optimizer update is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SELECTION_RE = re.compile(
    r'<selection\s+ref=["\']#?(\d+)#?["\']\s*>\s*'
    r'<value>(.*?)</value>\s*</selection>',
    re.IGNORECASE | re.DOTALL,
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: Any, references: Any) -> int:
    candidates = [references] if isinstance(references, str) else list(references or [])
    prediction = normalize_answer(prediction)
    return int(bool(prediction) and prediction in {normalize_answer(item) for item in candidates})


def parse_selection(text: str) -> dict[str, str] | None:
    match = SELECTION_RE.search(str(text or ""))
    if not match:
        return None
    return {"ref": match.group(1), "value": match.group(2).strip()}


def collect_action_prompts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        trace = row.get("raw_trace") or {}
        events = list(trace.get("sft_events") or [])
        for node in trace.get("nodes_execution") or []:
            if node.get("reason_execution_method") != "model_typed_selection":
                continue
            node_id = str(node.get("id") or "")
            matches = [
                event
                for event in events
                if event.get("agent_name") == "ReasonerAgent"
                and str((event.get("metadata") or {}).get("node_id") or "") == node_id
            ]
            if len(matches) != 1:
                continue
            handoff = node.get("typed_handoff") or {}
            selected_ref = str(handoff.get("selected_ref") or "")
            sources = []
            if selected_ref:
                sources.append(
                    {
                        "id": selected_ref,
                        "task": str(handoff.get("selected_task") or ""),
                        "value": str(handoff.get("selected_value") or ""),
                        "candidate": str(handoff.get("selected_candidate") or ""),
                    }
                )
            for source in handoff.get("rejected_sources") or []:
                sources.append(
                    {
                        "id": str(source.get("id") or ""),
                        "task": str(source.get("task") or ""),
                        "value": str(source.get("value") or ""),
                        "candidate": str(source.get("candidate") or ""),
                    }
                )
            if len(sources) < 2 or any(
                not source["id"] or not source["candidate"] or not source["value"]
                for source in sources
            ):
                continue
            actions.append(
                {
                    "row_index": row_index,
                    "question": str(row.get("question") or ""),
                    "gold_answer": row.get("gold_answer") or [],
                    "messages": list(matches[0].get("messages") or []),
                    "sources": sources,
                }
            )
    return actions


def validate_generated_action(action: Mapping[str, Any], response: str) -> dict[str, Any]:
    selection = parse_selection(response)
    by_id = {str(source["id"]): source for source in action["sources"]}
    selected = by_id.get(str((selection or {}).get("ref") or ""))
    error = ""
    if selection is None:
        error = "missing_selection"
    elif selected is None:
        error = "undeclared_ref"
    elif normalize_text(selection["value"]) != normalize_text(selected["value"]):
        error = "value_ref_mismatch"
    if error:
        return {"valid": False, "error": error, "response": response}
    alternatives = [source for source_id, source in by_id.items() if source_id != selection["ref"]]
    factual_em = exact_match(selected["candidate"], action["gold_answer"])
    alternative_em = max(
        (exact_match(source["candidate"], action["gold_answer"]) for source in alternatives),
        default=0,
    )
    return {
        "valid": True,
        "error": "",
        "response": response,
        "selected_ref": selection["ref"],
        "selected_candidate": selected["candidate"],
        "factual_em": factual_em,
        "alternative_em": alternative_em,
        "decisive": factual_em != alternative_em,
    }


def _chat_text(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )


def _response_lengths(tokens: torch.Tensor, eos_token_id: int | None) -> list[int]:
    lengths = []
    for row in tokens.tolist():
        length = len(row)
        if eos_token_id is not None and eos_token_id in row:
            length = row.index(eos_token_id) + 1
        lengths.append(max(1, length))
    return lengths


def _trainable_last_block(model) -> tuple[list[torch.nn.Parameter], str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not layers:
        raise ValueError("model does not expose model.layers")
    block = layers[-1]
    for parameter in block.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in block.parameters() if parameter.requires_grad]
    return parameters, f"{type(block).__name__}:last"


def _objective_gradient(
    model,
    parameters: Sequence[torch.nn.Parameter],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    micro_batch_size: int,
) -> list[torch.Tensor]:
    for parameter in parameters:
        parameter.grad = None
    count = sequences.shape[0]
    for start in range(0, count, micro_batch_size):
        stop = min(count, start + micro_batch_size)
        outputs = model(
            input_ids=sequences[start:stop],
            attention_mask=attention_mask[start:stop],
            use_cache=False,
        )
        logits = outputs.logits[:, :-1].float()
        targets = sequences[start:stop, 1:]
        token_log_probs = torch.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        mask = response_mask[start:stop, 1:].float()
        sequence_scores = (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
        loss = -(
            advantages[start:stop] * weights[start:stop] * sequence_scores
        ).sum() / count
        loss.backward()
        del outputs, logits, token_log_probs, sequence_scores, loss
    gradients = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        gradients.append(gradient.detach().float().cpu().clone())
        parameter.grad = None
    return gradients


def _dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    return sum(float((a.double() * b.double()).sum().item()) for a, b in zip(left, right))


def _gradient_metrics(
    em: Sequence[torch.Tensor], real: Sequence[torch.Tensor], shuffled: Sequence[torch.Tensor]
) -> dict[str, float]:
    em_norm = math.sqrt(max(0.0, _dot(em, em)))
    real_norm = math.sqrt(max(0.0, _dot(real, real)))
    shuffled_norm = math.sqrt(max(0.0, _dot(shuffled, shuffled)))
    real_em = _dot(real, em)
    shuffled_em = _dot(shuffled, em)
    extra_real = [a - b for a, b in zip(real, em)]
    extra_shuffled = [a - b for a, b in zip(shuffled, em)]
    extra_real_norm = math.sqrt(max(0.0, _dot(extra_real, extra_real)))
    extra_shuffled_norm = math.sqrt(max(0.0, _dot(extra_shuffled, extra_shuffled)))
    return {
        "em_norm": em_norm,
        "weighted_norm": real_norm,
        "shuffled_norm": shuffled_norm,
        "weighted_em_cosine": real_em / max(1e-30, real_norm * em_norm),
        "shuffled_em_cosine": shuffled_em / max(1e-30, shuffled_norm * em_norm),
        "weighted_extra_to_em_norm_ratio": extra_real_norm / max(1e-30, em_norm),
        "shuffled_extra_to_em_norm_ratio": extra_shuffled_norm / max(1e-30, em_norm),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < args.alpha <= 0.2):
        raise ValueError("alpha must be in (0, 0.2]")
    if args.output.exists() or args.output.parent.exists():
        raise FileExistsError(f"refusing to reuse output path: {args.output.parent}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = json.loads(args.trajectories.read_text(encoding="utf-8"))
    actions = collect_action_prompts(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    prompt_texts = [_chat_text(tokenizer, action["messages"]) for action in actions]
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).cuda().eval()
    input_ids = inputs["input_ids"].cuda()
    input_attention = inputs["attention_mask"].cuda()
    input_width = input_ids.shape[1]
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=input_attention,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[:, input_width:]
    responses = tokenizer.batch_decode(continuation, skip_special_tokens=True)
    audited = [
        {**action, **validate_generated_action(action, response)}
        for action, response in zip(actions, responses)
    ]
    valid_indices = [index for index, item in enumerate(audited) if item["valid"]]
    valid = [audited[index] for index in valid_indices]
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "cp_typed_selection_one_batch_gradient_audit_v1",
        "model": str(args.model),
        "trajectory_source": str(args.trajectories),
        "frozen_upstream_action_prompts": True,
        "model_generated_reason_actions": True,
        "optimizer_steps": 0,
        "alpha": args.alpha,
        "prompt_cases": len(actions),
        "valid_generated_actions": len(valid),
        "format_valid_rate": len(valid) / max(1, len(actions)),
        "invalid_reasons": {
            reason: sum(item.get("error") == reason for item in audited)
            for reason in sorted({item.get("error") for item in audited if item.get("error")})
        },
        "decisive_actions": sum(item.get("decisive", False) for item in valid),
        "factual_wins": sum(
            item.get("factual_em", 0) > item.get("alternative_em", 0) for item in valid
        ),
        "alternative_wins": sum(
            item.get("alternative_em", 0) > item.get("factual_em", 0) for item in valid
        ),
        "gradient_scope": None,
        "gradient_metrics": None,
        "actions": audited,
        "evidence_scope": {
            "offline_signal_audit": True,
            "one_batch_gradient_check": False,
            "optimizer_update": False,
            "short_training": False,
            "formal_evaluation": False,
        },
    }
    enough_signal = bool(
        len(valid) >= 8
        and report["decisive_actions"] >= 4
        and report["factual_wins"] > 0
        and report["alternative_wins"] > 0
    )
    if enough_signal:
        selected = torch.tensor(valid_indices, device=generated.device, dtype=torch.long)
        sequences = generated.index_select(0, selected)
        selected_input_attention = input_attention.index_select(0, selected)
        lengths = _response_lengths(
            continuation.index_select(0, selected), tokenizer.eos_token_id
        )
        response_mask = torch.zeros_like(sequences, dtype=torch.bool)
        full_attention = torch.zeros_like(sequences, dtype=torch.long)
        full_attention[:, :input_width] = selected_input_attention
        for row_index, length in enumerate(lengths):
            response_mask[row_index, input_width : input_width + length] = True
            full_attention[row_index, input_width : input_width + length] = 1
        advantages = torch.tensor(
            [1.0 if item["factual_em"] else -1.0 for item in valid],
            device=generated.device,
        )
        decisive = torch.tensor(
            [float(item["decisive"]) for item in valid], device=generated.device
        )
        shuffled_values = decisive.tolist()
        for _ in range(32):
            random.shuffle(shuffled_values)
            if shuffled_values != decisive.tolist():
                break
        base_weights = torch.ones_like(advantages)
        real_weights = 1.0 + args.alpha * decisive
        shuffled_weights = 1.0 + args.alpha * torch.tensor(
            shuffled_values, device=generated.device
        )
        parameters, scope = _trainable_last_block(model)
        model.config.use_cache = False
        gradients = {}
        for name, weights in (
            ("em", base_weights),
            ("weighted", real_weights),
            ("shuffled", shuffled_weights),
        ):
            gradients[name] = _objective_gradient(
                model,
                parameters,
                sequences,
                full_attention,
                response_mask,
                advantages,
                weights,
                args.micro_batch_size,
            )
            torch.cuda.empty_cache()
        metrics = _gradient_metrics(
            gradients["em"], gradients["weighted"], gradients["shuffled"]
        )
        finite = all(math.isfinite(value) for value in metrics.values())
        report["gradient_scope"] = {
            "module": scope,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "micro_batch_size": args.micro_batch_size,
        }
        report["gradient_metrics"] = metrics
        report["evidence_scope"]["one_batch_gradient_check"] = True
        report["passed"] = bool(
            finite
            and metrics["em_norm"] > 0
            and metrics["weighted_extra_to_em_norm_ratio"] > 1e-5
            and metrics["weighted_em_cosine"] > 0.90
        )
    else:
        report["passed"] = False
        report["stop_reason"] = "insufficient_valid_bidirectional_causal_signal"
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "actions"}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
