#!/usr/bin/env python3
"""Measure the causal value of reason conclusions by removing them before final synthesis."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import string
from typing import Any, Sequence

REASON_BLOCK = re.compile(
    r"(?:\n\n|\A)Task\s+\d+\s+\[REASON\]:.*?\nConclusion:.*?(?=\n\nTask\s+\d+\s+\[|\Z)",
    flags=re.DOTALL,
)


def normalize(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(answer: str, gold: Sequence[str]) -> bool:
    normalized = normalize(answer)
    return bool(normalized) and any(normalized == normalize(candidate) for candidate in gold)


def extract_answer(text: str) -> str:
    match = re.fullmatch(r"\s*<answer>(.*?)</answer>\s*", str(text or ""), flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def counterfactual_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str] | None:
    events = row.get("trace", {}).get("sft_events", [])
    summaries = [event for event in events if event.get("agent_name") == "AnswerSummarizationAgent"]
    if not summaries:
        return None
    event = summaries[-1]
    messages = [dict(message) for message in event.get("messages", [])]
    if not messages:
        return None
    changed = False
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        rewritten, count = REASON_BLOCK.subn("", content)
        if count:
            message["content"] = rewritten.strip()
            changed = True
    if not changed:
        return None
    return messages, str(event.get("response", ""))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["data_source"], "ALL")].append(row)
        buckets[(row["data_source"], row["question_type"])].append(row)
    report = {}
    for (source, question_type), values in sorted(buckets.items()):
        original = sum(row["original_em"] for row in values)
        counterfactual = sum(row["counterfactual_em"] for row in values)
        report[f"{source}/{question_type}"] = {
            "count": len(values),
            "original_em": original / len(values),
            "without_reason_em": counterfactual / len(values),
            "em_delta": (original - counterfactual) / len(values),
            "reason_helped": sum(row["original_em"] and not row["counterfactual_em"] for row in values),
            "reason_hurt": sum(not row["original_em"] and row["counterfactual_em"] for row in values),
            "unchanged": sum(row["original_em"] == row["counterfactual_em"] for row in values),
        }
    return report


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = json.loads(args.input.read_text(encoding="utf-8"))
    cases = []
    for row in source_rows:
        prepared = counterfactual_messages(row)
        if prepared is None:
            continue
        messages, original_response = prepared
        cases.append(
            {
                "data_source": row.get("data_source", ""),
                "row_index": row.get("row_index"),
                "question": row.get("question", ""),
                "question_type": row.get("question_type", ""),
                "gold_answers": row.get("gold_answers", []),
                "original_answer": extract_answer(original_response) or row.get("final_answer", ""),
                "messages": messages,
            }
        )
    print(f"[CONFIG] counterfactual_cases={len(cases)} batch_size={args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    results = []
    for start in range(0, len(cases), args.batch_size):
        batch = cases[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
            for case in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for case, raw_output in zip(batch, outputs):
            answer = extract_answer(raw_output)
            results.append(
                {
                    **{key: value for key, value in case.items() if key != "messages"},
                    "counterfactual_raw_output": raw_output,
                    "counterfactual_answer": answer,
                    "original_em": int(exact_match(case["original_answer"], case["gold_answers"])),
                    "counterfactual_em": int(exact_match(answer, case["gold_answers"])),
                }
            )
        print(f"[PROGRESS] {min(start + len(batch), len(cases))}/{len(cases)}")

    report = summarize(results)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
