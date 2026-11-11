#!/usr/bin/env python3
"""Evaluate the released Qwen2.5-3B Search-R1 checkpoint on EWC worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import requests
import torch
import transformers

from evaluate_ewc_pilot import em, paired_summary


INSTRUCTION = """Answer the given question.

You must conduct reasoning inside <think> and </think> first every time you get new information.

After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.

You can search as many times as your want.

If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
"""
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
STOP_STRINGS = ("</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n")


class SearchStop(transformers.StoppingCriteria):
    def __init__(self, tokenizer: Any) -> None:
        self.tokens = [tokenizer.encode(value, add_special_tokens=False) for value in STOP_STRINGS]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **_: Any) -> bool:
        return any(len(ids) <= input_ids.shape[1] and input_ids[0, -len(ids):].tolist() == ids for ids in self.tokens)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--overlay-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-instances", type=int)
    return parser.parse_args()


def load(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"instance_id", "pair_id", "variant", "question", "gold_answer"}
    for index, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{index}: missing {sorted(missing)}")
    return rows if limit is None else rows[:limit]


def select(overlay: str, instance_id: str, timeout: float) -> None:
    response = requests.post(overlay.rstrip("/") + "/select", json={"instance_id": instance_id}, timeout=timeout)
    response.raise_for_status()
    if response.json().get("active_instance_id") != instance_id:
        raise RuntimeError(f"overlay selected a different instance for {instance_id}")


def search(overlay: str, query: str, timeout: float) -> str:
    response = requests.post(
        overlay.rstrip("/") + "/retrieve",
        json={"queries": [query], "topk": 3, "return_scores": True},
        timeout=timeout,
    )
    response.raise_for_status()
    documents = response.json()["result"][0]
    values = []
    for index, item in enumerate(documents, 1):
        content = str(item["document"]["contents"])
        title, _, text = content.partition("\n")
        values.append(f"Doc {index}(Title: {title}) {text}")
    return "\n".join(values)


def initial_prompt(tokenizer: Any, question: str) -> str:
    question = question.strip()
    content = INSTRUCTION.format(question=question if question.endswith("?") else question + "?")
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    return content


def run(model: Any, tokenizer: Any, overlay: str, question: str, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    prompt = initial_prompt(tokenizer, question)
    stop = transformers.StoppingCriteriaList([SearchStop(tokenizer)])
    trace: list[dict[str, Any]] = []
    for turn in range(1, args.max_turns + 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=stop,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature,
        )[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        answers = ANSWER_RE.findall(text)
        if answers:
            trace.append({"turn": turn, "output": text, "search_query": None, "observation": None})
            return {"answer": answers[-1].strip(), "trace": trace, "turns": turn, "format_failure": False}
        queries = SEARCH_RE.findall(text)
        query = queries[-1].strip() if queries else ""
        observation = search(overlay, query, args.timeout) if query else ""
        trace.append({"turn": turn, "output": text, "search_query": query or None, "observation": observation})
        if not query:
            return {"answer": "", "trace": trace, "turns": turn, "format_failure": True}
        prompt += f"\n\n{text}<information>{observation}</information>\n\n"
    return {"answer": "", "trace": trace, "turns": args.max_turns, "format_failure": True}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    rows = load(args.instances, args.max_instances)
    if not rows:
        raise SystemExit("no instances")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    results = []
    for index, row in enumerate(rows, 1):
        native = {"answer": "", "trace": [], "turns": 0, "format_failure": True}
        error = None
        try:
            select(args.overlay_url, str(row["instance_id"]), args.timeout)
            native = run(model, tokenizer, args.overlay_url, str(row["question"]), device, args)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        answer = str(native["answer"])
        results.append({
            "instance_id": row["instance_id"], "pair_id": row["pair_id"], "variant": row["variant"],
            "question": row["question"], "gold_answer": row["gold_answer"], "final_answer": answer,
            "em": em(answer, list(row["gold_answer"])), "pipeline_error": error is not None,
            "format_failure": bool(native["format_failure"]), "error": error,
            "turns": int(native["turns"]), "trace": native["trace"],
        })
        print(json.dumps({"completed": index, "total": len(rows), "variant": row["variant"]}), flush=True)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results) + "\n", encoding="utf-8")
    summary = {
        "benchmark": "EWC-QA validation v6", "instances": len(results), "em": sum(row["em"] for row in results),
        "em_rate": sum(row["em"] for row in results) / len(results),
        "pipeline_errors": sum(bool(row["pipeline_error"]) for row in results),
        "format_failures": sum(bool(row["format_failure"]) for row in results),
        "paired": paired_summary(results) if args.max_instances is None else None,
        "protocol": {"runtime": "released Search-R1 Qwen2.5 infer.py loop", "selection": "serial overlay /select",
                     "instances_sha256": digest(args.instances), "max_turns": args.max_turns,
                     "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "seed": args.seed},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["pipeline_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
