#!/usr/bin/env python3
"""Evaluate the public Qwen3 Search-R1 checkpoint on the frozen full manifest.

The runtime follows the upstream OpenRLHF-Agent Qwen3 protocol: thinking is
enabled, the model emits native tool calls, and every search observation is
fed back before the next assistant turn.  The output is deliberately kept
separate from DAGENT traces and is converted to the common scored-row schema
only after each dataset is complete.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from openrlhf_agent.model.backends.base import CompletionBackend, GenerationResult


DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "bamboogle")
ANSWER_RE = re.compile(r"(?im)^\s*Answer:\s*(.+?)\s*$")

SYSTEM_PROMPT = """
You are a helpful assistant.

## Output Rules
- First provide a clear markdown explanation of the solution.
- Then end exactly with:
  `Answer: <final_answer>`
- The answer line must contain only the final answer in canonical form.
- Do not add any text after the final answer line.
""".strip()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True, help="vLLM OpenAI base URL, e.g. http://127.0.0.1:8009/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-rows", type=int, help="Optional smoke limit; omit for the complete manifest.")
    parser.add_argument(
        "--backend-mode",
        choices=("official", "openai-completions"),
        default="official",
        help="Use the official token endpoint or standard vLLM completions compatibility mode.",
    )
    return parser.parse_args()


def load_manifest(root: Path) -> dict[str, list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    summary = json.loads((root / "manifest_summary.json").read_text(encoding="utf-8"))
    if int(summary.get("total_selected_rows", -1)) != 22517:
        raise RuntimeError("unexpected full manifest size")
    result: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        rows = pq.read_table(root / f"{dataset}.parquet", use_threads=False).to_pylist()
        expected = int(summary["datasets"][dataset]["selected_rows"])
        if len(rows) != expected:
            raise RuntimeError(f"manifest size mismatch for {dataset}")
        result[dataset] = rows
    return result


def answer_values(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(str(text or ""))
    return matches[-1].strip() if matches else ""


def score_answer(prediction: str, golds: list[str]) -> tuple[float, float]:
    from qa_manager.compute_metrics import compute_scores

    metrics = compute_scores([prediction], [golds])
    return float(metrics["em"]), float(metrics["f1"])


class OpenAICompletionCompatBackend(CompletionBackend):
    """Preserve the completion protocol over standard vLLM OpenAI endpoints."""

    def __init__(self, *, model: str, base_url: str, timeout: float = 600.0):
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/").removesuffix("/v1"), timeout=timeout
        )

    async def tokenize(self, prompt: str, *, add_special_tokens: bool = True) -> list[int]:
        response = await self.client.post(
            "/tokenize",
            json={
                "model": self.model,
                "prompt": prompt,
                "add_special_tokens": add_special_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["tokens"]

    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int | None = None,
        *,
        images: Any = None,
        sampling_params: dict[str, Any] | None = None,
        return_logprobs: bool = False,
        session_id: str | None = None,
    ) -> GenerationResult:
        params = dict(sampling_params or {})
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "presence_penalty",
            "frequency_penalty",
        ):
            if params.get(key) is not None:
                payload[key] = params[key]
        response = await self.client.post("/v1/completions", json=payload)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        text = str(choice.get("text") or "")
        token_ids = await self.tokenize(text, add_special_tokens=False)
        return GenerationResult(
            text=text,
            token_ids=token_ids,
            finish_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self.client.aclose()


async def run_one(
    backend: Any,
    retriever_url: str,
    question: str,
    max_turns: int,
    max_tokens: int,
) -> dict[str, Any]:
    from openrlhf_agent.agentkit import AgentRuntime
    from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
    from openrlhf_agent.agentkit.tools import WikiSearchTool
    from openrlhf_agent.model import Qwen3Protocol

    runtime = AgentRuntime(
        backend=backend,
        protocol=Qwen3Protocol(enable_thinking=True),
        environment=FunctionCallEnvironment(
            system_prompt=SYSTEM_PROMPT,
            tools=[WikiSearchTool(base_url=retriever_url)],
            max_steps=max_turns,
        ),
        max_tokens=max_tokens,
    )
    trace: list[dict[str, Any]] = []
    final_text = ""
    turns = 0
    async for message in runtime.run_steps([{"role": "user", "content": question}]):
        trace.append(message)
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            final_text = message["content"]
            turns += 1
    return {"final_text": final_text, "trace": trace, "turns": turns}


async def evaluate_dataset(args: argparse.Namespace, dataset: str, rows: list[dict[str, Any]]) -> None:
    backend = args.backend
    output_dir = args.output_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=False) if not output_dir.exists() else None
    path = output_dir / f"search_r1_{dataset}.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any] | None] = [None] * len(rows)
    async def worker(index: int, row: dict[str, Any]) -> None:
        async with semaphore:
            started = time.time()
            try:
                result = await run_one(
                    backend,
                    args.retriever_url,
                    str(row["question"]),
                    args.max_turns,
                    args.max_tokens,
                )
                prediction = extract_answer(result["final_text"])
                em, f1 = score_answer(prediction, answer_values(row["golden_answers"]))
                results[index] = {
                    "question": row["question"],
                    "golden_answers": answer_values(row["golden_answers"]),
                    "prediction": prediction,
                    "em": em,
                    "f1": f1,
                    "trace": result["trace"],
                    "turns": result["turns"],
                    "elapsed_seconds": time.time() - started,
                    "error": None,
                }
            except Exception as exc:  # retain a row; fail-closed scorer rejects it later
                results[index] = {
                    "question": row["question"],
                    "golden_answers": answer_values(row["golden_answers"]),
                    "prediction": "",
                    "em": math.nan,
                    "f1": math.nan,
                    "trace": [],
                    "turns": 0,
                    "elapsed_seconds": time.time() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    await asyncio.gather(*(worker(index, row) for index, row in enumerate(rows)))
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] {dataset} rows={len(results)} path={path}")


async def main_async(args: argparse.Namespace) -> None:
    from openrlhf_agent.model import VLLMCompletionBackend

    manifest = load_manifest(args.manifest_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.backend_mode == "official":
        args.backend = VLLMCompletionBackend(
            base_url=args.api_base_url,
            api_key="empty",
            model=args.model,
            timeout=args.timeout,
        )
    else:
        args.backend = OpenAICompletionCompatBackend(
            base_url=args.api_base_url,
            model=args.model,
            timeout=args.timeout,
        )
    try:
        for dataset in DATASETS:
            rows = manifest[dataset]
            if args.max_rows is not None:
                if args.max_rows < 1:
                    raise ValueError("--max-rows must be positive")
                rows = rows[: args.max_rows]
            await evaluate_dataset(args, dataset, rows)
    finally:
        await args.backend.aclose()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
