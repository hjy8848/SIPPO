#!/usr/bin/env python3
"""Batch-evaluate external search agents on JADE's frozen full manifest.

This runner keeps external protocols separate from DAGENT traces.  It speaks
the completion API exposed by vLLM, executes the model's XML search loop, and
writes one scored JSON file per dataset.  The runner is intentionally
fail-closed: it never overwrites an output root and it records execution or
format failures instead of turning them into valid-looking zero scores.

The protocol files under ``configs/external_eval`` pin the prompt family,
stop strings, continuation template, and retrieval response format for each
baseline.  They are not claims that the native upstream code and JADE's
retriever are scientifically identical; the resulting run manifest records
the substitutions explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "bamboogle")
DEFAULT_PROTOCOL_ROOT = Path(__file__).resolve().parents[1] / "configs" / "external_eval"

SEARCH_R1_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

AUTOREFINE_PROMPT = """You are a helpful assistant excel at answering questions with multi-turn search engine calling. \\
To answer questions, you must first reason through the available information using <think> and </think>. \\
If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. \\
After each search, you need to summarize and refine the existing documents in <refine> and </refine>. \\
You may send multiple search requests if needed. \\
Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>. Question: {question}\n"""

O2_SYSTEM_PROMPT = """As a expert researcher, provide comprehensive key findings for open-ended queries and precise answers to other specific questions. Each time you receive new information, you MUST first engage in reasoning within the <think> and </think> tags.\nAfter reasoning, if you realize that you lack certain knowledge, you can invoke a SEARCH action with distinct queries (one to five) using the <search>\\n<query>QUERY</query>\\n<query>QUERY</query>\\n</search> format to obtain relevant learnings, which will be presented between the <learnings> and </learnings> tags.\nYou are allowed to perform searches as many times as necessary.\nIf you determine that no additional external knowledge is required, you can directly present the output within the <answer> and </answer> tags."""
O2_ERROR_PROMPT = """The response you attempted before is invalid. If you plan to execute actions like SEARCH, you need to enclose the SEARCH queries within the <search> and </search> tags. Furthermore, the required queries for the SEARCH action should be placed between the <query> and </query> tags. Moreover, if you wish to present the final output for the initial query, you must wrap the result within the <answer> and </answer> tags.\n"""

REOPD_TOOL_SYSTEM_PROMPT = """You are a helpful assistant that can use tools to answer questions. You have access to a search tool.

# Tools
You may call one function at a time. Return a function call as JSON inside a <tool_call>...</tool_call> block:
<tools>
<tool name=\"search\" description=\"Search for information.\" args=\"{\\\"query\\\": string}\" />
</tools>
<tool_call>
{\"name\": \"search\", \"arguments\": {\"query\": \"your search query\"}}
</tool_call>
After a tool call, the result will be provided in a <tool_response>...</tool_response> block. When you are ready to answer, output only the answer inside \\boxed{...}."""

HIPRAG_SHORT_PROMPT = """Answer user questions by thinking step-by-step. Your entire reasoning process must be encapsulated within a single <think></think> block, which contains one or more <step></step> blocks. Each step must begin with your analysis in <reasoning>. If you identify a knowledge gap, you may use <search>query</search> to query a search engine; search results will then be provided in a <context> tag. Every step must end with a <conclusion> summarizing what you learned in that step.
After your thinking process is complete, provide the final, conclusive answer inside an <answer> tag placed immediately after the closing </think> tag. You can use as many steps as you need. Ensure all XML tags are properly formed and nested.
**## Output Format Specification**
The <think> block contains the steps, and the <answer> block follows it.
"""

RAG_PROMPT = """Answer the question using the retrieved passages below. If the passages do not contain enough information, use your best judgment. Return only a concise answer inside <answer>...</answer>.

Question: {question}

Retrieved passages:
{search_results}
"""

IRCOT_PROMPT = """Answer the question by reasoning one sentence at a time. On each turn, either produce exactly one useful intermediate reasoning sentence inside <think>...</think>, or, when you have enough evidence, give the concise final answer inside <answer>...</answer>. Retrieved passages from earlier turns remain available in the conversation. Do not repeat an earlier reasoning sentence.

Question: {question}
"""

TAG_RE = {
    "search": re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE),
    "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
}
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=(
            "search_r1",
            "zerosearch",
            "hiprag",
            "reopd",
            "opd",
            "o2searcher",
            "autorefine_base",
            "autorefine_instruct",
            "rag",
            "ircot",
        ),
        required=True,
    )
    parser.add_argument("--model", required=True, help="Model name used by the completion endpoint")
    parser.add_argument("--tokenizer", type=Path, help="Local tokenizer directory for upstream chat-template parity")
    parser.add_argument("--api-base-url", required=True, help="vLLM OpenAI base URL, with or without /v1")
    parser.add_argument("--retriever-url", required=True, help="Retriever base URL exposing the configured protocol endpoint")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--retrieval-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-rows", type=int, help="Limit each dataset for a smoke run; omit for all rows")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.protocol_config or (DEFAULT_PROTOCOL_ROOT / f"{args.baseline}.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("baseline") != args.baseline:
        raise ValueError(f"protocol baseline mismatch: {config.get('baseline')} != {args.baseline}")
    config["path"] = str(config_path.resolve())
    config["sha256"] = sha256(config_path)
    config["max_turns"] = args.max_turns or int(config["max_turns"])
    config["max_tokens"] = args.max_tokens or int(config["max_tokens"])
    config["temperature"] = config.get("temperature", 0.7) if args.temperature is None else args.temperature
    config["top_p"] = config.get("top_p") if args.top_p is None else args.top_p
    config["top_k"] = int(config.get("top_k", 0))
    config["repetition_penalty"] = float(config.get("repetition_penalty", 1.0))
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.retrieval_retries < 1:
        raise ValueError("--retrieval-retries must be positive")
    return config


def answer_values(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def load_manifest(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import pyarrow.parquet as pq

    summary_path = root / "manifest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("total_selected_rows", -1)) != 22_517:
        raise RuntimeError(f"unexpected full manifest size in {summary_path}")
    result: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        path = root / f"{dataset}.parquet"
        if sha256(path) != summary["datasets"][dataset]["parquet_sha256"]:
            raise RuntimeError(f"manifest SHA mismatch: {path}")
        rows = pq.read_table(path, use_threads=False).to_pylist()
        expected = int(summary["datasets"][dataset]["selected_rows"])
        if len(rows) != expected:
            raise RuntimeError(f"manifest row mismatch for {dataset}: {len(rows)} != {expected}")
        for row in rows:
            if not answer_values(row.get("golden_answers")):
                raise RuntimeError(f"missing gold answer in {dataset}: {row.get('question')}")
        result[dataset] = rows
    return result, summary


class CompletionClient:
    def __init__(self, base_url: str, model: str, timeout: float, concurrency: int) -> None:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The selected evaluation environment is missing httpx; "
                "use the JADE runtime or install requirements.txt"
            ) from exc
        base = base_url.rstrip("/")
        self.endpoint = f"{base}/completions" if base.endswith("/v1") else f"{base}/v1/completions"
        self.model = model
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max(8, concurrency * 2), max_keepalive_connections=max(8, concurrency)),
        )

    async def generate(self, prompt: str, config: Mapping[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(config["max_tokens"]),
            "temperature": float(config["temperature"]),
            "seed": int(config["seed"]),
            "stop": list(config["stop"]),
            "top_k": int(config["top_k"]),
            "repetition_penalty": float(config["repetition_penalty"]),
            # The OpenAI Python SDK expands extra_body into top-level vLLM
            # request fields; raw HTTP must do that explicitly.
            "include_stop_str_in_output": bool(config.get("include_stop_str_in_output", True)),
        }
        if config.get("top_p") is not None:
            payload["top_p"] = float(config["top_p"])
        response = await self.client.post(self.endpoint, json=payload)
        response.raise_for_status()
        choices = response.json().get("choices") or []
        if not choices:
            raise RuntimeError("completion response has no choices")
        return str(choices[0].get("text") or "")

    async def aclose(self) -> None:
        await self.client.aclose()


class RetrieverClient:
    def __init__(self, base_url: str, config: Mapping[str, Any], timeout: float, retries: int, concurrency: int) -> None:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The selected evaluation environment is missing httpx; "
                "use the JADE runtime or install requirements.txt"
            ) from exc
        self.config = config
        self.endpoint = urljoin(base_url.rstrip("/") + "/", str(config["retriever_path"]).lstrip("/"))
        self.format = str(config["retrieval_format"])
        self.prompt_family = str(config.get("prompt_family", ""))
        self.topk = int(config["retrieval_topk"])
        self.retries = retries
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max(8, concurrency * 2), max_keepalive_connections=max(8, concurrency)),
        )

    async def retrieve(self, query: str | list[str]) -> str:
        if self.format == "zerosearch_wiki":
            payload: dict[str, Any] = {"query": query, "top_k": self.topk}
        elif str(self.config.get("prompt_family")) == "o2_searcher":
            # O2 emits several independent queries, while the shared retriever
            # expects them as one flat batch rather than a nested list.
            queries = query if isinstance(query, list) else [query]
            payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        elif self.format == "o2_closedended":
            queries = query if isinstance(query, list) else [query]
            payload = {"queries": [queries], "topk": self.topk}
        else:
            payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = await self.client.post(self.endpoint, json=payload)
                response.raise_for_status()
                return self._format_documents(response.json(), query)
            except Exception as exc:  # retain the final exception for fail-closed row accounting
                last_error = exc
                if attempt + 1 < self.retries:
                    await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))
        raise RuntimeError(f"retriever failed after {self.retries} attempts: {last_error}") from last_error

    def _format_documents(self, payload: Any, query: str | list[str] | None = None) -> str:
        if self.prompt_family == "o2_searcher":
            # Match O2's official closed-ended wrapper: each query's top-k
            # contents are concatenated into one learning string, without the
            # Search-R1 Doc/Title prefixes used by other adapters.
            results = payload.get("result") if isinstance(payload, dict) else None
            if not isinstance(results, list):
                raise RuntimeError("O2 retriever returned an invalid response")
            query_count = len(query) if isinstance(query, list) else 1
            values: list[str] = []
            for query_results in results[:query_count]:
                if not isinstance(query_results, list):
                    continue
                contents = [
                    str((item.get("document") or {}).get("contents") or "")
                    for item in query_results[: self.topk]
                    if isinstance(item, dict)
                ]
                values.append(" ".join(content for content in contents if content))
            return "\n".join(values)
        if self.format == "o2_closedended":
            if not isinstance(payload, list) or not payload:
                raise RuntimeError("O2 closed-ended retriever returned an invalid response")
            return str(payload[0])
        if self.format == "zerosearch_wiki":
            documents = payload if isinstance(payload, list) else payload.get("result", [])
            return "\n".join(f"Doc {index}: {str(item.get('text', ''))}" for index, item in enumerate(documents, 1))
        results = payload.get("result") if isinstance(payload, dict) else None
        documents = results[0] if isinstance(results, list) and results else []
        values: list[str] = []
        for index, item in enumerate(documents, 1):
            content = str((item.get("document") or {}).get("contents") or "")
            title, _, text = content.partition("\n")
            values.append(f"Doc {index}(Title: {title}) {text}")
        return "\n".join(values)

    async def aclose(self) -> None:
        await self.client.aclose()


def load_tokenizer(path: Path | None) -> Any:
    if path is None:
        return None
    import transformers

    return transformers.AutoTokenizer.from_pretrained(str(path))


def format_initial_prompt(
    question: str,
    config: Mapping[str, Any],
    tokenizer: Any,
    search_results: str = "",
) -> str:
    normalized = question.strip()
    if not normalized.endswith("?"):
        normalized += "?"
    family = str(config["prompt_family"])
    if family == "rag":
        content = RAG_PROMPT.format(question=normalized, search_results=search_results)
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return content
    if family == "ircot":
        content = IRCOT_PROMPT.format(question=normalized)
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return content
    if family == "autorefine":
        content = AUTOREFINE_PROMPT.format(question=normalized)
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return content
    if family == "o2_searcher":
        messages = [
            {"role": "system", "content": O2_SYSTEM_PROMPT},
            {"role": "user", "content": "Initial query: " + normalized},
        ]
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return (
            f"<|im_start|>system\n{O2_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nInitial query: {normalized}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    if family == "search_r1_tool":
        messages = [
            {"role": "system", "content": REOPD_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": normalized},
        ]
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return (
            f"<|im_start|>system\n{REOPD_TOOL_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{normalized}<|im_end|>\n<|im_start|>assistant\n"
        )
    if family == "hiprag_short":
        content = f"{HIPRAG_SHORT_PROMPT}\nUser Question: {normalized}"
        suffix = "\n<think>\n<step>\n    <reasoning>"
    else:
        content = SEARCH_R1_PROMPT.format(question=normalized)
        suffix = ""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        content = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return content + suffix


def close_truncated_tag(text: str) -> str:
    for tag in ("search", "answer", "tool_call", "think"):
        opening = f"<{tag}>"
        closing = f"</{tag}>"
        if opening.lower() in text.lower() and closing.lower() not in text.lower():
            text += closing
            break
    return text


def normalize_answer_tags(text: str) -> str:
    """Accept only line-anchored answer-tag variants seen in model output.

    Some checkpoints emit ``answer`` or ``(answer)`` on its own line while
    still emitting the canonical closing tag.  Normalizing these narrow
    variants avoids treating a recoverable terminal format quirk as a
    pipeline failure; the raw output remains in the trace for auditing.
    """
    return re.sub(
        r"(?im)^(?P<indent>\s*)(?:<\s*|[([])?answer(?:\s*>|[)\]])?\s*$",
        r"\g<indent><answer>",
        text,
    )


def _tool_call_query(text: str) -> str | None:
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("name") != "search":
        return None
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    query = arguments.get("query")
    return str(query).strip() if query else None


def first_event(text: str, config: Mapping[str, Any] | None = None) -> tuple[str, Any] | None:
    candidates: list[tuple[int, str, str]] = []
    for kind, pattern in TAG_RE.items():
        if config and str(config.get("prompt_family")) == "o2_searcher" and kind == "search":
            continue
        match = pattern.search(text)
        if match:
            candidates.append((match.end(), kind, match.group(1).strip()))
    if config and str(config.get("prompt_family")) == "search_r1_tool":
        tool_match = TOOL_CALL_RE.search(text)
        query = _tool_call_query(text)
        if tool_match and query:
            candidates.append((tool_match.end(), "search", query))
        boxed = BOXED_RE.search(text)
        if boxed:
            candidates.append((boxed.end(), "boxed_answer", boxed.group(1).strip()))
    if config and str(config.get("prompt_family")) == "o2_searcher":
        search_match = re.search(r"<search>(.*?)</search>", text, re.DOTALL | re.IGNORECASE)
        answer_match = TAG_RE["answer"].search(text)
        if search_match:
            queries = [item.strip() for item in re.findall(r"<query>(.*?)</query>", search_match.group(1), re.DOTALL | re.IGNORECASE) if item.strip()]
            if queries:
                candidates.append((search_match.end(), "search", json.dumps(queries, ensure_ascii=False)))
        if answer_match:
            candidates.append((answer_match.end(), "answer", answer_match.group(1).strip()))
    if not candidates:
        return None
    _, kind, value = min(candidates, key=lambda item: item[0])
    return kind, value


def extract_last_answer(text: str, config: Mapping[str, Any] | None = None) -> str:
    matches = TAG_RE["answer"].findall(text)
    if matches:
        return matches[-1].strip()
    if config and str(config.get("prompt_family")) == "search_r1_tool":
        boxed = BOXED_RE.findall(text)
        if boxed:
            return boxed[-1].strip()
    return ""


async def run_one(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    tokenizer: Any,
    completion: CompletionClient,
    retriever: RetrieverClient,
) -> dict[str, Any]:
    question = str(row["question"])
    family = str(config.get("prompt_family", ""))
    if family == "rag":
        observation = await retriever.retrieve(question)
        prompt = format_initial_prompt(question, config, tokenizer, observation)
        request_config = dict(config)
        if tokenizer is not None:
            input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            context_budget = max(0, int(config.get("max_context_len", 8192)) - len(input_ids) - 16)
            if context_budget <= 0:
                return {"answer": "", "trace": [], "turns": 0, "search_calls": 1, "format_failure": True}
            request_config["max_tokens"] = min(int(config["max_tokens"]), context_budget)
        raw_output = close_truncated_tag(await completion.generate(prompt, request_config))
        output_text = normalize_answer_tags(raw_output)
        answer = extract_last_answer(output_text, config)
        return {
            "answer": answer,
            "trace": [{"turn": 1, "output": raw_output, "search_query": question, "observation": observation}],
            "turns": 1,
            "search_calls": 1,
            "format_failure": not bool(answer),
        }

    prompt = format_initial_prompt(question, config, tokenizer)
    transcript = ""
    trace: list[dict[str, Any]] = []
    search_calls = 0
    format_failure = False
    for turn in range(1, int(config["max_turns"]) + 1):
        request_config = dict(config)
        if tokenizer is not None:
            input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            context_budget = max(0, int(config.get("max_context_len", 8192)) - len(input_ids) - 16)
            if context_budget <= 0:
                format_failure = True
                break
            request_config["max_tokens"] = min(int(config["max_tokens"]), context_budget)
        raw_output = await completion.generate(prompt, request_config)
        raw_output = close_truncated_tag(raw_output)
        output_text = normalize_answer_tags(raw_output)
        transcript += output_text
        event = first_event(output_text, config)
        trace_item: dict[str, Any] = {
            "turn": turn,
            "output": raw_output,
            "search_query": None,
            "observation": None,
        }
        if output_text != raw_output:
            trace_item["normalized_output"] = output_text
        if family == "ircot":
            if event is not None and event[0] == "answer":
                trace.append(trace_item)
                return {
                    "answer": extract_last_answer(transcript, config),
                    "trace": trace,
                    "turns": turn,
                    "search_calls": search_calls,
                    "format_failure": False,
                }
            thoughts = re.findall(r"<think>(.*?)</think>", output_text, re.DOTALL | re.IGNORECASE)
            query = thoughts[-1].strip() if thoughts else ""
            if not query or search_calls >= int(config.get("max_retrieval_calls", 6)):
                format_failure = True
                trace.append(trace_item)
                break
            search_calls += 1
            observation = await retriever.retrieve(query)
            trace_item["search_query"] = query
            trace_item["observation"] = observation
            trace.append(trace_item)
            continuation_key = (
                "final_answer_continuation"
                if search_calls >= int(config.get("max_retrieval_calls", 6))
                else "continuation"
            )
            continuation = str(config.get(continuation_key, config["continuation"]))
            prompt += continuation.format(output_text=output_text, search_results=observation)
            continue
        if event is None:
            if str(config.get("prompt_family")) == "o2_searcher":
                # Match O2's official recovery path for an invalid/empty
                # response instead of failing before the remaining steps.
                continuation = (
                    "<|im_end|>\n<|im_start|>user\n"
                    + O2_ERROR_PROMPT
                    + "<|im_end|>\n<|im_start|>assistant\n"
                )
                prompt += output_text + continuation
                trace_item["continuation"] = O2_ERROR_PROMPT
                trace.append(trace_item)
                continue
            if config.get("continue_without_search", False):
                prompt += output_text + str(config["no_search_continuation"])
                trace_item["continuation"] = str(config["no_search_continuation"])
                trace.append(trace_item)
                continue
            format_failure = True
            trace.append(trace_item)
            break
        kind, value = event
        if kind in {"answer", "boxed_answer"}:
            trace.append(trace_item)
            return {
                "answer": extract_last_answer(transcript, config),
                "trace": trace,
                "turns": turn,
                "search_calls": search_calls,
                "format_failure": False,
            }
        search_calls += 1
        retrieval_query: str | list[str] = value
        if str(config.get("prompt_family")) == "o2_searcher":
            retrieval_query = json.loads(str(value))
            trace_item["search_query"] = retrieval_query
        else:
            trace_item["search_query"] = value
        observation = await retriever.retrieve(retrieval_query)
        trace_item["observation"] = observation
        trace.append(trace_item)
        prompt += str(config["continuation"]).format(output_text=output_text, search_results=observation)
    return {
        "answer": extract_last_answer(transcript, config),
        "trace": trace,
        "turns": len(trace),
        "search_calls": search_calls,
        "format_failure": True,
    }


async def score_answer(prediction: str, gold: list[str]) -> tuple[float, float]:
    from qa_manager.compute_metrics import compute_scores

    metrics = compute_scores([prediction], [gold])
    return float(metrics["em"]), float(metrics["f1"])


async def evaluate_dataset(
    dataset: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tokenizer: Any,
    completion: CompletionClient,
    retriever: RetrieverClient,
    output_dir: Path,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any] | None] = [None] * len(rows)

    async def worker(index: int, row: dict[str, Any]) -> None:
        async with semaphore:
            started = time.time()
            try:
                native = await run_one(row, config, tokenizer, completion, retriever)
                prediction = str(native["answer"])
                em, f1 = await score_answer(prediction, answer_values(row["golden_answers"]))
                results[index] = {
                    "dataset": dataset,
                    "sample_index": int(row["sample_index"]),
                    "question": row["question"],
                    "golden_answers": answer_values(row["golden_answers"]),
                    "prediction": prediction,
                    "em": em,
                    "f1": f1,
                    "pipeline_error": bool(native["format_failure"]),
                    "format_failure": bool(native["format_failure"]),
                    "turns": int(native["turns"]),
                    "search_calls": int(native["search_calls"]),
                    "trace": native["trace"],
                    "elapsed_seconds": time.time() - started,
                    "error": "format_failure" if native["format_failure"] else None,
                }
            except Exception as exc:
                results[index] = {
                    "dataset": dataset,
                    "sample_index": int(row["sample_index"]),
                    "question": row["question"],
                    "golden_answers": answer_values(row["golden_answers"]),
                    "prediction": "",
                    "em": 0.0,
                    "f1": 0.0,
                    "pipeline_error": True,
                    "format_failure": False,
                    "turns": 0,
                    "search_calls": 0,
                    "trace": [],
                    "elapsed_seconds": time.time() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    await asyncio.gather(*(worker(index, row) for index, row in enumerate(rows)))
    finalized = [row for row in results if row is not None]
    if len(finalized) != len(rows):
        raise RuntimeError(f"internal result cardinality error for {dataset}")
    output_path = output_dir / "trajectories" / f"{config['baseline']}_{dataset}.json"
    output_path.write_text(json.dumps(finalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = sum(bool(row["pipeline_error"]) for row in finalized)
    format_failures = sum(bool(row["format_failure"]) for row in finalized)
    print(f"[DATASET] {dataset} rows={len(finalized)} errors={errors} path={output_path}", flush=True)
    return {
        "rows": len(finalized),
        "pipeline_errors": errors,
        "format_failures": format_failures,
        "sha256": sha256(output_path),
    }


async def main_async(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    manifest, summary = load_manifest(args.manifest_dir)
    config = load_protocol(args)
    config["seed"] = args.seed
    tokenizer = load_tokenizer(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "trajectories").mkdir()
    run_manifest = {
        "schema_version": 1,
        "baseline": args.baseline,
        "model": args.model,
        "protocol": config,
        "manifest_dir": str(args.manifest_dir.resolve()),
        "manifest_summary_sha256": sha256(args.manifest_dir / "manifest_summary.json"),
        "manifest_total_rows": int(summary["total_selected_rows"]),
        "dataset_counts": {dataset: len(rows) for dataset, rows in manifest.items()},
        "retriever_url": args.retriever_url,
        "decoding": {
            "concurrency": args.concurrency,
            "seed": args.seed,
            "max_rows_per_dataset": args.max_rows,
        },
        "result_file_template": f"{args.baseline}_{{dataset}}.json",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    completion = CompletionClient(args.api_base_url, args.model, args.timeout, args.concurrency)
    retriever = RetrieverClient(args.retriever_url, config, args.timeout, args.retrieval_retries, args.concurrency)
    try:
        results: dict[str, Any] = {}
        for dataset in DATASETS:
            rows = manifest[dataset]
            if args.max_rows is not None:
                rows = rows[: args.max_rows]
            results[dataset] = await evaluate_dataset(
                dataset, rows, args, config, tokenizer, completion, retriever, args.output_dir
            )
        summary_output = {
            "schema_version": 1,
            "baseline": args.baseline,
            "model": args.model,
            "manifest_summary_sha256": run_manifest["manifest_summary_sha256"],
            "datasets": results,
            "total_rows": sum(item["rows"] for item in results.values()),
            "pipeline_error_rows": sum(item["pipeline_errors"] for item in results.values()),
            "format_failure_rows": sum(item["format_failures"] for item in results.values()),
            "complete_manifest": args.max_rows is None and sum(item["rows"] for item in results.values()) == 22_517,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary_output, ensure_ascii=False, indent=2), flush=True)
        if summary_output["pipeline_error_rows"]:
            raise SystemExit(1)
    finally:
        await completion.aclose()
        await retriever.aclose()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
