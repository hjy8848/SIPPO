#!/usr/bin/env python3
"""Evaluate the released StepSearch checkpoint on JADE's frozen four-split manifest.

This is a thin StepSearch-specific controller around the existing JADE external
baseline clients and scorer.  It preserves StepSearch's plan/observation prompt
and document rendering while using the Search-R1-shaped Wiki18 retriever API.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import re
import string
import time
from pathlib import Path
from typing import Any, Mapping

from evaluate_external_baseline_full_answerable import (
    DATASETS,
    CompletionClient,
    RetrieverClient,
    answer_values,
    close_truncated_tag,
    load_manifest,
    sha256,
)


STEPSEARCH_PROMPT = """## Background
You are a deep AI research assistant with search tool
 You should first think about your research plan or what to search for next. 

## Response format
1. You must make search plan inside <plan> and </plan> for in the beginning and after observation.
2. After plan, if you find you lack some knowledge, you can call a search engine by <search> search keyword </search> and it will return the search results between <information> and </information>.
3. You must conduct observation inside <observation> and </observation> for EVERY searched document, e.g.<observation>Based on retrieved inforamtion, Doc1 ....; Doc2...; Doc3 ...;</observation>
4. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>

Please follow the loop of plan, search, information, observation, plan ... until the you can answer original question.
 Question: {question}
"""

INVALID_ACTION = (
    "\n<warning> My previous action is invalid. If I want to search, I should put the query "
    "between <search> and </search>. If I want to give the final answer, I should put the answer "
    "between <answer> and </answer>. Let me try again. </warning>\n"
)
FINAL_SEARCH_WARNING = (
    "<warning> Now, I can no longer call <search></search> for external information. "
    "I must analyze and answer based on the existing information. "
    "Now let me make a final thought and give an answer. </warning>\n\n"
)


_CANONICAL_ACTION_RE = re.compile(
    r"<(?P<kind>search|answer)>(?P<value>.*?)</(?P=kind)>",
    re.DOTALL | re.IGNORECASE,
)
_OFFICIAL_ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
_LABELED_ACTION_RE = re.compile(
    r"(?im)^[ \t]*(?:\d+[.)]?[ \t]*)?"
    r"(?:[<(\[]\s*)?(?:\*\*)?(?P<kind>search|answer)(?:\*\*)?\s*(?:[)>\]])?"
    r"[ \t]*(?::|：)?[ \t]*"
)
_ACTION_CLOSE_RE = {
    "search": re.compile(r"</search>", re.IGNORECASE),
    "answer": re.compile(r"</answer>", re.IGNORECASE),
}


def _stepsearch_events(text: str) -> list[tuple[int, int, str, str]]:
    """Return canonical and observed StepSearch action variants in source order.

    The released checkpoint usually follows the XML protocol, but the full-run
    traces also contain forms such as ``(search) query </search>`` and
    ``answer: text\n</answer>``.  These are controller-level syntax variants,
    not new model actions, so we normalize only the first action and keep the
    raw response in the trace for auditability.
    """
    events: list[tuple[int, int, str, str]] = []
    for match in _CANONICAL_ACTION_RE.finditer(text):
        events.append((match.start(), match.end(), match.group("kind").lower(), match.group("value").strip()))

    for match in _LABELED_ACTION_RE.finditer(text):
        kind = match.group("kind").lower()
        value_start = match.end()
        remainder = text[value_start:]

        # Handle ``search: <search> query </search>`` without treating the
        # nested tag as a separate later action.
        nested = re.match(
            rf"[ \t]*<\s*{kind}\s*>(.*?)</\s*{kind}\s*>",
            remainder,
            re.DOTALL | re.IGNORECASE,
        )
        if nested:
            end = value_start + nested.end()
            events.append((match.start(), end, kind, nested.group(1).strip()))
            continue

        close = _ACTION_CLOSE_RE[kind].search(remainder)
        information = re.search(r"<information>|\n[ \t]*(?:information|observation)\s*:", remainder, re.IGNORECASE)
        newline = remainder.find("\n")
        boundaries = [position for position in (close.start() if close else -1,
                                                  information.start() if information else -1,
                                                  newline) if position >= 0]
        value_end = min(boundaries) if boundaries else len(remainder)
        value = remainder[:value_end].strip()
        end = value_start + value_end
        if close and close.start() == value_end:
            end = value_start + close.end()
        if value:
            events.append((match.start(), end, kind, value))

    return sorted(events, key=lambda item: (item[0], item[1]))


def stepsearch_first_event(text: str) -> tuple[str, str, str] | None:
    """Return ``(kind, value, action_text)`` for the first parsed action."""
    events = _stepsearch_events(text)
    if not events:
        return None
    start, end, kind, value = events[0]
    action_text = text[:end]
    if not re.search(rf"<\s*{kind}\s*>", action_text, re.IGNORECASE):
        action_text = text[:start] + f"<{kind}>{value}</{kind}>"
    return kind, value, action_text


def stepsearch_official_first_event(text: str) -> tuple[str, str, str] | None:
    """Apply StepSearch's strict, case-sensitive XML action parser."""
    match = _OFFICIAL_ACTION_RE.search(text)
    if match is None:
        return None
    return match.group(1), match.group(2).strip(), text[:match.end()]


class StepSearchRetriever(RetrieverClient):
    """Render JADE's scored Wiki18 response as StepSearch expects."""

    def _format_documents(self, payload: Any, query: str | list[str] | None = None) -> str:
        results = payload.get("result") if isinstance(payload, dict) else None
        documents = results[0] if isinstance(results, list) and results else []
        values: list[str] = []
        for index, item in enumerate(documents[: self.topk], 1):
            document = item.get("document") if isinstance(item, dict) else {}
            document = document if isinstance(document, dict) else {}
            contents = str(document.get("contents") or "")
            if contents:
                title, _, text = contents.partition("\n")
            else:
                title = str(document.get("title") or "")
                text = str(document.get("content") or document.get("text") or "")
            values.append(f"Doc {index}<## Title: {title} ##> {text}\n")
        return "".join(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retrieval-retries", type=int, default=3)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-total-rows", type=int, help="Smoke-test total across all four datasets")
    parser.add_argument("--parser-mode", choices=("strict", "recovery"), default="strict")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "MTCP-20260923-v6-stepsearch-native-eval",
        "baseline": "stepsearch",
        "official_repo": "https://github.com/Zillwang/StepSearch",
        "prompt_family": "stepsearch",
        "retriever_path": "/retrieve",
        "retrieval_format": "stepsearch",
        "retrieval_topk": 3,
        "max_turns": 5,
        "max_tokens": 800,
        "max_prompt_tokens": 4096,
        "max_observation_tokens": 800,
        "final_generation_after_turn_limit": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 42,
        "repetition_penalty": 1.0,
        "include_stop_str_in_output": True,
        "stop": [
            "</search>", " </search>", "</search>\n", " </search>\n",
            "</search>\n\n", " </search>\n\n",
            "</answer>", " </answer>", "</answer>\n", " </answer>\n",
        ],
        "invalid_continuation": INVALID_ACTION,
        "answer_metric": "official StepSearch eval.py normalization, exact match, and whitespace-token F1",
        "adapter_note": "StepSearch-native prompt, strict action parser, five-turn controller, rolling 4096-token prompt, 800-token response/observation limits, final generation, and temperature/top_p=1; shared JADE Wiki18/E5 top-k=3 retrieval backend.",
    }


def run_config(seed: int, parser_mode: str) -> dict[str, Any]:
    config = protocol()
    config["seed"] = int(seed)
    config["parser_mode"] = parser_mode
    return config


def format_prompt(question: str, tokenizer: Any = None) -> str:
    normalized = question.strip()
    if not normalized.endswith("?"):
        normalized += "?"
    instruction = STEPSEARCH_PROMPT.format(question=normalized)
    if tokenizer is not None and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            add_generation_prompt=True, tokenize=False,
        )
    return instruction


def truncate_text_tokens(text: str, tokenizer: Any, max_tokens: int, *, keep: str) -> str:
    """Apply StepSearch's token-level observation/prompt truncation to text."""
    if tokenizer is None:
        return text
    if keep not in {"head", "tail"}:
        raise ValueError(f"unsupported truncation direction: {keep}")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) <= max_tokens:
        return text
    kept_ids = ids[:max_tokens] if keep == "head" else ids[-max_tokens:]
    return tokenizer.decode(
        kept_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _official_normalize_answer(text: str) -> str:
    text = text.lower()
    punctuation = set(string.punctuation)
    text = "".join(char for char in text if char not in punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def official_answer_metrics(prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    """Match StepSearch's eval.py exact-match and whitespace-token F1."""
    gold_answers = answer_values(gold_answers)
    normalized_prediction = _official_normalize_answer(prediction)
    exact_match = float(any(
        normalized_prediction == _official_normalize_answer(gold)
        for gold in gold_answers
    ))
    prediction_tokens = normalized_prediction.split()
    f1_scores: list[float] = []
    for gold in gold_answers:
        gold_tokens = _official_normalize_answer(gold).split()
        overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
        if overlap == 0:
            f1_scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
        recall = overlap / len(gold_tokens) if gold_tokens else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall))
    return exact_match, max(f1_scores, default=0.0)


def write_compact_json(path: Path, value: Any) -> None:
    """Stream compact JSON to disk without building a second full-size string."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


async def run_one(row: Mapping[str, Any], config: Mapping[str, Any], tokenizer: Any,
                  completion: CompletionClient, retriever: StepSearchRetriever) -> dict[str, Any]:
    prompt = truncate_text_tokens(
        format_prompt(str(row["question"]), tokenizer),
        tokenizer,
        int(config["max_prompt_tokens"]),
        keep="tail",
    )
    trace: list[dict[str, Any]] = []
    search_calls = 0
    parse_action = (stepsearch_official_first_event
                    if config.get("parser_mode", "strict") == "strict"
                    else stepsearch_first_event)
    for turn in range(1, int(config["max_turns"]) + 1):
        prompt = truncate_text_tokens(
            prompt, tokenizer, int(config["max_prompt_tokens"]), keep="tail"
        )
        request_config = dict(config)
        raw_output = await completion.generate(prompt, request_config)
        if config.get("parser_mode", "strict") == "recovery":
            raw_output = close_truncated_tag(raw_output)
        parsed = parse_action(raw_output)
        trace_item: dict[str, Any] = {
            "turn": turn, "output": raw_output,
            "search_query": None, "observation": None,
        }
        if parsed is None:
            invalid_continuation = str(config["invalid_continuation"])
            if turn == int(config["max_turns"]):
                invalid_continuation += FINAL_SEARCH_WARNING
            invalid_continuation = truncate_text_tokens(
                invalid_continuation, tokenizer,
                int(config["max_observation_tokens"]), keep="head",
            )
            prompt = truncate_text_tokens(
                prompt + raw_output + invalid_continuation,
                tokenizer, int(config["max_prompt_tokens"]), keep="tail",
            )
            trace_item["continuation"] = invalid_continuation
            trace.append(trace_item)
            continue
        kind, value, action_text = parsed
        if action_text != raw_output:
            trace_item["normalized_output"] = action_text
        if kind in {"answer", "boxed_answer"}:
            trace.append(trace_item)
            return {
                "answer": value,
                "trace": trace, "turns": turn,
                "search_calls": search_calls, "format_failure": False,
            }
        search_calls += 1
        trace_item["search_query"] = value
        retrieval_results = await retriever.retrieve(value)
        observation = f"\n\n<information> {retrieval_results} </information>\n\n"
        if turn == int(config["max_turns"]):
            observation += FINAL_SEARCH_WARNING
        observation = truncate_text_tokens(
            observation, tokenizer,
            int(config["max_observation_tokens"]), keep="head",
        )
        trace_item["retrieval_results"] = retrieval_results
        trace_item["observation"] = observation
        trace.append(trace_item)
        prompt = truncate_text_tokens(
            prompt + action_text + observation,
            tokenizer, int(config["max_prompt_tokens"]), keep="tail",
        )

    # StepSearch performs one final generation after the tool-turn loop while
    # active trajectories receive the last-turn warning above.
    turn = int(config["max_turns"]) + 1
    request_config = dict(config)
    raw_output = await completion.generate(prompt, request_config)
    if config.get("parser_mode") == "recovery":
        raw_output = close_truncated_tag(raw_output)
    parsed = parse_action(raw_output)
    trace_item = {"turn": turn, "output": raw_output, "search_query": None,
                  "observation": None, "final_generation": True}
    if parsed is not None:
        kind, value, action_text = parsed
        if action_text != raw_output:
            trace_item["normalized_output"] = action_text
        if kind == "answer":
            trace.append(trace_item)
            return {"answer": value, "trace": trace, "turns": turn,
                    "search_calls": search_calls, "format_failure": False,
                    "failure_reason": None}
        trace_item["search_requested_after_limit"] = True
    else:
        trace_item["continuation"] = str(config["invalid_continuation"])
    trace.append(trace_item)
    return {
        "answer": "",
        "trace": trace, "turns": len(trace),
        "search_calls": search_calls,
        "format_failure": parsed is None,
        "failure_reason": "format_failure" if parsed is None else "search_after_turn_limit",
    }


async def main_async(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    if args.max_total_rows is not None and args.max_total_rows < len(DATASETS):
        raise ValueError("--max-total-rows must include at least one row per dataset")
    if args.max_rows is not None and args.max_total_rows is not None:
        raise ValueError("--max-rows and --max-total-rows are mutually exclusive")
    manifest, summary = load_manifest(args.manifest_dir)
    if args.max_total_rows is not None:
        quotas = {dataset: 0 for dataset in DATASETS}
        for index in range(args.max_total_rows):
            quotas[DATASETS[index % len(DATASETS)]] += 1
        if any(quotas[dataset] > len(manifest[dataset]) for dataset in DATASETS):
            raise ValueError("--max-total-rows exceeds available rows in a dataset")
        selected = {dataset: manifest[dataset][:quotas[dataset]] for dataset in DATASETS}
    else:
        selected = {
            dataset: rows[:args.max_rows] if args.max_rows is not None else rows
            for dataset, rows in manifest.items()
        }
    config = run_config(args.seed, args.parser_mode)
    tokenizer = None
    if args.tokenizer:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "trajectories").mkdir()
    run_manifest = {
        "schema_version": 1, "baseline": "stepsearch", "model": args.model,
        "protocol": config, "manifest_dir": str(args.manifest_dir.resolve()),
        "manifest_summary_sha256": sha256(args.manifest_dir / "manifest_summary.json"),
        "manifest_total_rows": int(summary["total_selected_rows"]),
        "dataset_counts": {dataset: len(rows) for dataset, rows in manifest.items()},
        "retriever_url": args.retriever_url,
        "decoding": {"concurrency": args.concurrency, "seed": args.seed,
                     "max_rows_per_dataset": args.max_rows, "max_total_rows": args.max_total_rows,
                     "parser_mode": args.parser_mode},
        "selected_counts": {dataset: len(rows) for dataset, rows in selected.items()},
        "result_file_template": "stepsearch_{dataset}.json",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completion = CompletionClient(args.api_base_url, args.model, args.timeout, args.concurrency)
    retriever = StepSearchRetriever(args.retriever_url, config, args.timeout, args.retrieval_retries, args.concurrency)
    try:
        results: dict[str, Any] = {}
        semaphore = asyncio.Semaphore(args.concurrency)
        output_rows = {dataset: [None] * len(selected[dataset]) for dataset in DATASETS}

        async def worker(dataset: str, index: int, row: dict[str, Any]) -> None:
            async with semaphore:
                started = time.time()
                try:
                    native = await run_one(row, config, tokenizer, completion, retriever)
                    prediction = str(native["answer"])
                    em, f1 = official_answer_metrics(prediction, answer_values(row["golden_answers"]))
                    output_rows[dataset][index] = {
                        "dataset": dataset, "sample_index": int(row["sample_index"]),
                        "question": row["question"], "golden_answers": answer_values(row["golden_answers"]),
                        "prediction": prediction, "em": em, "f1": f1,
                        "pipeline_error": False,
                        "format_failure": bool(native["format_failure"]),
                        "failure_reason": native.get("failure_reason"),
                        "turns": int(native["turns"]), "search_calls": int(native["search_calls"]),
                        "trace": native["trace"], "elapsed_seconds": time.time() - started,
                        "error": None,
                    }
                except Exception as exc:
                    output_rows[dataset][index] = {
                        "dataset": dataset, "sample_index": int(row["sample_index"]),
                        "question": row["question"], "golden_answers": answer_values(row["golden_answers"]),
                        "prediction": "", "em": 0.0, "f1": 0.0, "pipeline_error": True,
                        "format_failure": False, "failure_reason": None,
                        "turns": 0, "search_calls": 0, "trace": [],
                        "elapsed_seconds": time.time() - started,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        await asyncio.gather(*(
            worker(dataset, index, row)
            for dataset in DATASETS for index, row in enumerate(selected[dataset])
        ))
        for dataset in DATASETS:
            finalized = [row for row in output_rows[dataset] if row is not None]
            path = args.output_dir / "trajectories" / f"stepsearch_{dataset}.json"
            write_compact_json(path, finalized)
            errors = sum(bool(row["pipeline_error"]) for row in finalized)
            format_failures = sum(bool(row["format_failure"]) for row in finalized)
            no_answers = sum(not bool(row["prediction"]) for row in finalized)
            results[dataset] = {"rows": len(finalized), "pipeline_errors": errors,
                                "format_failures": format_failures,
                                "no_answer_rows": no_answers}
            print(f"[DATASET] {dataset} rows={len(finalized)} errors={errors} path={path}", flush=True)
        summary_output = {
            "schema_version": 1, "baseline": "stepsearch", "model": args.model,
            "manifest_summary_sha256": run_manifest["manifest_summary_sha256"], "datasets": results,
            "total_rows": sum(item["rows"] for item in results.values()),
            "pipeline_error_rows": sum(item["pipeline_errors"] for item in results.values()),
            "format_failure_rows": sum(item["format_failures"] for item in results.values()),
            "no_answer_rows": sum(item["no_answer_rows"] for item in results.values()),
            "complete_manifest": args.max_rows is None and args.max_total_rows is None
            and sum(item["rows"] for item in results.values()) == 22517,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary_output, ensure_ascii=False, indent=2), flush=True)
        if summary_output["pipeline_error_rows"]:
            raise SystemExit(1)
    finally:
        await completion.aclose()
        await retriever.aclose()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
