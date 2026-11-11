#!/usr/bin/env python3
"""Run a controlled Search-o1 reimplementation or its frozen 50-question smoke.

The prompt/controller and Reason-in-Documents stage follow the upstream
Search-o1 code. Only the search backend is adapted to the pinned Wiki18/E5
retriever used by the JADE controlled evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from qa_manager.compute_metrics import compute_scores
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from search_o1_official_prompts import (
    get_multiqa_search_o1_instruction,
    get_task_instruction_openqa,
    get_webpage_to_reasonchain_instruction,
)


DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "bamboogle")
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
SEARCH_RE = re.compile(re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY), re.S)
BOXED_RE = re.compile(r"\\boxed\{(.*)\}", re.S)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--full-eval", action="store_true", help="evaluate every row in the frozen manifest")
    parser.add_argument("--chunk-size", type=int, default=256, help="rows kept live in memory at once")
    parser.add_argument("--preflight-only", action="store_true", help="validate manifest and print row counts without loading a model")
    parser.add_argument("--seed", type=int, default=42, help="vLLM sampling seed for this evaluation replicate")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-search-limit", type=int, default=10)
    parser.add_argument("--max-turn", type=int, default=15)
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument("--max-doc-len", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--generation-top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reason-max-tokens", type=int, default=8192)
    parser.add_argument("--retrieval-retries", type=int, default=3)
    return parser.parse_args()


def answer_values(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def load_rows(root: Path, limit: int | None) -> list[dict[str, Any]]:
    summary_path = root / "manifest_summary.json"
    manifest_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(manifest_summary["total_selected_rows"]) != 22517:
        raise RuntimeError("unexpected full manifest total")
    quotas: dict[str, int] | None = None
    if limit == 50:
        quotas = {"2wikimultihopqa": 14, "hotpotqa": 13, "musique": 13, "bamboogle": 10}
    elif limit is not None:
        base = limit // len(DATASETS)
        quotas = {name: base for name in DATASETS}
        for name in DATASETS[: limit - base * len(DATASETS)]:
            quotas[name] += 1
    selected: list[dict[str, Any]] = []
    for dataset in DATASETS:
        parquet_path = root / f"{dataset}.parquet"
        if sha256(parquet_path) != manifest_summary["datasets"][dataset]["parquet_sha256"]:
            raise RuntimeError(f"manifest checksum mismatch: {parquet_path}")
        rows = pq.read_table(parquet_path, use_threads=False).to_pylist()
        expected = int(manifest_summary["datasets"][dataset]["selected_rows"])
        if len(rows) != expected:
            raise RuntimeError(f"manifest row count mismatch for {dataset}: {len(rows)} != {expected}")
        if quotas is None:
            chosen = sorted(rows, key=lambda row: int(row["sample_index"]))
        else:
            ranked = sorted(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{dataset}:{row['sample_index']}:{row['question']}".encode("utf-8")
                ).hexdigest(),
            )
            chosen = ranked[: quotas[dataset]]
        for row in chosen:
            selected.append(
                {
                    "dataset": dataset,
                    "sample_index": int(row["sample_index"]),
                    "question": str(row["question"]),
                    "golden_answers": answer_values(row["golden_answers"]),
                }
            )
    return sorted(selected, key=lambda row: (DATASETS.index(row["dataset"]), row["sample_index"]))


def extract_answer(text: str) -> str:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def extract_info(text: str) -> str:
    marker = "**Final Information**"
    if marker in text:
        return text.split(marker)[-1].replace("\n", "").strip("`").strip()
    marker = "**Modified Reasoning Steps**"
    if marker in text:
        return text.split(marker)[-1].strip("`").strip()
    return "No helpful information found."


def initial_prompt(tokenizer: Any, question: str, max_search_limit: int) -> str:
    content = get_multiqa_search_o1_instruction(max_search_limit) + get_task_instruction_openqa(question)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def document_text(result: Any, max_doc_len: int) -> str:
    if not isinstance(result, dict):
        return ""
    document = result.get("document") or {}
    contents = str(document.get("contents") or "")
    title, _, body = contents.partition("\n")
    body = body[:max_doc_len]
    return f"Title: {title}\n{body}".strip()


def format_documents(results: list[Any], max_doc_len: int) -> str:
    blocks = []
    for index, item in enumerate(results, 1):
        document = item.get("document") if isinstance(item, dict) else None
        score = item.get("score") if isinstance(item, dict) else None
        record = {
            "title": str((document or {}).get("title") or ""),
            "context": document_text(item, max_doc_len),
            "score": score,
        }
        blocks.append(f"**Web Page {index}:**\n{json.dumps(record, ensure_ascii=False, indent=2)}")
    return "\n".join(blocks)


def truncate_reasoning(output: str) -> str:
    steps = output.replace("\n\n", "\n").split("\n")
    numbered = "".join(f"Step {index + 1}: {step}\n\n" for index, step in enumerate(steps))
    previous_steps = numbered.split("\n\n")
    if len(previous_steps) <= 5:
        return "\n\n".join(previous_steps).strip()
    retained = ""
    for index, step in enumerate(previous_steps):
        if index == 0 or index >= len(previous_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
            retained += step + "\n\n"
        elif retained[-len("\n\n...\n\n"):] != "\n\n...\n\n":
            retained += "...\n\n"
    return retained.strip("\n")


def retrieve_batch(client: httpx.Client, url: str, queries: list[str], topk: int, retries: int) -> list[list[Any]]:
    payload = {"queries": queries, "topk": topk, "return_scores": True}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json().get("result")
            if not isinstance(result, list) or len(result) != len(queries):
                raise RuntimeError("retriever returned an invalid result cardinality")
            return [item if isinstance(item, list) else [] for item in result]
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"retriever failed after {retries} attempts: {last_error}")


def wait_for_retriever(client: httpx.Client, url: str, topk: int, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = retrieve_batch(client, url, ["capital of France"], topk, retries=1)
            print(f"[RETRIEVER_READY] probe_documents={len(result[0])}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(5)
    raise TimeoutError(f"retriever did not become ready within {timeout_seconds}s: {last_error}")


def generate(llm: Any, prompts: list[str], tokenizer: Any, args: argparse.Namespace, *, reason: bool = False) -> list[str]:
    params = SamplingParams(
        max_tokens=args.reason_max_tokens if reason else args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.generation_top_k,
        repetition_penalty=1.05 if reason else 1.0,
        stop=[tokenizer.eos_token] if reason else [END_SEARCH_QUERY, tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    return [item.outputs[0].text for item in outputs]


def score_row(row: dict[str, Any]) -> None:
    prediction = extract_answer(row["output"])
    metrics = compute_scores([prediction], [row["golden_answers"]])
    row["prediction"] = prediction
    row["em"] = float(metrics["em"])
    row["f1"] = float(metrics["f1"])


def make_sequence(row: dict[str, Any], tokenizer: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        **row,
        "prompt": initial_prompt(tokenizer, row["question"], args.max_search_limit),
        "output": "",
        "trace": [],
        "search_calls": 0,
        "turn": 0,
        "finished": False,
        "pipeline_error": False,
        "format_failure": False,
        "executed_search_queries": set(),
        "error": None,
    }


def compact_result(item: dict[str, Any], row_index: int, elapsed: float, seed: int) -> dict[str, Any]:
    trace = []
    for event in item["trace"]:
        docs = event.get("documents") or []
        trace.append(
            {
                "turn": event["turn"],
                "raw_output": event.get("raw_output", ""),
                "search_query": event.get("search_query"),
                "retrieved_documents": [
                    {
                        "title": str(((doc.get("document") or {}).get("title") or "")),
                        "score": doc.get("score"),
                    }
                    for doc in docs
                    if isinstance(doc, dict)
                ],
                "reason_output": event.get("reason_output"),
                "controller_message": event.get("controller_message"),
                "retrieval_error": event.get("retrieval_error"),
            }
        )
    record = {
        "dataset": item["dataset"],
        "seed": seed,
        "sample_index": item["sample_index"],
        "question": item["question"],
        "golden_answers": item["golden_answers"],
        "output": item["output"],
        "prediction": item["prediction"],
        "em": item["em"],
        "f1": item["f1"],
        "trace": trace,
        "search_calls": item["search_calls"],
        "turn": item["turn"],
        "pipeline_error": item["pipeline_error"],
        "format_failure": item["format_failure"],
        "executed_search_queries": sorted(item["executed_search_queries"]),
        "error": item["error"],
        "progress_row_index": row_index,
        "elapsed_seconds": elapsed,
    }
    return record


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("batch-size and chunk-size must be positive")
    if args.full_eval and args.chunk_size < args.batch_size:
        raise ValueError("full-eval chunk-size must be at least batch-size")
    if not args.full_eval and args.limit != 50:
        raise ValueError("sample mode is frozen to the 50-question smoke; use --full-eval for the complete manifest")
    if args.output_dir.exists() and not args.preflight_only:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = load_rows(args.manifest_dir, None if args.full_eval else args.limit)
    if args.full_eval and len(rows) != 22517:
        raise RuntimeError(f"full-eval manifest row count mismatch: {len(rows)} != 22517")
    if args.preflight_only:
        counts = {name: sum(row["dataset"] == name for row in rows) for name in DATASETS}
        print(json.dumps({"status": "passed", "rows": len(rows), "per_dataset": counts}, indent=2))
        return

    process_started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress_file = (args.output_dir / "progress.jsonl").open("x", encoding="utf-8")
    model_init_started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        seed=args.seed,
    )
    model_init_seconds = time.time() - model_init_started
    evaluation_started = time.time()
    aggregate = {
        "rows": 0,
        "pipeline_errors": 0,
        "format_failures": 0,
        "em_sum": 0.0,
        "f1_sum": 0.0,
        "search_calls": 0,
        "per_dataset": {
            name: {"rows": 0, "em_sum": 0.0, "f1_sum": 0.0, "pipeline_errors": 0}
            for name in DATASETS
        },
    }
    full_result_file = None
    all_sample_results: list[dict[str, Any]] = []
    if args.full_eval:
        full_result_file = (args.output_dir / "results.jsonl").open("x", encoding="utf-8")

    def persist(item: dict[str, Any], row_index: int) -> None:
        score_row(item)
        elapsed = time.time() - evaluation_started
        if args.full_eval:
            record = compact_result(item, row_index, elapsed, args.seed)
            assert full_result_file is not None
            full_result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            full_result_file.flush()
        else:
            record = {key: value for key, value in item.items() if key not in {"prompt", "finished"}}
            record["progress_row_index"] = row_index
            record["seed"] = args.seed
            record["executed_search_queries"] = sorted(item["executed_search_queries"])
            record["elapsed_seconds"] = elapsed
            progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress_file.flush()
            all_sample_results.append(record)
        dataset_stats = aggregate["per_dataset"][item["dataset"]]
        aggregate["rows"] += 1
        aggregate["pipeline_errors"] += int(bool(item["pipeline_error"]))
        aggregate["format_failures"] += int(bool(item["format_failure"]))
        aggregate["em_sum"] += item["em"]
        aggregate["f1_sum"] += item["f1"]
        aggregate["search_calls"] += item["search_calls"]
        dataset_stats["rows"] += 1
        dataset_stats["em_sum"] += item["em"]
        dataset_stats["f1_sum"] += item["f1"]
        dataset_stats["pipeline_errors"] += int(bool(item["pipeline_error"]))

    with httpx.Client(timeout=120.0) as client:
        wait_for_retriever(client, args.retriever_url, args.retrieval_topk)
        for chunk_start in range(0, len(rows), args.chunk_size if args.full_eval else args.limit):
            chunk_rows = rows[chunk_start : chunk_start + (args.chunk_size if args.full_eval else args.limit)]
            sequences = [make_sequence(row, tokenizer, args) for row in chunk_rows]
            persisted_rows: set[int] = set()
            while True:
                active = [item for item in sequences if not item["finished"]]
                if not active:
                    break
                print(
                    f"[TURN] completed={aggregate['rows']}/{len(rows)} active_rows={len(active)} "
                    f"batch_size={args.batch_size} chunk_size={len(sequences)}",
                    flush=True,
                )
                for start in range(0, len(active), args.batch_size):
                    batch = active[start : start + args.batch_size]
                    for item in batch:
                        item["turn"] += 1
                    outputs = generate(llm, [item["prompt"] for item in batch], tokenizer, args)
                    search_batch = []
                    search_items = []
                    for item, text in zip(batch, outputs):
                        item["output"] += text
                        item["prompt"] += text
                        event = {
                            "turn": item["turn"],
                            "raw_output": text,
                            "search_query": None,
                            "documents": [],
                        }
                        if not args.full_eval:
                            event["generated_prompt"] = item["prompt"][:-len(text)] if text else item["prompt"]
                        match = SEARCH_RE.search(text)
                        query = match.group(1).strip() if match else None
                        if query and text.rstrip().endswith(END_SEARCH_QUERY) and query not in item["executed_search_queries"] and item["search_calls"] < args.max_search_limit:
                            event["search_query"] = query
                            search_batch.append(query)
                            search_items.append((item, event))
                            item["executed_search_queries"].add(query)
                        elif query and text.rstrip().endswith(END_SEARCH_QUERY) and item["search_calls"] >= args.max_search_limit:
                            message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            item["prompt"] += message
                            item["output"] += message
                            event["controller_message"] = "maximum_search_limit_exceeded"
                        elif query and text.rstrip().endswith(END_SEARCH_QUERY):
                            message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                            item["prompt"] += message
                            item["output"] += message
                            event["controller_message"] = "duplicate_search_query"
                        else:
                            item["finished"] = True
                            if not query and not extract_answer(item["output"]):
                                item["format_failure"] = True
                                item["error"] = "no_final_answer"
                        item["trace"].append(event)
                    if search_batch:
                        try:
                            retrieved = retrieve_batch(client, args.retriever_url, search_batch, args.retrieval_topk, args.retrieval_retries)
                        except Exception as exc:
                            for item, event in search_items:
                                item["finished"] = True
                                item["pipeline_error"] = True
                                item["error"] = f"{type(exc).__name__}: {exc}"
                                event["retrieval_error"] = item["error"]
                            continue
                        reason_prompts = []
                        for (item, event), docs in zip(search_items, retrieved):
                            item["search_calls"] += 1
                            event["documents"] = (
                                [
                                    {
                                        "document": {
                                            "title": str(((doc.get("document") or {}).get("title") or ""))
                                        },
                                        "score": doc.get("score"),
                                    }
                                    for doc in docs
                                    if isinstance(doc, dict)
                                ]
                                if args.full_eval
                                else docs
                            )
                            formatted = format_documents(docs, args.max_doc_len)
                            prior = truncate_reasoning(item["output"])
                            reason_content = get_webpage_to_reasonchain_instruction(prior, event["search_query"], formatted)
                            reason_prompts.append(
                                tokenizer.apply_chat_template(
                                    [{"role": "user", "content": reason_content}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                            )
                            if not args.full_eval:
                                event["reason_prompt"] = reason_prompts[-1]
                        analyses = generate(llm, reason_prompts, tokenizer, args, reason=True)
                        for (item, _event), analysis in zip(search_items, analyses):
                            info = extract_info(analysis)
                            append_text = f"\n\n{BEGIN_SEARCH_RESULT}{info}{END_SEARCH_RESULT}\n\n"
                            item["prompt"] += append_text
                            item["output"] += append_text
                            item["trace"][-1]["reason_output"] = analysis
                for item in sequences:
                    if not item["finished"] and item["turn"] >= args.max_turn:
                        item["finished"] = True
                        item["format_failure"] = not bool(extract_answer(item["output"]))
                        item["error"] = "max_turn_exceeded_without_final_answer" if item["format_failure"] else None
                for local_index, item in enumerate(sequences):
                    if not item["finished"] or local_index in persisted_rows:
                        continue
                    persist(item, chunk_start + local_index)
                    persisted_rows.add(local_index)
                progress_file.write(
                    json.dumps(
                        {
                            "completed_rows": aggregate["rows"],
                            "total_rows": len(rows),
                            "pipeline_errors": aggregate["pipeline_errors"],
                            "format_failures": aggregate["format_failures"],
                            "elapsed_seconds": time.time() - evaluation_started,
                        }
                    )
                    + "\n"
                )
                progress_file.flush()
                print(f"[PROGRESS] completed_rows={aggregate['rows']}/{len(rows)}", flush=True)
            del sequences

    progress_file.close()
    if full_result_file is not None:
        full_result_file.close()
    if args.full_eval:
        per_dataset = {
            name: {
                "rows": values["rows"],
                "em": values["em_sum"] / values["rows"] if values["rows"] else 0.0,
                "f1": values["f1_sum"] / values["rows"] if values["rows"] else 0.0,
                "pipeline_errors": values["pipeline_errors"],
            }
            for name, values in aggregate["per_dataset"].items()
        }
    else:
        per_dataset = {}
        for dataset in DATASETS:
            subset = [row for row in all_sample_results if row["dataset"] == dataset]
            per_dataset[dataset] = {
                "rows": len(subset),
                "em": sum(row["em"] for row in subset) / len(subset),
                "f1": sum(row["f1"] for row in subset) / len(subset),
                "pipeline_errors": sum(bool(row["pipeline_error"]) for row in subset),
            }
        save_json(args.output_dir / "search_o1_qwen25_3b_50.json", all_sample_results)
    summary = {
        "rows": aggregate["rows"],
        "seed": args.seed,
        "pipeline_errors": aggregate["pipeline_errors"],
        "format_failures": aggregate["format_failures"],
        "em": aggregate["em_sum"] / aggregate["rows"],
        "f1": aggregate["f1_sum"] / aggregate["rows"],
        "macro_em": sum(value["em"] for value in per_dataset.values()) / len(DATASETS),
        "macro_f1": sum(value["f1"] for value in per_dataset.values()) / len(DATASETS),
        "search_calls": aggregate["search_calls"],
        "elapsed_seconds": time.time() - evaluation_started,
        "model_init_seconds": model_init_seconds,
        "end_to_end_seconds": time.time() - process_started,
        "per_dataset": per_dataset,
    }
    protocol = {
        "schema_version": 1,
        "baseline": "search_o1_controlled_reimplementation_qwen25_3b",
        "official_repo": "https://github.com/RUC-NLPIR/Search-o1",
        "official_prompt_snapshot_sha256": "200177e1bda361bf94046efd7ec2ca1e9ae7f52bdb7e21fd05491a04c9f5e319",
        "official_prompt": "multiqa_search_o1_instruction + task_instruction_openqa",
        "official_controller": "interleaved search query + Reason-in-Documents continuation",
        "backbone": str(args.model_path),
        "retriever_url": args.retriever_url,
        "retrieval_topk": args.retrieval_topk,
        "max_search_limit": args.max_search_limit,
        "max_turn": args.max_turn,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p, "generation_top_k": args.generation_top_k, "repetition_penalty": 1.0},
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size if args.full_eval else None,
        "seed": args.seed,
        "evaluation_scope": "full_manifest" if args.full_eval else "stable_hash_50_question_smoke",
        "manifest_dir": str(args.manifest_dir),
        "manifest_summary_sha256": sha256(args.manifest_dir / "manifest_summary.json"),
        "selection": "all manifest rows sorted by dataset and sample_index" if args.full_eval else "stable SHA-256 ranking within each dataset; quotas 14/13/13/10",
        "adapter_boundary": "only the external search backend is replaced by pinned Wiki18/E5 top-3; no DAGENT planner, node memory, QueryRewrite, PSC, or answer repair",
        "scorer": "JADE shared EM/F1 scorer (qa_manager.compute_metrics.compute_scores)",
        "status": "full-manifest controlled reimplementation; not the official QwQ-32B Search-o1 result" if args.full_eval else "50-question smoke; not an official QwQ-32B Search-o1 result",
        "decoding_repetition_penalty": {"search_generation": 1.0, "reason_generation": 1.05},
    }
    save_json(args.output_dir / "retriever_preflight.json", {"status": "passed", "query": "capital of France", "topk": args.retrieval_topk})
    save_json(args.output_dir / "protocol.json", protocol)
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
