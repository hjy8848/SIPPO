#!/usr/bin/env python3
"""Probe a ROSA retriever endpoint on 2Wiki data."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_manager.normalize_answers import normalize_answer


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def extract_question(row: pd.Series) -> str:
    extra = row.get("extra_info") or {}
    if isinstance(extra, dict) and extra.get("question"):
        return str(extra["question"])
    prompt = row.get("prompt")
    for item in to_list(prompt):
        if isinstance(item, dict) and item.get("content"):
            return str(item["content"])
    reward_model = row.get("reward_model") or {}
    if isinstance(reward_model, dict) and reward_model.get("question"):
        return str(reward_model["question"])
    return ""


def extract_answers(row: pd.Series) -> list[str]:
    extra = row.get("extra_info") or {}
    answers = []
    if isinstance(extra, dict):
        answers.extend(to_list(extra.get("answer")))
    reward_model = row.get("reward_model") or {}
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth") or {}
        if isinstance(gt, dict):
            answers.extend(to_list(gt.get("target")))
    return [str(answer).strip() for answer in answers if str(answer).strip()]


def extract_docs(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        if "result" in payload:
            result = payload["result"]
            if isinstance(result, list) and result and isinstance(result[0], list):
                return extract_docs(result[0])
            return extract_docs(result)
        if "documents" in payload:
            return extract_docs(payload["documents"])
        if "document" in payload:
            document = payload["document"]
            if isinstance(document, dict):
                return [str(document.get("contents", document.get("text", document)))]
            return [str(document)]
        if "contents" in payload:
            return [str(payload["contents"])]
        if "text" in payload:
            return [str(payload["text"])]
    if isinstance(payload, list):
        docs: list[str] = []
        for item in payload:
            docs.extend(extract_docs(item))
        return docs
    if payload is None:
        return []
    return [str(payload)]


def answer_hit(docs: list[str], answers: list[str]) -> bool | None:
    normalized_answers = [
        normalize_answer(answer)
        for answer in answers
        if normalize_answer(answer) not in {"yes", "no", "noanswer"}
    ]
    if not normalized_answers:
        return None
    joined_docs = normalize_answer("\n".join(docs))
    return any(answer and answer in joined_docs for answer in normalized_answers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Parquet data file.")
    parser.add_argument("--retriever-url", required=True, help="Retriever POST endpoint.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", default="retriever_diagnosis.jsonl")
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    rng = random.Random(args.seed)
    indices = list(range(len(frame)))
    rng.shuffle(indices)
    indices = indices[: min(args.sample, len(indices))]

    rows = []
    ok = nonempty = api_error = no_docs = answer_eval = answer_hits = 0
    latencies = []

    for order, idx in enumerate(indices, start=1):
        row = frame.iloc[idx]
        question = extract_question(row)
        answers = extract_answers(row)
        started = time.time()
        error = ""
        payload = None
        docs: list[str] = []
        try:
            resp = requests.post(
                args.retriever_url,
                json={"queries": [question], "topk": args.topk, "return_scores": True},
                timeout=args.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            docs = extract_docs(payload)
            ok += 1
        except Exception as exc:
            error = str(exc)
            api_error += 1

        latency = time.time() - started
        latencies.append(latency)
        has_docs = bool([doc for doc in docs if doc.strip()])
        if has_docs:
            nonempty += 1
        else:
            no_docs += 1

        hit = answer_hit(docs, answers)
        if hit is not None:
            answer_eval += 1
            answer_hits += int(hit)

        rows.append(
            {
                "order": order,
                "row_index": int(idx),
                "question": question,
                "answers": answers,
                "ok": not error,
                "error": error,
                "num_docs": len(docs),
                "answer_hit": hit,
                "latency_sec": latency,
                "doc_previews": [doc[:600] for doc in docs[: args.topk]],
                "raw_payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else None,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    print(f"data={args.data}")
    print(f"retriever_url={args.retriever_url}")
    print(f"sample={total}, topk={args.topk}")
    print(f"request_ok={ok}/{total} = {ok / total if total else 0:.3f}")
    print(f"api_error={api_error}/{total} = {api_error / total if total else 0:.3f}")
    print(f"nonempty_docs={nonempty}/{total} = {nonempty / total if total else 0:.3f}")
    print(f"no_docs={no_docs}/{total} = {no_docs / total if total else 0:.3f}")
    if answer_eval:
        print(f"answer_hit@{args.topk}={answer_hits}/{answer_eval} = {answer_hits / answer_eval:.3f}")
    else:
        print("answer_hit skipped: sampled answers were yes/no/noanswer only or missing")
    if latencies:
        print(f"latency_mean_sec={sum(latencies) / len(latencies):.3f}")
        print(f"latency_max_sec={max(latencies):.3f}")
    print(f"wrote={out_path.resolve()}")


if __name__ == "__main__":
    main()
