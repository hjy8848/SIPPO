#!/usr/bin/env python3
"""Add retriever-aware gold titles without modifying official annotations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_manager.normalize_answers import normalize_answer


_LOCAL = threading.local()


def to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return to_python(value.as_py())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        return to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    return value


def unique_text(value: Any) -> list[str]:
    value = to_python(value)
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    output = []
    for item in value:
        text = str(item).strip().strip('"')
        if text and text not in output:
            output.append(text)
    return output


def session() -> requests.Session:
    value = getattr(_LOCAL, "session", None)
    if value is None:
        value = requests.Session()
        _LOCAL.session = value
    return value


def retrieve_batch(
    url: str,
    titles: list[str],
    topk: int,
    timeout: float,
    retries: int,
) -> dict[str, int | None]:
    for attempt in range(retries + 1):
        try:
            response = session().post(
                url,
                json={"queries": titles, "topk": topk, "return_scores": True},
                timeout=timeout,
            )
            response.raise_for_status()
            batches = response.json().get("result") or []
            if len(batches) != len(titles):
                raise RuntimeError(
                    f"retriever returned {len(batches)} batches for {len(titles)} titles"
                )
            result: dict[str, int | None] = {}
            for title, documents in zip(titles, batches, strict=True):
                target = normalize_answer(title)
                rank = None
                for index, item in enumerate(documents, start=1):
                    contents = str((item.get("document") or {}).get("contents", "")).strip()
                    retrieved_title = contents.split("\n", 1)[0].strip().strip('"')
                    if normalize_answer(retrieved_title) == target:
                        rank = index
                        break
                result[title] = rank
            return result
        except (requests.RequestException, RuntimeError, ValueError):
            if attempt >= retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable retrieval retry state")


def load_cache(path: Path, url: str, topk: int) -> dict[str, int | None]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("retriever_url") != url or int(payload.get("topk", -1)) != topk:
        raise ValueError(
            f"cache settings differ from this run: {path}; use a different cache file"
        )
    return {
        str(title): (None if rank is None else int(rank))
        for title, rank in payload.get("title_ranks", {}).items()
    }


def save_cache(path: Path, url: str, topk: int, ranks: dict[str, int | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "retriever_url": url,
                "topk": topk,
                "title_ranks": dict(sorted(ranks.items())),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def enrich_frame(
    frame: pd.DataFrame,
    ranks: dict[str, int | None],
    topk: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    extra_infos = []
    statistics: dict[str, Counter] = defaultdict(Counter)
    retained_counts: dict[str, list[int]] = defaultdict(list)

    for _, row in frame.iterrows():
        source = str(row.get("data_source", "unknown"))
        info = to_python(row.get("extra_info", {}))
        info = info if isinstance(info, dict) else {}
        official = unique_text(info.get("golden_titles", []))
        retrievable = [title for title in official if ranks.get(title) is not None]
        info["retrievable_golden_titles"] = retrievable
        info["gold_retriever_filter"] = {
            "method": "exact_normalized_title_self_query",
            "topk": topk,
        }
        extra_infos.append(info)

        statistics[source]["rows"] += 1
        statistics[source]["rows_with_official_gold"] += bool(official)
        statistics[source]["rows_with_retrievable_gold"] += bool(retrievable)
        statistics[source]["official_titles"] += len(official)
        statistics[source]["retrievable_titles"] += len(retrievable)
        retained_counts[source].append(len(retrievable))

    output["extra_info"] = extra_infos
    report = {}
    for source, counts in sorted(statistics.items()):
        rows = max(1, counts["rows"])
        official_titles = max(1, counts["official_titles"])
        report[source] = {
            **dict(counts),
            "row_coverage": counts["rows_with_retrievable_gold"] / rows,
            "title_retention": counts["retrievable_titles"] / official_titles,
            "mean_retrievable_titles": sum(retained_counts[source]) / rows,
        }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    if len(args.input) != len(args.output):
        parser.error("--input and --output must be provided the same number of times")
    if args.topk <= 0 or args.batch_size <= 0 or args.concurrency <= 0:
        parser.error("--topk, --batch-size, and --concurrency must be positive")

    frames = [pd.read_parquet(path) for path in args.input]
    all_titles = sorted(
        {
            title
            for frame in frames
            for info in frame["extra_info"]
            for title in unique_text(to_python(info).get("golden_titles", []))
        }
    )
    ranks = load_cache(args.cache, args.retriever_url, args.topk)
    pending = [title for title in all_titles if title not in ranks]
    batches = [
        pending[index : index + args.batch_size]
        for index in range(0, len(pending), args.batch_size)
    ]
    print(
        f"[CONFIG] unique_titles={len(all_titles)} cached={len(all_titles) - len(pending)} "
        f"pending={len(pending)} topk={args.topk} concurrency={args.concurrency}"
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                retrieve_batch,
                args.retriever_url,
                batch,
                args.topk,
                args.timeout,
                args.retries,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            ranks.update(future.result())
            completed += len(futures[future])
            if completed % (args.batch_size * 10) == 0 or completed == len(pending):
                save_cache(args.cache, args.retriever_url, args.topk, ranks)
                print(f"[PROGRESS] checked={completed}/{len(pending)}")
    save_cache(args.cache, args.retriever_url, args.topk, ranks)

    reports = []
    for input_path, output_path, frame in zip(
        args.input, args.output, frames, strict=True
    ):
        enriched, report = enrich_frame(frame, ranks, args.topk)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(output_path, index=False)
        reports.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "rows": len(enriched),
                "by_source": report,
            }
        )
        print(f"[DONE] {output_path}")

    report_path = args.output[0].with_suffix(".retriever_gold_report.json")
    report_path.write_text(
        json.dumps(
            {
                "retriever_url": args.retriever_url,
                "topk": args.topk,
                "unique_titles": len(all_titles),
                "retrievable_unique_titles": sum(
                    rank is not None for title, rank in ranks.items() if title in all_titles
                ),
                "outputs": reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[REPORT] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
