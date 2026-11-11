"""Generate ShoppingComp predictions from a running ROSA shopping endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from qa_manager.ROSA.shoppingcomp_protocol import normalize_final_answer, validate_final_answer


OFFICIAL_DATASET_REPO = "huaixiao/ShoppingComp"
OFFICIAL_FILES = {
    "ShoppingComp_97_20260127.en.jsonl",
    "ShoppingComp_97_20260127.zh.jsonl",
    "ShoppingComp_traps_48_20260127.en.jsonl",
    "ShoppingComp_traps_48_20260127.zh.jsonl",
}
PRODUCT_RE = re.compile(r"<product_name>(.*?)</product_name>", re.DOTALL | re.IGNORECASE)
CANDIDATE_RE = re.compile(
    r"<candidate_product_list>.*?</candidate_product_list>", re.DOTALL | re.IGNORECASE
)
SAFETY_RE = re.compile(r"<safety_assessment>.*?</safety_assessment>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ShoppingItem:
    uuid: str
    question: str
    position: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    filename = path.name
    if filename not in OFFICIAL_FILES:
        raise FileNotFoundError(
            f"Dataset not found: {value}. Expected a local file or one of {sorted(OFFICIAL_FILES)}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub or pass a downloaded ShoppingComp JSONL file."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=OFFICIAL_DATASET_REPO,
            repo_type="dataset",
            filename=filename,
        )
    ).resolve()


def load_items(path: Path, offset: int = 0, limit: int = 0) -> list[ShoppingItem]:
    items: list[ShoppingItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            uuid = str(row.get("uuid") or "").strip()
            question = str(row.get("question") or "").strip()
            if not uuid or not question:
                raise ValueError(f"Dataset row {position + 1} lacks uuid or question")
            if uuid in seen:
                raise ValueError(f"Duplicate dataset uuid: {uuid}")
            seen.add(uuid)
            items.append(ShoppingItem(uuid=uuid, question=question, position=len(items)))
    selected = items[offset : offset + limit if limit > 0 else None]
    if not selected:
        raise ValueError("The selected ShoppingComp slice is empty")
    return selected


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            yield row


def load_journal(path: Path, expected_uuids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path) or []:
        uuid = str(row.get("uuid") or "")
        if uuid not in expected_uuids:
            raise ValueError(f"Journal contains an unexpected uuid: {uuid}")
        if uuid in rows:
            raise ValueError(f"Journal contains a duplicate uuid: {uuid}")
        rows[uuid] = row
    return rows


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_jsonl_dump(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_response(final_answer: str) -> tuple[str, list[str], str | None]:
    candidate = CANDIDATE_RE.search(final_answer)
    safety = SAFETY_RE.search(final_answer)
    if candidate:
        response = normalize_final_answer(candidate.group(0).strip())
        format_issue = validate_final_answer(response)
        if format_issue:
            return response, [], format_issue
        products = []
        seen = set()
        for raw_name in PRODUCT_RE.findall(response):
            name = " ".join(re.sub(r"<[^>]+>", " ", raw_name).split()).strip()
            if name and name.casefold() not in seen:
                seen.add(name.casefold())
                products.append(name)
        if not products:
            return response, [], "candidate_product_list has no product_name"
        return response, products, None
    if safety:
        response = normalize_final_answer(safety.group(0).strip())
        return response, [], validate_final_answer(response)
    return final_answer.strip(), [], "missing candidate_product_list or safety_assessment"


def count_retrieval_calls(trace: dict[str, Any]) -> int:
    calls = 0
    for node in trace.get("nodes_execution") or []:
        if node.get("type") != "search":
            continue
        attempted = node.get("attempted_queries")
        calls += len(attempted) if isinstance(attempted, list) and attempted else 1
    return calls


def request_prediction(
    item: ShoppingItem,
    *,
    api_url: str,
    model_name: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    started = time.time()
    error = None
    trace: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            response = requests.post(api_url, json={"query": item.question}, timeout=timeout)
            response.raise_for_status()
            trace = response.json()
            if not isinstance(trace, dict):
                raise ValueError("ROSA response is not a JSON object")
            error = None
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 8))

    contract_error = None
    if trace and trace.get("protocol_profile") != "shoppingcomp":
        contract_error = (
            "ROSA endpoint did not confirm protocol_profile=shoppingcomp; "
            f"got {trace.get('protocol_profile')!r}"
        )
    pipeline_error = trace.get("error") or trace.get("execution_error") if trace else None
    final_answer = str(trace.get("final_answer") or "") if trace else ""
    response_text, products, format_error = extract_response(final_answer)
    errors = [value for value in (error, contract_error, pipeline_error, format_error) if value]
    prediction = {
        "uuid": item.uuid,
        "question": item.question,
        "model_results": [
            {
                "model": model_name,
                "products": products,
                "response": response_text,
                "success": not errors,
            }
        ],
    }
    return {
        "uuid": item.uuid,
        "position": item.position,
        "prediction": prediction,
        "trajectory": {
            "uuid": item.uuid,
            "question": item.question,
            "model": model_name,
            "products": products,
            "search_nodes": sum(
                1 for node in (trace.get("nodes_execution") or []) if node.get("type") == "search"
            ),
            "retrieval_calls": count_retrieval_calls(trace),
            "format_error": format_error,
            "contract_error": contract_error,
            "pipeline_error": pipeline_error,
            "request_error": error,
            "elapsed_seconds": time.time() - started,
            "raw_trace": trace,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def materialize(
    items: list[ShoppingItem], journal_rows: dict[str, dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    ordered = [journal_rows[item.uuid] for item in items if item.uuid in journal_rows]
    predictions = [row["prediction"] for row in ordered]
    trajectories = [row["trajectory"] for row in ordered]
    atomic_jsonl_dump(predictions, output_dir / "predictions.jsonl")
    atomic_jsonl_dump(trajectories, output_dir / "trajectories.jsonl")

    failures = [row for row in trajectories if not row.get("raw_trace") or any(
        row.get(key)
        for key in ("format_error", "contract_error", "pipeline_error", "request_error")
    )]
    summary = {
        "expected": len(items),
        "completed": len(ordered),
        "pending": len(items) - len(ordered),
        "successful": len(ordered) - len(failures),
        "failures": len(failures),
        "empty_product_lists": sum(
            not bool(row["prediction"]["model_results"][0].get("products")) for row in ordered
        ),
        "search_nodes": sum(row.get("search_nodes", 0) for row in trajectories),
        "retrieval_calls": sum(row.get("retrieval_calls", 0) for row in trajectories),
        "failure_uuids": [row["uuid"] for row in failures],
    }
    atomic_json_dump(summary, output_dir / "generation_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Local JSONL or official HF filename")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8178/rosa/chat")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8078/retrieve")
    parser.add_argument("--search-backend", default="ddgs")
    parser.add_argument("--search-region", default="us-en")
    parser.add_argument("--search-cache", default="")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = resolve_dataset(args.dataset)
    items = load_items(dataset_path, offset=args.offset, limit=args.limit)
    output_dir = args.output_dir.resolve()
    journal_path = output_dir / "generation_journal.jsonl"
    protocol_path = output_dir / "generation_protocol.json"

    if output_dir.exists() and any(output_dir.iterdir()) and not (args.resume or args.audit_only):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_uuids = {item.uuid for item in items}
    journal_rows = load_journal(journal_path, expected_uuids)

    protocol = {
        "protocol": "ShoppingComp prediction generation without judge",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "selected_offset": args.offset,
        "selected_limit": args.limit,
        "selected_count": len(items),
        "model_name": args.model_name,
        "api_url": args.api_url,
        "retriever_url": args.retriever_url,
        "search_backend": args.search_backend,
        "search_region": args.search_region,
        "search_cache": args.search_cache,
        "concurrency": args.concurrency,
        "judge_run": False,
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise SystemExit("Resume protocol differs from the existing generation_protocol.json")
    else:
        atomic_json_dump(protocol, protocol_path)

    if args.audit_only:
        print(json.dumps(materialize(items, journal_rows, output_dir), indent=2))
        return 0

    pending = [item for item in items if item.uuid not in journal_rows]
    print(
        f"[INFO] ShoppingComp generation expected={len(items)} "
        f"completed={len(journal_rows)} pending={len(pending)}"
    )
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                request_prediction,
                item,
                api_url=args.api_url,
                model_name=args.model_name,
                timeout=args.timeout,
                retries=max(0, args.retries),
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            row = future.result()
            append_journal(journal_path, row)
            journal_rows[row["uuid"]] = row
            summary = materialize(items, journal_rows, output_dir)
            print(
                f"[PROGRESS] {summary['completed']}/{summary['expected']} "
                f"success={summary['successful']} failures={summary['failures']}"
            )

    summary = materialize(items, journal_rows, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
