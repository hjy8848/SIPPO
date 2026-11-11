"""Run ShoppingBench examples against ROSA and score exact product IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from qa_manager.ROSA.shoppingbench_protocol import (
    extract_product_ids,
    normalize_final_answer,
    validate_final_answer,
)


@dataclass(frozen=True)
class ShoppingBenchItem:
    sample_id: str
    position: int
    query: str
    reward: list[dict[str, Any]]
    voucher: dict[str, Any] | None

    @property
    def expected_product_ids(self) -> list[str]:
        return [str(item["product_id"]) for item in self.reward]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_items(path: Path, task: str, offset: int, limit: int) -> list[ShoppingBenchItem]:
    rows: list[ShoppingBenchItem] = []
    with path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            query = str(payload.get("query") or "").strip()
            reward = payload.get("reward")
            if task == "product" and isinstance(reward, dict):
                reward = [reward]
            if not query or not isinstance(reward, list) or not reward:
                raise ValueError(f"Invalid ShoppingBench row {position + 1}")
            if any(not isinstance(item, dict) or "product_id" not in item for item in reward):
                raise ValueError(f"Invalid reward in ShoppingBench row {position + 1}")
            rows.append(
                ShoppingBenchItem(
                    sample_id=f"{task}-{position:04d}",
                    position=position,
                    query=query,
                    reward=reward,
                    voucher=payload.get("voucher"),
                )
            )
    selected = rows[offset : offset + limit if limit > 0 else None]
    if not selected:
        raise ValueError("Selected ShoppingBench slice is empty")
    return selected


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
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_journal(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if sample_id not in expected_ids or sample_id in rows:
                raise ValueError(f"Invalid sample_id in {path}:{line_number}: {sample_id}")
            rows[sample_id] = row
    return rows


def score_ids(predicted: list[str], expected: list[str]) -> dict[str, Any]:
    matched = sum(
        index < len(predicted) and predicted[index] == product_id
        for index, product_id in enumerate(expected)
    )
    return {
        "matched_product_ids": matched,
        "expected_product_count": len(expected),
        "exact_id_fraction": matched / len(expected),
        "first_id_exact": bool(predicted) and predicted[0] == expected[0],
        "exact_sequence_success": predicted == expected,
    }


def retrieval_calls(trace: dict[str, Any]) -> int:
    total = 0
    for node in trace.get("nodes_execution") or []:
        if node.get("type") != "search":
            continue
        attempts = node.get("attempted_queries")
        total += len(attempts) if isinstance(attempts, list) and attempts else 1
    return total


def contract_flag(arm: str) -> bool | None:
    if arm == "original":
        return False
    if arm == "contract":
        return True
    if arm == "default":
        return None
    raise ValueError(f"Unknown evaluation arm: {arm}")


def load_plan_overrides(
    path: Path | None,
    items: list[ShoppingBenchItem],
    *,
    preserve_text: bool = False,
) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Plan override manifest must be a JSON object")
    expected = {item.sample_id for item in items}
    overrides = {
        str(sample_id): str(plan) if preserve_text else str(plan).strip()
        for sample_id, plan in payload.items()
        if str(sample_id) in expected
    }
    missing = sorted(expected - set(overrides))
    if missing:
        raise ValueError(f"Plan override manifest is missing: {missing}")
    invalid = [sample_id for sample_id, plan in overrides.items() if "<plan" not in plan]
    if invalid:
        raise ValueError(f"Invalid plan override(s): {invalid}")
    return overrides


def request_prediction(
    item: ShoppingBenchItem,
    *,
    api_url: str,
    timeout: float,
    retries: int,
    arm: str,
    plan_override: str | None = None,
    planner_output_override: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    request_error = None
    trace: dict[str, Any] = {}
    expected_contract = contract_flag(arm)
    request_payload: dict[str, Any] = {"query": item.query}
    if expected_contract is not None:
        request_payload["contract_preserving_dag_v1"] = expected_contract
    if plan_override and planner_output_override:
        raise ValueError(
            "plan_override and planner_output_override are mutually exclusive"
        )
    if plan_override:
        request_payload["plan_override"] = plan_override
    if planner_output_override:
        request_payload["planner_output_override"] = planner_output_override
    for attempt in range(retries + 1):
        try:
            response = requests.post(api_url, json=request_payload, timeout=timeout)
            response.raise_for_status()
            trace = response.json()
            if not isinstance(trace, dict):
                raise ValueError("ROSA response is not a JSON object")
            request_error = None
            break
        except Exception as exc:
            request_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 8))

    contract_errors = []
    if trace and trace.get("protocol_profile") != "shoppingbench":
        contract_errors.append(
            f"Expected shoppingbench profile, got {trace.get('protocol_profile')!r}"
        )
    if trace and expected_contract is not None:
        observed_contract = (trace.get("ablation_config") or {}).get(
            "contract_preserving_dag_v1"
        )
        if observed_contract is not expected_contract:
            contract_errors.append(
                "Expected contract_preserving_dag_v1="
                f"{expected_contract}, got {observed_contract!r}"
            )
    if trace and plan_override:
        if trace.get("plan_override_used") is not True:
            contract_errors.append("ROSA did not acknowledge plan_override")
        if trace.get("planner_output_override_used") is True:
            contract_errors.append("ROSA reported both override modes")
    if trace and planner_output_override:
        if trace.get("planner_output_override_used") is not True:
            contract_errors.append("ROSA did not acknowledge planner_output_override")
        if trace.get("plan_override_used") is True:
            contract_errors.append("ROSA reported both override modes")
        if trace.get("raw_plan_output") != planner_output_override:
            contract_errors.append("ROSA changed the frozen raw Planner output")
        compiler = (trace.get("contract_protocol") or {}).get("plan_compiler") or {}
        if compiler.get("enabled") is not True or compiler.get("applied") is not True:
            contract_errors.append("ROSA did not apply the system plan compiler")
        if (trace.get("contract_protocol") or {}).get("replan_attempted") is True:
            contract_errors.append("ROSA replanned a frozen Planner output")
    contract_error = "; ".join(contract_errors) or None
    pipeline_error = trace.get("error") or trace.get("execution_error") if trace else None
    final_answer = normalize_final_answer(str(trace.get("final_answer") or "")) if trace else ""
    format_error = validate_final_answer(final_answer) if trace else "missing ROSA trace"
    predicted_ids = extract_product_ids(final_answer) if not format_error else []
    score = score_ids(predicted_ids, item.expected_product_ids)
    return {
        "sample_id": item.sample_id,
        "arm": arm,
        "position": item.position,
        "query": item.query,
        "expected_product_ids": item.expected_product_ids,
        "predicted_product_ids": predicted_ids,
        "score": score,
        "format_error": format_error,
        "contract_error": contract_error,
        "pipeline_error": pipeline_error,
        "request_error": request_error,
        "search_nodes": sum(
            node.get("type") == "search" for node in (trace.get("nodes_execution") or [])
        ),
        "retrieval_calls": retrieval_calls(trace),
        "elapsed_seconds": time.time() - started,
        "raw_trace": trace,
    }


def materialize(
    items: list[ShoppingBenchItem], rows: dict[str, dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    ordered = [rows[item.sample_id] for item in items if item.sample_id in rows]
    for row in ordered:
        row["score"] = score_ids(
            row.get("predicted_product_ids") or [],
            row.get("expected_product_ids") or [],
        )
    atomic_jsonl_dump(ordered, output_dir / "results.jsonl")
    failures = [
        row
        for row in ordered
        if any(
            row.get(key)
            for key in ("format_error", "contract_error", "pipeline_error", "request_error")
        )
    ]
    summary = {
        "expected": len(items),
        "completed": len(ordered),
        "pending": len(items) - len(ordered),
        "valid_trajectories": len(ordered) - len(failures),
        "failures": len(failures),
        "exact_sequence_successes": sum(
            bool(row["score"]["exact_sequence_success"]) for row in ordered
        ),
        "exact_sequence_success_rate": (
            sum(bool(row["score"]["exact_sequence_success"]) for row in ordered) / len(ordered)
            if ordered
            else 0.0
        ),
        "first_id_exact_successes": sum(
            bool(row["score"].get("first_id_exact")) for row in ordered
        ),
        "first_id_exact_rate": (
            sum(bool(row["score"].get("first_id_exact")) for row in ordered) / len(ordered)
            if ordered
            else 0.0
        ),
        "mean_exact_id_fraction": (
            sum(row["score"]["exact_id_fraction"] for row in ordered) / len(ordered)
            if ordered
            else 0.0
        ),
        "search_nodes": sum(row["search_nodes"] for row in ordered),
        "retrieval_calls": sum(row["retrieval_calls"] for row in ordered),
        "failure_sample_ids": [row["sample_id"] for row in failures],
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", choices=("product", "shop", "voucher"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8178/rosa/chat")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--arm", choices=("original", "contract", "default"), default="default"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--plan-override-manifest", type=Path)
    parser.add_argument("--planner-output-override-manifest", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = args.dataset.expanduser().resolve()
    items = load_items(dataset, args.task, args.offset, args.limit)
    plan_override_path = (
        args.plan_override_manifest.expanduser().resolve()
        if args.plan_override_manifest is not None
        else None
    )
    planner_output_override_path = (
        args.planner_output_override_manifest.expanduser().resolve()
        if args.planner_output_override_manifest is not None
        else None
    )
    if plan_override_path and planner_output_override_path:
        raise SystemExit(
            "--plan-override-manifest and --planner-output-override-manifest "
            "are mutually exclusive"
        )
    plan_overrides = load_plan_overrides(plan_override_path, items)
    planner_output_overrides = load_plan_overrides(
        planner_output_override_path, items, preserve_text=True
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.resume or args.audit_only):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol": "ShoppingBench exact product-ID evaluation without judge",
        "task": args.task,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "offset": args.offset,
        "limit": args.limit,
        "selected_count": len(items),
        "api_url": args.api_url,
        "arm": args.arm,
        "contract_preserving_dag_v1": contract_flag(args.arm),
        "concurrency": args.concurrency,
        "metric": "position-aligned exact product ID",
        "judge_run": False,
        "plan_override_manifest": str(plan_override_path) if plan_override_path else None,
        "plan_override_manifest_sha256": (
            sha256_file(plan_override_path) if plan_override_path else None
        ),
        "planner_output_override_manifest": (
            str(planner_output_override_path) if planner_output_override_path else None
        ),
        "planner_output_override_manifest_sha256": (
            sha256_file(planner_output_override_path)
            if planner_output_override_path
            else None
        ),
    }
    protocol_path = output_dir / "evaluation_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise SystemExit("Existing evaluation protocol does not match this run")
    else:
        atomic_json_dump(protocol, protocol_path)

    journal_path = output_dir / "journal.jsonl"
    rows = read_journal(journal_path, {item.sample_id for item in items})
    if args.audit_only:
        print(json.dumps(materialize(items, rows, output_dir), indent=2))
        return 0

    pending = [item for item in items if item.sample_id not in rows]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                request_prediction,
                item,
                api_url=args.api_url,
                timeout=args.timeout,
                retries=max(0, args.retries),
                arm=args.arm,
                plan_override=plan_overrides.get(item.sample_id),
                planner_output_override=planner_output_overrides.get(item.sample_id),
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            row = future.result()
            append_journal(journal_path, row)
            rows[row["sample_id"]] = row
            summary = materialize(items, rows, output_dir)
            print(
                f"[PROGRESS] {summary['completed']}/{summary['expected']} "
                f"exact={summary['exact_sequence_successes']} failures={summary['failures']}"
            )

    summary = materialize(items, rows, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
