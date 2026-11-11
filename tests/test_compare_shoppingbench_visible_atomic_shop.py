import json
from pathlib import Path

from scripts.compare_shoppingbench_visible_atomic_shop import compare


def row(index: int, *, fallback: bool, failure: bool = False) -> dict:
    question = (
        "Find one shop for all these items. Product details: "
        f"(1) item {index} a; (2) item {index} b."
    )
    raw_plan = (
        f"<plan><search>combined item {index}</search></plan>"
        if fallback
        else f"<plan><search>q=item {index} a</search><search>q=item {index} b</search></plan>"
    )
    queries = (
        [f"item {index} a", f"item {index} b"]
        if fallback
        else [f"item {index} a", f"item {index} b"]
    )
    return {
        "sample_id": f"shop-{index:04d}",
        "query": question,
        "score": {"exact_id_fraction": 1.0, "exact_sequence_success": True, "first_id_exact": True},
        "format_error": "failed" if failure else None,
        "contract_error": None,
        "pipeline_error": None,
        "request_error": None,
        "retrieval_calls": 2,
        "raw_trace": {
            "raw_plan_output": raw_plan,
            "planner_output_override_used": True,
            "plan_override_used": False,
            "sft_events": [],
            "contract_protocol": {
                "plan_compiler": {
                    "applied": True,
                    "request_kind": "shop",
                    "queries": queries,
                }
            },
        },
    }


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")


def test_shop_gate_requires_exactly_seven_fallbacks_and_nonregression(tmp_path: Path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    rows = [row(index, fallback=index < 7) for index in range(16)]
    write(baseline, rows)
    write(candidate, rows)

    report = compare(baseline, candidate, tmp_path / "report")

    assert report["passed"] is True
    assert sum(value["fallback_observed"] for value in report["paired"]) == 7
    assert report["training_authorized"] is False
