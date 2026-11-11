import json
from pathlib import Path

from scripts.analyze_shoppingbench_official_vs_asc import (
    analyze,
    raw_tool_calls,
    read_stable_jsonl,
)


def test_raw_tool_calls_recovers_unterminated_closing_tag() -> None:
    content = (
        '<tool_call>[{"name":"recommend_product","parameters":'
        '{"product_ids":"1,2"}}]'
    )
    assert raw_tool_calls(content)[0]["parameters"]["product_ids"] == "1,2"


def test_stable_jsonl_requires_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": 1}) + "\n", encoding="utf-8")
    rows = read_stable_jsonl(
        path, expected_rows=1, settle_seconds=0.001, timeout_seconds=1
    )
    assert rows == [{"id": 1}]


def test_stable_jsonl_does_not_split_unicode_line_separator(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    expected = {"description": "left\u2028right"}
    path.write_text(
        json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = read_stable_jsonl(
        path, expected_rows=1, settle_seconds=0.001, timeout_seconds=1
    )
    assert rows == [expected]


def test_analysis_keeps_strict_and_recovered_results_separate() -> None:
    query = "one shop"
    trajectory = [
        {
            "prompt": [],
            "completion": {
                "content": (
                    '<tool_call>[{"name":"find_product","parameters":{"q":"x"}}]'
                ),
                "message": {},
            },
            "extra_info": {"query": query, "step": 1},
        },
        {
            "prompt": [],
            "completion": {
                "content": (
                    '<tool_call>[{"name":"recommend_product","parameters":'
                    '{"product_ids":"1,2"}}]'
                ),
                "message": {},
            },
            "extra_info": {"query": query, "step": 2},
        },
    ]
    reference = {
        "sample_id": "shop-0000",
        "question": query,
        "expected_product_ids": ["1", "2"],
        "baseline_product_ids": [],
        "baseline_score": {
            "exact_sequence_success": False,
            "exact_id_fraction": 0.0,
        },
        "probe": {"predicted_product_ids": []},
        "probe_score": {
            "exact_sequence_success": False,
            "exact_id_fraction": 0.0,
        },
    }
    report = analyze([trajectory], {query: reference}, retry_count=10)
    row = report["rows"][0]
    assert row["official_strict_product_ids"] == []
    assert row["official_diagnostic_recovered_product_ids"] == ["1", "2"]
    assert row["official_format_recovery_needed"] is True
    assert report["summary"]["official_missing_page_calls"] == 1
    assert report["summary"]["official_tool_retry_log_count"] == 10
