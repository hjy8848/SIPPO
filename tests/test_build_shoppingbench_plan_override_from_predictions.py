import json

import pytest

from scripts.build_shoppingbench_plan_override_from_predictions import build, build_many


def test_builds_byte_preserving_no_gold_override_manifest(tmp_path):
    predictions = tmp_path / "results.jsonl"
    rows = [
        {
            "sample_id": f"voucher-{index:04d}",
            "expected_product_ids": ["gold-must-not-be-read"],
            "raw_trace": {
                "raw_plan_output": f"  <plan><search>q=item {index}</search></plan>\n"
            },
        }
        for index in range(2)
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    output = tmp_path / "override"
    manifest = build(predictions, output, expected_rows=2)
    overrides = json.loads((output / "planner_output_overrides.json").read_text())

    assert manifest["gold_fields_read"] is False
    assert manifest["fields_read"] == ["sample_id", "raw_trace.raw_plan_output"]
    assert manifest["plans_modified"] is False
    assert manifest["runtime_request_field"] == "planner_output_override"
    assert overrides == {
        "voucher-0000": "  <plan><search>q=item 0</search></plan>\n",
        "voucher-0001": "  <plan><search>q=item 1</search></plan>\n",
    }
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        build(predictions, output, expected_rows=2)


def test_builds_one_byte_preserving_manifest_from_multiple_tasks(tmp_path):
    paths = []
    for task in ("product", "shop", "voucher"):
        path = tmp_path / f"{task}.jsonl"
        path.write_text(json.dumps({
            "sample_id": f"{task}-0000",
            "raw_trace": {"raw_plan_output": f"<plan><search>{task}</search></plan>"},
        }) + "\n", encoding="utf-8")
        paths.append(path)

    output = tmp_path / "combined"
    manifest = build_many(paths, output, expected_rows=3)
    overrides = json.loads((output / "planner_output_overrides.json").read_text())
    assert sorted(overrides) == ["product-0000", "shop-0000", "voucher-0000"]
    assert manifest["rows"] == 3
    assert len(manifest["source_predictions_sha256"]) == 3
