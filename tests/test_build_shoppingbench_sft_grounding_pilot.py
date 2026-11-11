import json

import pytest

from scripts.build_shoppingbench_sft_grounding_pilot import build


def make_row(task, index):
    product_ids = [str(index + 100)] if task != "shop" else [str(index + 100), str(index + 200)]
    rewards = [{"product_id": value, "title": [f"Product {value}"]} for value in product_ids]
    return {
        "spec_id": f"val-{task}-{index:05d}",
        "split": "val",
        "task": task,
        "query": f"Visible query {task} {index}",
        "reward": rewards[0] if len(rewards) == 1 else rewards,
        "source_product_ids": product_ids,
        "planner_input_fields": ["query"],
        "gold_runtime_access": False,
    }


def test_builds_fixed_unfiltered_validation_pilot(tmp_path):
    source = tmp_path / "val.jsonl"
    rows = [make_row(task, index) for task in ("voucher", "product", "shop") for index in range(3, -1, -1)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "pilot"

    manifest = build(source, output, per_task=2)

    assert manifest["total_rows"] == 6
    assert manifest["selection"].startswith("first rows after lexical spec_id sort")
    assert manifest["gold_runtime_access"] is False
    for task in ("product", "shop", "voucher"):
        selected = [json.loads(line) for line in (output / f"{task}.jsonl").read_text().splitlines()]
        assert [row["spec_id"] for row in selected] == [
            f"val-{task}-00000",
            f"val-{task}-00001",
        ]


def test_rejects_reuse_and_runtime_gold(tmp_path):
    source = tmp_path / "val.jsonl"
    rows = [make_row(task, 0) for task in ("product", "shop", "voucher")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "pilot"
    build(source, output, per_task=1)
    with pytest.raises(FileExistsError):
        build(source, output, per_task=1)

    rows[0]["gold_runtime_access"] = True
    leaked = tmp_path / "leaked.jsonl"
    leaked.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime leakage"):
        build(leaked, tmp_path / "leaked-output", per_task=1)
