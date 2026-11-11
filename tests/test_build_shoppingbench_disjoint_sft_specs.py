import argparse
import json

from scripts.build_shoppingbench_disjoint_sft_specs import (
    assign_shop_cardinalities,
    build,
    shop_candidate_per_mille,
    visible_item,
)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def product(product_id, shop_id):
    title_code = "".join(
        chr(ord("a") + (product_id // divisor) % 26)
        for divisor in (1, 26, 26 * 26, 26 * 26 * 26)
    )
    return {
        "product": {
            "product_id": str(900_000_000_000 + product_id),
            "shop_id": str(shop_id),
            "title": f"Catalog item {title_code}",
            "price": 100 + product_id,
            "sku_options": {"1": {"color_family": "blue"}},
            "attributes": {"material": ["steel"]},
            "service": ["COD"],
        }
    }


def test_builds_disjoint_specs_without_runtime_gold_access(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for task, product_id in zip(("product", "shop", "voucher"), (1, 2, 3)):
        reward = {"product_id": str(product_id), "title": [f"Catalog item {product_id}"]}
        if task != "product":
            reward = [reward]
        write_jsonl(
            test_dir / f"synthesize_{task}_test.jsonl",
            [{"query": f"held out query {task}", "reward": reward}],
        )

    documents = tmp_path / "documents.jsonl"
    rows = [
        product(product_id, shop_id)
        for shop_id in range(10, 210)
        for product_id in range(shop_id * 10, shop_id * 10 + 4)
    ]
    write_jsonl(documents, rows)
    output = tmp_path / "output"
    report = build(
        argparse.Namespace(
            documents=documents,
            test_dir=test_dir,
            output_dir=output,
            count_per_task=2,
            shop_group_size=2,
            val_percent=40,
            seed=20260811,
        )
    )

    assert all(report["gates"].values())
    assert report["rows"] == {"train": 6, "val": 6}
    all_rows = []
    for split in ("train", "val"):
        split_rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert {row["task"] for row in split_rows} == {"product", "shop", "voucher"}
        assert all(row["planner_input_fields"] == ["query"] for row in split_rows)
        assert all(row["gold_runtime_access"] is False for row in split_rows)
        all_rows.extend(split_rows)
    assert not {"1", "2", "3"} & {
        product_id for row in all_rows for product_id in row["source_product_ids"]
    }


def test_refuses_to_reuse_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    args = argparse.Namespace(
        documents=tmp_path / "missing.jsonl",
        test_dir=tmp_path,
        output_dir=output,
        count_per_task=1,
        shop_group_size=2,
        val_percent=10,
        seed=1,
    )
    try:
        build(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output directory was reused")


def test_visible_item_preserves_raw_attribute_key():
    item = product(99, 8)["product"]
    item["sku_options"] = {}
    _, reward = visible_item(item)
    assert reward["attributes"] == [{"material": ["steel"]}]


def test_shop_sampling_rate_scales_with_requested_count():
    assert shop_candidate_per_mille(2) == 25
    assert shop_candidate_per_mille(500) == 108
    assert shop_candidate_per_mille(10000) == 1000


def test_shop_cardinality_assignment_swaps_only_incapable_positions():
    groups = [[{}] * size for size in (3, 4, 3, 4, 4, 3)]

    assigned = assign_shop_cardinalities(groups, (2, 3, 4))

    assert sorted(assigned) == [2, 2, 3, 3, 4, 4]
    assert all(len(group) >= cardinality for group, cardinality in zip(groups, assigned))


def test_balanced_voucher_cardinality_changes_only_voucher_specs(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for task, product_id in zip(("product", "shop", "voucher"), (1, 2, 3)):
        reward = {"product_id": str(product_id), "title": [f"Held out {product_id}"]}
        write_jsonl(
            test_dir / f"synthesize_{task}_test.jsonl",
            [{"query": f"held out {task}", "reward": reward if task == "product" else [reward]}],
        )
    documents = tmp_path / "documents.jsonl"
    write_jsonl(
        documents,
        [
            product(product_id, shop_id)
            for shop_id in range(10, 2010)
            for product_id in range(shop_id * 10, shop_id * 10 + 5)
        ],
    )

    def args(output, cardinalities):
        return argparse.Namespace(
            documents=documents,
            test_dir=test_dir,
            output_dir=output,
            count_per_task=4,
            shop_group_size=2,
            voucher_cardinalities=cardinalities,
            val_percent=40,
            seed=20260811,
        )

    single_root = tmp_path / "single"
    balanced_root = tmp_path / "balanced"
    build(args(single_root, [1]))
    report = build(args(balanced_root, [1, 2, 3, 4]))
    assert report["voucher_cardinality_counts"] == {
        "train": {"1": 1, "2": 1, "3": 1, "4": 1},
        "val": {"1": 1, "2": 1, "3": 1, "4": 1},
    }
    for split in ("train", "val"):
        single = [json.loads(line) for line in (single_root / f"{split}.jsonl").read_text().splitlines()]
        balanced = [json.loads(line) for line in (balanced_root / f"{split}.jsonl").read_text().splitlines()]
        assert [row for row in single if row["task"] != "voucher"] == [
            row for row in balanced if row["task"] != "voucher"
        ]
        voucher_rows = [row for row in balanced if row["task"] == "voucher"]
        assert [len(row["source_product_ids"]) for row in voucher_rows] == [1, 2, 3, 4]
        assert voucher_rows[0]["query"].startswith("Find this product:")
        assert all(row["query"].startswith("Find these products:") for row in voucher_rows[1:])


def test_balanced_shop_cardinality_changes_only_shop_specs(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for task, product_id in zip(("product", "shop", "voucher"), (1, 2, 3)):
        reward = {"product_id": str(product_id), "title": [f"Held out {product_id}"]}
        write_jsonl(
            test_dir / f"synthesize_{task}_test.jsonl",
            [{"query": f"held out {task}", "reward": reward if task == "product" else [reward]}],
        )
    documents = tmp_path / "documents.jsonl"
    write_jsonl(
        documents,
        [
            product(product_id, shop_id)
            for shop_id in range(10, 3010)
            for product_id in range(shop_id * 10, shop_id * 10 + 5)
        ],
    )

    def args(output, cardinalities):
        return argparse.Namespace(
            documents=documents,
            test_dir=test_dir,
            output_dir=output,
            count_per_task=6,
            shop_group_size=3,
            shop_cardinalities=cardinalities,
            voucher_cardinalities=[1, 2, 3, 4],
            val_percent=40,
            seed=20260811,
        )

    fixed_root = tmp_path / "fixed"
    balanced_root = tmp_path / "balanced"
    build(args(fixed_root, [3]))
    report = build(args(balanced_root, [2, 3, 4]))
    assert report["shop_cardinality_counts"] == {
        "train": {"2": 2, "3": 2, "4": 2},
        "val": {"2": 2, "3": 2, "4": 2},
    }
    for split in ("train", "val"):
        fixed = [json.loads(line) for line in (fixed_root / f"{split}.jsonl").read_text().splitlines()]
        balanced = [json.loads(line) for line in (balanced_root / f"{split}.jsonl").read_text().splitlines()]
        assert [row for row in fixed if row["task"] != "shop"] == [
            row for row in balanced if row["task"] != "shop"
        ]
        shop_rows = [row for row in balanced if row["task"] == "shop"]
        assert [len(row["source_product_ids"]) for row in shop_rows] == [2, 3, 4, 2, 3, 4]
