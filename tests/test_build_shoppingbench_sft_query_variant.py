import json

from scripts.audit_shoppingbench_sft_spec_retrieval import atomic_queries_from_visible_query
from scripts.build_shoppingbench_sft_query_variant import (
    build,
    document_frequency,
    variant_query,
    without_query,
)


def source_row(split, task, index, count=None):
    count = count if count is not None else (3 if task == "shop" else 1)
    products = [
        {
            "product_id": str(900000 + index * 10 + item),
            "title": [f"RareBrand {task} useful catalog item {index} model {item}"],
        }
        for item in range(count)
    ]
    atoms = [
        f"{product['title'][0]}; it should have color shade{item}"
        for item, product in enumerate(products)
    ]
    if task == "product":
        query = f"Find this product in the catalog: {atoms[0]}."
    elif task == "shop":
        query = "Find one shop that sells all of these items: " + "; ".join(
            f"({item}) {atom}" for item, atom in enumerate(atoms, 1)
        ) + "."
    else:
        details = (
            atoms[0]
            if len(atoms) == 1
            else "; ".join(f"({item}) {atom}" for item, atom in enumerate(atoms, 1))
        )
        query = (
            f"{'Find this product' if len(atoms) == 1 else 'Find these products'}: "
            f"{details}. My budget is 50 Philippine pesos. "
            "I have a platform voucher valid above 20 pesos with a fixed discount of 5 pesos."
        )
    row = {
        "spec_id": f"{split}-{task}-{index:05d}",
        "split": split,
        "task": task,
        "query": query,
        "reward": products[0] if len(products) == 1 else products,
        "source_product_ids": [product["product_id"] for product in products],
        "source_shop_ids": [str(7000 + index)],
        "planner_input_fields": ["query"],
        "gold_runtime_access": False,
    }
    if task == "voucher":
        row["voucher"] = {
            "budget": 50,
            "threshold": 20,
            "face_value": 5,
        }
    return row


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_changes_only_query_and_passes_query_style_gates(tmp_path):
    source = tmp_path / "source"
    formal = tmp_path / "formal"
    source.mkdir()
    formal.mkdir()
    originals = {}
    for split in ("train", "val"):
        rows = [source_row(split, task, index) for task in ("product", "shop", "voucher") for index in range(8)]
        originals[split] = rows
        write_jsonl(source / f"{split}.jsonl", rows)
    for task in ("product", "shop", "voucher"):
        write_jsonl(
            formal / f"synthesize_{task}_test.jsonl",
            [{"query": f"unrelated natural request for {task}", "reward": {"product_id": "1", "title": ["hidden"]}}],
        )

    output = tmp_path / "variant"
    report = build(source, formal, output)

    assert all(report["gates"].values())
    assert report["status"] == "query_style_ablation_only_not_training_data"
    for split in ("train", "val"):
        updated = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert len(updated) == len(originals[split])
        assert all(without_query(before) == without_query(after) for before, after in zip(originals[split], updated))
        assert all(after["query"] != before["query"] for before, after in zip(originals[split], updated))


def test_query_variant_preserves_multiple_voucher_atoms():
    row = source_row("train", "voucher", 3, count=3)
    frequencies, total = document_frequency([row])

    query, atoms = variant_query(row, frequencies, total)
    updated = {**row, "query": query}

    assert len(atoms) == 3
    assert query.count("(1)") == 1
    assert query.count("(2)") == 1
    assert query.count("(3)") == 1
    assert atomic_queries_from_visible_query(updated) == atoms
