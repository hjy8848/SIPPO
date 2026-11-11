import json
from pathlib import Path

from scripts.build_shoppingbench_disjoint_sft_specs import make_spec
from scripts.expand_shoppingbench_frozen_shop_cardinality import build


def product(product_id: int, shop_id: int) -> dict:
    return {
        "product_id": str(product_id),
        "shop_id": str(shop_id),
        "title": f"Product {product_id}",
        "price": 100.0 + product_id,
        "sku_options": {"default": {"color": "red"}},
        "attributes": {},
        "service": [],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_expands_only_frozen_shop_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    documents = []
    frozen: dict[str, list[dict]] = {}
    for split_index, split in enumerate(("train", "val")):
        rows = []
        base = 1000 + split_index * 100
        for index in range(6):
            shop_id = base + index
            products = [product(shop_id * 10 + offset, shop_id) for offset in range(4)]
            documents.extend(products)
            rows.append(make_spec("shop", products[:3], split, index))
        single = product(base * 10 + 90, base + 90)
        voucher_products = [product(base * 10 + 91 + offset, base + 91 + offset) for offset in range(2)]
        rows.append(make_spec("product", [single], split, 0))
        rows.append(make_spec("voucher", voucher_products, split, 0))
        documents.extend([single, *voucher_products])
        frozen[split] = rows
        write_jsonl(source / f"{split}.jsonl", rows)
    documents_path = tmp_path / "documents.jsonl"
    write_jsonl(documents_path, documents)

    output = tmp_path / "output"
    report = build(source, documents_path, output, seed=7)
    assert all(report["gates"].values())
    assert report["external_api_calls"] == 0
    for split in ("train", "val"):
        result = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert [len(row["source_product_ids"]) for row in result[:6]] == [2, 3, 4, 2, 3, 4]
        assert result[6:] == frozen[split][6:]
        for before, after in zip(frozen[split][:6], result[:6], strict=True):
            assert before["source_shop_ids"] == after["source_shop_ids"]
            assert after["source_product_ids"][: min(3, len(after["source_product_ids"]))] == before[
                "source_product_ids"
            ][: min(3, len(after["source_product_ids"]))]
            if len(after["source_product_ids"]) == 3:
                assert before == after


def test_refuses_to_reuse_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    try:
        build(tmp_path, tmp_path / "documents.jsonl", output, seed=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected overwrite guard")
