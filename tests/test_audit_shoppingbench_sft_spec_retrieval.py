import pytest

from scripts.audit_shoppingbench_sft_spec_retrieval import (
    atomic_queries_from_visible_query,
    audit_rows,
    write_report,
)


class Hit:
    def __init__(self, docid):
        self.docid = docid


class FakeSearcher:
    results = {
        "Alpha Product; it should have color red": ["100", "noise"],
        "Beta Product; it should offer COD": ["noise", "200"],
        "Gamma Product; it should have size large": ["300"],
    }

    def search(self, query, topk):
        return [Hit(value) for value in self.results.get(query, [])[:topk]]

    def doc(self, docid):
        return object() if docid in {"100", "200", "300"} else None


def row(task, split, index, query, ids):
    rewards = [{"product_id": value, "title": [value]} for value in ids]
    return {
        "spec_id": f"{split}-{task}-{index:05d}",
        "split": split,
        "task": task,
        "query": query,
        "reward": rewards[0] if len(rewards) == 1 else rewards,
        "source_product_ids": ids,
        "planner_input_fields": ["query"],
        "gold_runtime_access": False,
    }


def test_atomic_queries_use_only_visible_template_text():
    product = row(
        "product",
        "train",
        0,
        "Find this product in the catalog: Alpha Product; it should have color red.",
        ["100"],
    )
    shop = row(
        "shop",
        "train",
        1,
        "Find one shop that sells all of these items: (1) Alpha Product; it should have color red; "
        "(2) Beta Product; it should offer COD.",
        ["100", "200"],
    )
    voucher = row(
        "voucher",
        "val",
        0,
        "Find this product: Gamma Product; it should have size large. My budget is 50 Philippine "
        "pesos. I have a platform voucher valid above 20 pesos with a fixed discount of 5 pesos.",
        ["300"],
    )

    assert atomic_queries_from_visible_query(product) == [
        "Alpha Product; it should have color red"
    ]
    assert atomic_queries_from_visible_query(shop) == [
        "Alpha Product; it should have color red",
        "Beta Product; it should offer COD",
    ]
    assert atomic_queries_from_visible_query(voucher) == [
        "Gamma Product; it should have size large"
    ]
    multi_voucher = row(
        "voucher",
        "val",
        1,
        "Find these products: (1) Alpha Product; it should have color red; "
        "(2) Beta Product; it should offer COD. My budget is 100 Philippine pesos. "
        "I have a platform voucher valid above 50 pesos with a fixed discount of 10 pesos.",
        ["100", "200"],
    )
    assert atomic_queries_from_visible_query(multi_voucher) == [
        "Alpha Product; it should have color red",
        "Beta Product; it should offer COD",
    ]


def test_atomic_queries_support_partial_title_query_variant():
    product = row(
        "product",
        "train",
        0,
        "Could you help me locate an item? Product details: Alpha red; it should have size large.",
        ["100"],
    )
    shop = row(
        "shop",
        "train",
        1,
        "I need one seller for several items. Product details: (1) Alpha red; it should have size large; "
        "(2) Beta blue; it should offer COD.",
        ["100", "200"],
    )
    voucher = row(
        "voucher",
        "val",
        0,
        "I want to use a voucher on a purchase. Product details: Gamma large; it should have color red. "
        "Purchase terms: budget 50 Philippine pesos; platform voucher valid above 20 pesos with a "
        "fixed discount of 5 pesos.",
        ["300"],
    )

    assert atomic_queries_from_visible_query(product) == [
        "Alpha red; it should have size large"
    ]
    assert atomic_queries_from_visible_query(shop) == [
        "Alpha red; it should have size large",
        "Beta blue; it should offer COD",
    ]
    assert atomic_queries_from_visible_query(voucher) == [
        "Gamma large; it should have color red"
    ]
    multi_voucher = row(
        "voucher",
        "val",
        1,
        "I want to use a voucher on a purchase. Product details: "
        "(1) Alpha red; it should have size large; (2) Beta blue; it should offer COD. "
        "Purchase terms: budget 100 Philippine pesos; platform voucher valid above 50 pesos "
        "with a fixed discount of 10 pesos.",
        ["100", "200"],
    )
    assert atomic_queries_from_visible_query(multi_voucher) == [
        "Alpha red; it should have size large",
        "Beta blue; it should offer COD",
    ]


def test_audit_measures_corresponding_and_any_node_recall():
    rows = {
        "train": [
            row(
                "shop",
                "train",
                0,
                "Find one shop that sells all of these items: (1) Alpha Product; it should have color red; "
                "(2) Beta Product; it should offer COD.",
                ["100", "200"],
            )
        ],
        "val": [
            row(
                "voucher",
                "val",
                0,
                "Find this product: Gamma Product; it should have size large. My budget is 50 Philippine "
                "pesos. I have a platform voucher valid above 20 pesos with a fixed discount of 5 pesos.",
                ["300"],
            )
        ],
    }

    report = audit_rows(FakeSearcher(), rows, topks=(1, 2))

    assert report["contract"]["query_builder_reads_gold"] is False
    assert report["overall"]["direct_gold_doc_coverage"] == 1.0
    assert report["overall"]["corresponding_gold_recall@1"] == 2 / 3
    assert report["overall"]["any_node_gold_recall@1"] == 2 / 3
    assert report["overall"]["corresponding_gold_recall@2"] == 1.0
    assert report["overall"]["rows_with_all_corresponding@2"] == 1.0
    assert report["overall"]["retrieval_calls"] == 3


def test_audit_rejects_runtime_gold_or_misaligned_reward_ids():
    bad = row(
        "product",
        "train",
        0,
        "Find this product in the catalog: Alpha Product; it should have color red.",
        ["100"],
    )
    bad["gold_runtime_access"] = True
    with pytest.raises(ValueError, match="runtime leakage"):
        audit_rows(FakeSearcher(), {"train": [bad], "val": []})

    bad["gold_runtime_access"] = False
    bad["reward"]["product_id"] = "999"
    with pytest.raises(ValueError, match="source/reward"):
        audit_rows(FakeSearcher(), {"train": [bad], "val": []})


def test_write_report_allows_protocol_parent_but_refuses_output_reuse(tmp_path):
    output_dir = tmp_path / "immutable-run"
    output_dir.mkdir()
    (output_dir / "protocol.json").write_text("{}\n", encoding="utf-8")
    output = output_dir / "report.json"

    write_report(output, {"status": "ok"})

    assert output.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(output, {"status": "replacement"})
