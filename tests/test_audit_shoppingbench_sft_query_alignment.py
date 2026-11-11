import json

from scripts.audit_shoppingbench_sft_query_alignment import audit, summarize


def row(task, query, titles):
    products = [
        {"product_id": str(index), "title": [title]}
        for index, title in enumerate(titles, start=1)
    ]
    return {
        "task": task,
        "query": query,
        "reward": products[0] if task == "product" else products,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")


def test_summarize_detects_title_copy_and_template_collapse():
    rows = [
        row("product", f"Find this product: Exact Catalog Title {index}", [f"Exact Catalog Title {index}"])
        for index in range(4)
    ]

    report = summarize(rows)

    assert report["any_gold_title_exact_substring_rate"] == 1.0
    assert report["mean_gold_title_token_recall_in_query"] == 1.0
    assert report["top_five_token_prefix_share"] == 1.0
    assert report["product_cardinality_histogram"] == {"1": 4}


def test_audit_fails_leakage_diversity_and_cardinality_without_using_formal_for_thresholds(tmp_path):
    train = []
    val = []
    formal_dir = tmp_path / "formal"
    formal_dir.mkdir()
    for task in ("product", "shop", "voucher"):
        count = 1 if task != "shop" else 3
        titles = [f"Catalog Item {task} {index}" for index in range(count)]
        train.append(row(task, "Find this product: " + " and ".join(titles), titles))
        val.append(row(task, "Find this product: " + " plus ".join(titles), titles))
        write_jsonl(
            formal_dir / f"synthesize_{task}_test.jsonl",
            [row(task, f"Natural request for {task}", [f"Hidden Formal Title {task}"])],
        )
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    report, passed = audit(
        train_path,
        val_path,
        formal_dir,
        max_exact_title_rate=0.1,
        max_title_token_recall=0.7,
        max_prefix_share=0.25,
    )

    assert passed is False
    assert report["gates"]["formal_query_disjoint"] is True
    assert report["gates"]["no_exact_title_leakage"] is False
    assert report["gates"]["template_diversity"] is False
    assert report["gates"]["task_cardinality_coverage"] is False
    assert report["formal_test_role"].startswith("comparison only")
