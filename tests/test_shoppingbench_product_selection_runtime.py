import asyncio

from qa_manager.ROSA import shoppingbench_protocol
from qa_manager.ROSA.shoppingbench_product_selection_runtime import (
    annotate_product_selection,
    product_selection_decisions,
    product_selection_request,
    select_supported_products,
)


def _products(field, satisfying_value):
    return [
        {
            "product_id": "1",
            "attributes": '{"material": ["steel"]}',
        },
        {
            "product_id": "2",
            "attributes": f'{{"{field}": ["{satisfying_value}"]}}',
        },
    ]


def test_runtime_selection_and_annotation_use_the_same_corrected_product():
    products = _products("color_family", "silver")
    conclusion = "<conclusion>Product ID: 1</conclusion>"

    with product_selection_request("Basket; it should have color family silver"):
        selected = select_supported_products(products, conclusion)
        annotated = annotate_product_selection(products, conclusion)
        decisions = product_selection_decisions()

    assert [row["product_id"] for row in selected] == ["2"]
    assert [row["product_id"] for row in annotated if row["model_selected"]] == ["2"]
    assert len(decisions) == 1
    assert decisions[0]["baseline_id"] == "1"
    assert decisions[0]["selected_id"] == "2"
    assert decisions[0]["gold_used"] is False


def test_request_contexts_do_not_leak_across_concurrent_tasks():
    async def run(question, field, value):
        with product_selection_request(question):
            await asyncio.sleep(0)
            annotate_product_selection(
                _products(field, value),
                "<conclusion>Product ID: 1</conclusion>",
            )
            await asyncio.sleep(0)
            return product_selection_decisions()

    async def gather():
        return await asyncio.gather(
            run("Basket; it should have color family silver", "color_family", "silver"),
            run("Part; it should have brand pitsbike", "brand", "pitsbike"),
        )

    silver, pitsbike = asyncio.run(gather())

    assert silver[0]["constraint_field"] == "color family"
    assert silver[0]["constraint_value"] == "silver"
    assert pitsbike[0]["constraint_field"] == "brand"
    assert pitsbike[0]["constraint_value"] == "pitsbike"


def test_runtime_functions_remain_nonrecursive_after_protocol_monkeypatch(monkeypatch):
    monkeypatch.setattr(
        shoppingbench_protocol, "select_supported_products", select_supported_products
    )
    monkeypatch.setattr(
        shoppingbench_protocol, "annotate_product_selection", annotate_product_selection
    )

    with product_selection_request("Part; it should have brand pitsbike"):
        annotated = shoppingbench_protocol.annotate_product_selection(
            _products("brand", "pitsbike"),
            "<conclusion>Product ID: 1</conclusion>",
        )

    assert [row["product_id"] for row in annotated if row["model_selected"]] == ["2"]
