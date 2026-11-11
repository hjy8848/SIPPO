from copy import deepcopy

from qa_manager.ROSA.shoppingbench_product_selection_fallback import (
    apply_conservative_product_selection,
    structured_product_fields,
)


def _product(product_id, *, selected=False, sku_options=None, attributes=None):
    return {
        "product_id": str(product_id),
        "model_selected": selected,
        "sku_options": sku_options,
        "attributes": attributes,
    }


def test_unique_satisfier_replaces_a_contradicting_model_selection():
    products = [
        _product(1, selected=True, sku_options='{"1": {"color_family": "carbon"}}'),
        _product(2, sku_options='{"1": {"color_family": "silver"}}'),
    ]

    selected, metadata = apply_conservative_product_selection(
        "Basket; it should have color family silver", products
    )

    assert [row["product_id"] for row in selected if row["model_selected"]] == ["2"]
    assert metadata["applied"] is True
    assert metadata["decision"] == "unique_satisfier_replaces_non_satisfier"
    assert metadata["gold_used"] is False


def test_unique_satisfier_replaces_an_unverified_model_selection():
    products = [
        _product(1, selected=True, attributes='{"material": ["steel"]}'),
        _product(2, attributes='{"strainer": ["rectangle - 20x15cm"]}'),
    ]

    selected, metadata = apply_conservative_product_selection(
        "Basket; it should have strainer rectangle - 20x15cm", products
    )

    assert [row["product_id"] for row in selected if row["model_selected"]] == ["2"]
    assert metadata["baseline_status"] == "unverified"
    assert metadata["applied"] is True


def test_multiple_satisfiers_preserve_the_model_selection():
    products = [
        _product(1, selected=True, attributes='{"brand": ["other"]}'),
        _product(2, attributes='{"brand": ["pitsbike"]}'),
        _product(3, attributes='{"brand": ["pitsbike"]}'),
    ]

    selected, metadata = apply_conservative_product_selection(
        "Part; it should have brand pitsbike", products
    )

    assert [row["product_id"] for row in selected if row["model_selected"]] == ["1"]
    assert metadata["applied"] is False


def test_satisfying_model_selection_is_preserved():
    products = [
        _product(1, selected=True, attributes='{"brand": ["pitsbike"]}'),
        _product(2, attributes='{"brand": ["other"]}'),
    ]

    selected, metadata = apply_conservative_product_selection(
        "Part; it should have brand pitsbike", products
    )

    assert [row["product_id"] for row in selected if row["model_selected"]] == ["1"]
    assert metadata["baseline_status"] == "satisfies"
    assert metadata["applied"] is False


def test_missing_constraint_and_missing_model_selection_fail_closed():
    products = [_product(1, attributes='{"brand": ["pitsbike"]}')]

    selected, metadata = apply_conservative_product_selection("Find a useful part", products)

    assert not any(row["model_selected"] for row in selected)
    assert metadata["baseline_id"] == ""
    assert metadata["applied"] is False


def test_scalar_list_keeps_its_parent_key():
    fields = structured_product_fields(
        _product(1, attributes='{"specifications": {"brand": ["PitsBike", "PB"]}}')
    )

    assert fields["brand"] == ["PitsBike", "PB"]


def test_input_products_are_not_mutated():
    products = [
        _product(1, selected=True, attributes='{"color_family": ["black"]}'),
        _product(2, attributes='{"color_family": ["silver"]}'),
    ]
    original = deepcopy(products)

    apply_conservative_product_selection(
        "Basket; it should have color family silver", products
    )

    assert products == original
