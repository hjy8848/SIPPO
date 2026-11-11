import pytest

from qa_manager.ROSA.shoppingbench_product_query_fallback import (
    invalid_product_query,
    select_product_query,
    visible_single_product_query,
)


def test_visible_single_product_query_reads_only_product_details():
    question = (
        "Could you locate this? Product details: Alpha desk; "
        "it should have color family black."
    )
    assert visible_single_product_query(question) == (
        "Alpha desk; it should have color family black"
    )
    assert visible_single_product_query(
        "Product details: (1) Alpha; (2) Beta."
    ) == ""


@pytest.mark.parametrize(
    "query",
    (
        "",
        "PLAIN_QUERY_OR_JSON_FILTER",
        "SEARCH_QUERY_OR_JSON_FILTER",
        "Beybiko packs&amp;size[]=diaper s",
        "Beybiko packs&size[]=diaper s",
    ),
)
def test_invalid_product_query_rejects_observed_failure_shapes(query):
    assert invalid_product_query(query)


def test_product_query_fallback_preserves_valid_model_query():
    query, metadata = select_product_query(
        "Product details: Alpha desk; it should have color black.",
        ["model-authored alpha desk"],
    )
    assert query == "model-authored alpha desk"
    assert metadata == {
        "applied": False,
        "reason": "model_query_valid",
        "gold_used": False,
    }


def test_product_query_fallback_replaces_invalid_query_without_gold():
    query, metadata = select_product_query(
        "Product details: Alpha desk; it should have color black.",
        ["PLAIN_QUERY_OR_JSON_FILTER"],
    )
    assert query == "Alpha desk; it should have color black"
    assert metadata["applied"] is True
    assert metadata["gold_used"] is False


def test_product_query_fallback_fails_closed_without_visible_description():
    with pytest.raises(ValueError, match="no visible single-product fallback"):
        select_product_query("Find me something.", ["PLAIN_QUERY_OR_JSON_FILTER"])
