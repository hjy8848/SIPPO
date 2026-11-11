from scripts.probe_adaptive_shop_rag import (
    canonical_query,
    constraint_status,
    run_probe,
)


def product(product_id, shop_id, *, color):
    return {
        "product_id": str(product_id),
        "shop_id": str(shop_id),
        "title": f"Product {product_id}",
        "price": 10.0,
        "attributes": {"color_family": [color]},
    }


class FakeRetriever:
    def _search_local_index(self, query, topk):
        q = query["q"]
        shop = str(query.get("shop_id") or "")
        if q == "Alpha red" and not shop:
            return [product(11, 100, color="red")]
        if q == "Beta blue" and not shop:
            return [product(22, 200, color="blue")]
        if q == "Beta blue" and shop == "100":
            return [product(33, 100, color="blue")]
        return []


class GlobalExpansionRetriever:
    def _search_local_index(self, query, topk):
        q = query["q"]
        page = int(query.get("page", 1))
        shop = str(query.get("shop_id") or "")
        if shop:
            return []
        if q == "Alpha red" and page == 1:
            return [product(11, 100, color="red")]
        if q == "Beta blue" and page == 1:
            return [product(22, 200, color="blue")]
        if q == "Alpha red" and page == 2:
            return [product(44, 300, color="red")]
        if q == "Beta blue" and page == 2:
            return [product(55, 300, color="blue")]
        return []


def test_canonical_query_keeps_title_and_constraint_value():
    assert canonical_query("Alpha; it should have color family red") == "Alpha red"


def test_constraint_status_uses_structured_attributes():
    candidate = product(11, 100, color="red")
    assert constraint_status(
        "Alpha; it should have color family red", candidate, ["color family"]
    ) == "satisfies"
    assert constraint_status(
        "Alpha; it should have color family blue", candidate, ["color family"]
    ) == "contradicts"


def test_probe_repairs_missing_slot_with_shop_conditioned_search():
    question = (
        "Find one shop. Product details: "
        "(1) Alpha; it should have color family red; "
        "(2) Beta; it should have color family blue."
    )
    report = run_probe(question, FakeRetriever(), max_pages=1, shop_beam=1)
    assert report["predicted_product_ids"] == ["11", "33"]
    assert report["selected_shop_id"] == "100"
    assert [action["action"] for action in report["actions"]] == [
        "search_global",
        "search_global",
        "search_in_shop",
    ]
    assert report["decision"]["constraint_statuses"] == ["satisfies", "satisfies"]


def test_probe_discovers_new_shop_by_global_page_expansion():
    question = (
        "Find one shop. Product details: "
        "(1) Alpha; it should have color family red; "
        "(2) Beta; it should have color family blue."
    )
    report = run_probe(
        question,
        GlobalExpansionRetriever(),
        max_pages=2,
        shop_beam=1,
        max_search_calls=8,
    )
    assert report["predicted_product_ids"] == ["44", "55"]
    assert report["selected_shop_id"] == "300"
    assert report["search_calls"] == 5
    assert report["actions"][-2]["page"] == 2
    assert report["actions"][-1]["page"] == 2
