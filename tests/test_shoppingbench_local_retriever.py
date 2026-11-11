from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "exp" / "shoppingbench_local_retriever.py"
    spec = importlib.util.spec_from_file_location("shoppingbench_local_retriever_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, path: str, params: dict[str, Any] | None = None) -> FakeResponse:
        params = params or {}
        self.calls.append((path, params))
        if path == "/find_product":
            return FakeResponse(
                [
                    {
                        "product_id": "591486855",
                        "shop_id": "1706996",
                        "title": "Arborea Hybrid ghost crash cymbals",
                        "price": 420.0,
                        "service": ["official", "freeShipping"],
                        "sold_count": 10,
                    }
                ]
            )
        if path == "/view_product_information":
            return FakeResponse(
                [
                    {
                        "product_id": "591486855",
                        "description": "B20 cast bronze crash cymbal",
                        "sku_options": {"1": {"size": "18 inch"}},
                        "attributes": {"material": ["B20 bronze"]},
                    }
                ]
            )
        return FakeResponse({"status": "ok"})


def test_plain_query_merges_search_and_product_details():
    module = load_module()
    retriever = module.ShoppingBenchRetriever()
    fake_client = FakeClient()
    retriever._client = fake_client

    results = retriever.search("B20 cast bronze crash cymbals", 5)

    assert len(results) == 1
    contents = results[0]["document"]["contents"]
    assert "Product ID: 591486855" in contents
    assert "B20 cast bronze crash cymbal" in contents
    assert "SKU Options:" in contents
    assert fake_client.calls[0] == (
        "/find_product",
        {"q": "B20 cast bronze crash cymbals", "page": 1},
    )
    assert fake_client.calls[1] == (
        "/view_product_information",
        {"product_ids": "591486855"},
    )


def test_structured_query_preserves_official_filters():
    module = load_module()
    retriever = module.ShoppingBenchRetriever()
    fake_client = FakeClient()
    retriever._client = fake_client
    query = json.dumps(
        {
            "q": "rubber wheel",
            "page": 2,
            "shop_id": "1706996",
            "price": "75-95",
            "sort": "priceasc",
            "service": ["official", "COD", "invalid"],
        }
    )

    retriever.search(query, 3)

    assert fake_client.calls[0] == (
        "/find_product",
        {
            "q": "rubber wheel",
            "page": 2,
            "shop_id": "1706996",
            "price": "75-95",
            "sort": "priceasc",
            "service": "official,COD",
        },
    )


def test_product_id_query_uses_information_endpoint_only():
    module = load_module()
    retriever = module.ShoppingBenchRetriever()
    fake_client = FakeClient()
    retriever._client = fake_client

    results = retriever.search('{"product_ids":"591486855"}', 1)

    assert len(results) == 1
    assert fake_client.calls == [
        ("/view_product_information", {"product_ids": "591486855"})
    ]


def test_invalid_structured_query_returns_no_results():
    module = load_module()
    retriever = module.ShoppingBenchRetriever()
    fake_client = FakeClient()
    retriever._client = fake_client

    assert retriever.search('{"page": 1}', 5) == []
    assert fake_client.calls == []


def test_product_id_query_uses_complete_local_index_when_available():
    module = load_module()
    retriever = module.ShoppingBenchRetriever()
    fake_client = FakeClient()
    retriever._client = fake_client

    class FakeDocument:
        def raw(self):
            return json.dumps(
                {
                    "product": {
                        "product_id": "591486855",
                        "shop_id": "693408",
                        "title": "Arborea Hybrid ghost crash cymbals",
                        "price": 5500.0,
                        "service": [],
                        "sold_count": 1,
                    }
                }
            )

    class FakeSearcher:
        def doc(self, product_id):
            return FakeDocument() if product_id == "591486855" else None

    retriever._searcher = FakeSearcher()
    results = retriever.search('{"product_ids":"591486855"}', 1)

    assert "Shop ID: 693408" in results[0]["document"]["contents"]
    assert "Price: 5500.0" in results[0]["document"]["contents"]
    assert fake_client.calls == []


def test_local_only_mode_rejects_network_fallback(monkeypatch):
    module = load_module()
    retriever = module.ShoppingBenchRetriever()
    monkeypatch.setattr(module, "LOCAL_ONLY", True)

    with pytest.raises(RuntimeError, match="Network fallback is disabled"):
        retriever._request_json("/find_product", {"q": "wheel"})


def test_plain_query_uses_local_index_and_applies_structured_filters():
    module = load_module()
    retriever = module.ShoppingBenchRetriever()
    fake_client = FakeClient()
    retriever._client = fake_client

    products = {
        "doc-a": {
            "product_id": "100",
            "shop_id": "7",
            "title": "Rubber plugs premium",
            "price": 90.0,
            "service": ["official", "COD"],
        },
        "doc-b": {
            "product_id": "200",
            "shop_id": "7",
            "title": "Rubber plugs economy",
            "price": 80.0,
            "service": ["official", "COD"],
        },
        "doc-c": {
            "product_id": "300",
            "shop_id": "8",
            "title": "Rubber plugs wrong shop",
            "price": 85.0,
            "service": ["official", "COD"],
        },
    }

    class FakeDocument:
        def __init__(self, product):
            self.product = product

        def raw(self):
            return json.dumps({"product": self.product})

    class FakeHit:
        def __init__(self, docid, score):
            self.docid = docid
            self.score = score

    class FakeSearcher:
        def search(self, query, topk):
            assert query == "rubber plugs"
            assert topk >= 100
            return [
                FakeHit("doc-a", 3.0),
                FakeHit("doc-b", 2.0),
                FakeHit("doc-c", 1.0),
            ]

        def doc(self, docid):
            return FakeDocument(products[docid])

    retriever._searcher = FakeSearcher()
    query = json.dumps(
        {
            "q": "rubber plugs",
            "shop_id": "7",
            "price": "75-95",
            "service": ["official", "COD"],
            "sort": "priceasc",
        }
    )
    results = retriever.search(query, 3)

    assert [
        line
        for result in results
        for line in result["document"]["contents"].splitlines()
        if line.startswith("Product ID:")
    ] == ["Product ID: 200", "Product ID: 100"]
    assert fake_client.calls == []
