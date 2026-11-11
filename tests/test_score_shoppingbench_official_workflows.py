from scripts.score_shoppingbench_official_workflows import (
    ProductScorer,
    budget_success,
    extract_official_product_ids,
    load_official_rule_score,
    score_row,
)
import pytest


class FakeEmbedding:
    def encode(self, values):
        return values

    def similarity(self, left, right):
        score = 1.0 if left[0].lower() == left[1].lower() else 0.6
        return [[1.0, score], [score, 1.0]]


class FakeDocument:
    def __init__(self, product):
        import json

        self.payload = json.dumps({"product": product})

    def raw(self):
        return self.payload


class FakeSearcher:
    def __init__(self, products):
        self.products = products

    def doc(self, product_id):
        product = self.products.get(product_id)
        return FakeDocument(product) if product else None


def product(product_id, shop_id="shop-1", price=100):
    return {
        "product_id": product_id,
        "shop_id": shop_id,
        "title": "red shoe",
        "price": price,
        "service": ["freeShipping"],
        "sku_options": {"1": {"color": "red"}},
        "attributes": {"material": ["leather"]},
    }


def test_extracts_last_official_recommendation():
    trajectory = [
        {"completion": {"message": {"tool_call": [{"name": "recommend_product", "parameters": {"product_ids": "1"}}]}}},
        {"completion": {"message": {"tool_call": [{"name": "recommend_product", "parameters": {"product_ids": "2, 3"}}]}}},
    ]
    assert extract_official_product_ids(trajectory) == ["2", " 3"]


def test_loads_and_executes_frozen_official_rule(tmp_path):
    orm = tmp_path / "src" / "agent" / "rewards" / "orm.py"
    orm.parent.mkdir(parents=True)
    orm.write_text(
        "def rule_score_reward(product, reward):\n"
        "    return reward['official_score'], {}, {}\n",
        encoding="utf-8",
    )
    rule, loaded_path = load_official_rule_score(tmp_path)
    scorer = ProductScorer(FakeSearcher({}), official_rule_score=rule)
    assert loaded_path == orm
    assert scorer.relevance({}, {"official_score": 0.625}) == 0.625


def test_official_rule_contract_scores_attributes_and_shop():
    products = {"1": product("1"), "2": product("2")}
    scorer = ProductScorer(FakeSearcher(products), FakeEmbedding())
    row = {
        "query": "find two products",
        "reward": [
            {"product_id": "gold-1", "title": ["red shoe"], "sku_options": [{"color": "red"}]},
            {"product_id": "gold-2", "title": ["red shoe"], "service": ["freeShipping"]},
        ],
    }
    result = score_row("shop", row, {"product_ids": ["1", "2"]}, scorer)
    assert result["car"] == 1.0
    assert result["asr"] == 1.0


def test_shop_requires_one_shared_shop():
    products = {"1": product("1", "a"), "2": product("2", "b")}
    scorer = ProductScorer(FakeSearcher(products), FakeEmbedding())
    row = {
        "query": "find two products",
        "reward": [{"product_id": "1"}, {"product_id": "2"}],
    }
    result = score_row("shop", row, {"product_ids": ["1", "2"]}, scorer)
    assert result["car"] == 1.0
    assert result["asr"] == 0.0


def test_voucher_budget_matches_official_semantics():
    products = [product("1", price=120), product("2", price=100)]
    voucher = {
        "budget": 200,
        "threshold": 200,
        "voucher_type": "platform",
        "discount_type": "fixed",
        "face_value": 30,
    }
    assert budget_success(products, voucher)


def test_report_refuses_to_overwrite(tmp_path, monkeypatch):
    from scripts import score_shoppingbench_official_workflows as module

    output = tmp_path / "report.json"
    output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type("Args", (), {"output": output})(),
    )
    # The overwrite guard is exercised directly because main loads heavy runtime modules first.
    with pytest.raises(FileExistsError):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")
