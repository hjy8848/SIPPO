import pytest

from scripts.audit_shoppingbench_voucher_compiler_coverage import audit_rows


class Hit:
    def __init__(self, docid):
        self.docid = docid


class Searcher:
    results = {
        "first item": ["100", "noise"],
        "second item": ["200", "noise"],
        "single item": ["300", "noise"],
    }

    def search(self, query, topk):
        return [Hit(value) for value in self.results[query][:topk]]


def prediction(sample_id, expected, raw_queries, compiled_queries):
    raw = "<plan>" + "".join(
        f'<search id="{index}">q={query}</search>'
        for index, query in enumerate(raw_queries, start=1)
    ) + "</plan>"
    return {
        "sample_id": sample_id,
        "expected_product_ids": expected,
        "raw_trace": {
            "raw_plan_output": raw,
            "contract_protocol": {
                "plan_compiler": {"queries": compiled_queries, "gold_used": False}
            },
        },
    }


def test_expanded_model_queries_improve_retrieval_with_identity_control():
    rows = [
        prediction("voucher-0000", ["100", "200"], ["first item", "second item"], ["first item"]),
        prediction("voucher-0001", ["300"], ["single item"], ["single item"]),
    ]

    report, passed = audit_rows(Searcher(), rows, topk=2, max_searches=4)

    assert passed is True
    assert all(report["gates"].values())
    assert report["overall"]["current_corresponding_hits_rate"] == 2 / 3
    assert report["overall"]["expanded_corresponding_hits_rate"] == 1.0
    assert report["overall"]["corresponding_recall_delta"] == pytest.approx(1 / 3)
    assert report["overall"]["expanded_rows"] == 1
    assert report["scope_limit"].startswith("retrieval coverage only")
