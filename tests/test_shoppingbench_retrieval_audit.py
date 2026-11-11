from scripts.audit_shoppingbench_local_retrieval import audit_rows


class Hit:
    def __init__(self, docid):
        self.docid = docid


class FakeSearcher:
    def __init__(self):
        self.results = {
            "find alpha and beta": ["noise", "100"],
            "find gamma": ["noise"],
            "Alpha Product": ["100"],
            "Beta Product": ["200"],
            "Gamma Product": ["300"],
        }

    def search(self, query, topk):
        return [Hit(value) for value in self.results.get(query, [])[:topk]]

    def doc(self, docid):
        return object() if docid in {"100", "200", "300"} else None


def test_audit_separates_raw_retrieval_from_privileged_title_oracle():
    rows = {
        "shop": [
            {
                "query": "find alpha and beta",
                "reward": [
                    {"product_id": "100", "title": ["Alpha Product"]},
                    {"product_id": "200", "title": ["Beta Product"]},
                ],
            }
        ],
        "product": [
            {
                "query": "find gamma",
                "reward": {"product_id": "300", "title": ["Gamma Product"]},
            }
        ],
    }

    report = audit_rows(FakeSearcher(), rows, topks=(1, 3))

    assert report["contract"]["external_api_calls"] == 0
    assert report["overall"]["direct_gold_doc_coverage"] == 1.0
    raw = report["overall"]["modes"]["raw_question"]
    oracle = report["overall"]["modes"]["gold_title_oracle"]
    assert raw["gold_recall@1"] == 0.0
    assert raw["gold_recall@3"] == 1 / 3
    assert raw["rows_with_all_gold@3"] == 0.0
    assert oracle["gold_recall@1"] == 1.0
    assert oracle["rows_with_all_gold@1"] == 1.0
    assert oracle["retrieval_calls"] == 3
