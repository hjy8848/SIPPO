import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shopping_protocol_has_unlimited_prompt_and_official_output_schema():
    from qa_manager.ROSA import shoppingcomp_protocol as protocol

    unlimited = protocol.build_planner_system_message(0)
    capped = protocol.build_planner_system_message(8)
    assert "Use at most" not in unlimited
    assert "Use at most 8 nodes" in capped
    assert "<candidate_product_list>" in protocol.FINAL_SYSTEM_MSG
    assert "<product_name>" in protocol.FINAL_SYSTEM_MSG
    assert "<check_item>" in protocol.FINAL_SYSTEM_MSG
    assert "<safety_assessment>" in protocol.FINAL_SYSTEM_MSG
    assert "ONE_ATOMIC_USER_REQUIREMENT" not in protocol.FINAL_SYSTEM_MSG
    assert "EVIDENCE_BACKED_PRODUCT_FACTS" not in protocol.SEARCH_SYSTEM_MSG


def test_rosa_server_selects_shopping_profile_without_changing_default():
    source = (ROOT / "exp" / "rosa_server.py").read_text(encoding="utf-8")
    assert 'os.environ.get("ROSA_PROTOCOL_PROFILE", "qa")' in source
    assert 'if PROTOCOL_PROFILE == "shoppingcomp"' in source
    assert 'elif PROTOCOL_PROFILE != "qa"' in source
    assert '"protocol_profile": PROTOCOL_PROFILE' in source
    assert '"max_model_len": VLLM_MAX_MODEL_LEN' in source
    assert "max_tokens=FINAL_GEN_MAX_TOKENS" in source
    assert "final_answer_format_issue" in source
    assert "no grounded shopping evidence" in source
    assert "is_retrieval_failure_context(search_ctx)" in source


def test_prediction_adapter_extracts_products_and_safety():
    runner = load_module(
        "scripts/run_shoppingcomp_predictions.py", "shoppingcomp_prediction_runner_test"
    )
    response = """
    <candidate_product_list>
      <product><product_name>Sony Alpha 6700</product_name>
        <check_item><demand>4K video</demand><reason>Supported by source</reason>
        <is_satisfied>true</is_satisfied></check_item></product>
      <product><product_name>Canon EOS R7</product_name>
        <check_item><demand>4K video</demand><reason>Supported by source</reason>
        <is_satisfied>true</is_satisfied></check_item></product>
      <product><product_name>sony alpha 6700</product_name>
        <check_item><demand>4K video</demand><reason>Supported by source</reason>
        <is_satisfied>true</is_satisfied></check_item></product>
    </candidate_product_list>
    """
    parsed, products, error = runner.extract_response(response)
    assert parsed.startswith("<candidate_product_list>")
    assert products == ["Sony Alpha 6700", "Canon EOS R7"]
    assert error is None

    parsed, products, error = runner.extract_response(
        "<safety_assessment>This request is unsafe.</safety_assessment>"
    )
    assert parsed.startswith("<safety_assessment>")
    assert products == []
    assert error is None

    parsed, products, error = runner.extract_response(
        "<candidate_product_list><product><product_name>Camera</product_name>"
        "<check_item><demand>ONE_ATOMIC_USER_REQUIREMENT</demand>"
        "<reason>SUPPORTED_FINDING_WITH_SOURCE_URL</reason>"
        "<is_satisfied>true</isatisfied></check_item></product>"
        "</candidate_product_list>"
    )
    assert parsed.startswith("<candidate_product_list>")
    assert products == []
    assert error is not None


def test_known_is_satisfied_closing_tag_typo_is_normalized():
    from qa_manager.ROSA import shoppingcomp_protocol as protocol

    answer = (
        "<candidate_product_list><product><product_name>Camera X</product_name>"
        "<check_item><demand>4K video</demand><reason>Supported by source</reason>"
        "<is_satisfied>true</isatisfied></check_item></product>"
        "</candidate_product_list>"
    )
    normalized = protocol.normalize_final_answer(answer)
    assert "</isatisfied>" not in normalized
    assert "</is_satisfied>" in normalized
    assert protocol.validate_final_answer(normalized) is None


def test_dataset_loader_does_not_copy_gold_fields(tmp_path):
    runner = load_module(
        "scripts/run_shoppingcomp_predictions.py", "shoppingcomp_dataset_runner_test"
    )
    dataset = tmp_path / "shopping.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "question": "Find a camera",
                "product_list": [{"product_name": "SECRET"}],
                "scene_list": [{"rubric": "SECRET"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    items = runner.load_items(dataset)
    assert items == [runner.ShoppingItem(uuid="u1", question="Find a camera", position=0)]
    assert not hasattr(items[0], "product_list")


def test_journal_materialization_is_ordered_and_auditable(tmp_path):
    runner = load_module(
        "scripts/run_shoppingcomp_predictions.py", "shoppingcomp_journal_runner_test"
    )
    items = [
        runner.ShoppingItem(uuid="u1", question="q1", position=0),
        runner.ShoppingItem(uuid="u2", question="q2", position=1),
    ]

    def journal_row(uuid, question, success):
        return {
            "uuid": uuid,
            "prediction": {
                "uuid": uuid,
                "question": question,
                "model_results": [
                    {
                        "model": "model",
                        "products": ["p"] if success else [],
                        "response": "response",
                        "success": success,
                    }
                ],
            },
            "trajectory": {
                "uuid": uuid,
                "search_nodes": 1,
                "retrieval_calls": 2,
                "format_error": None if success else "bad format",
                "contract_error": None,
                "pipeline_error": None,
                "request_error": None,
                "raw_trace": {"protocol_profile": "shoppingcomp"},
            },
        }

    rows = {
        "u2": journal_row("u2", "q2", False),
        "u1": journal_row("u1", "q1", True),
    }
    summary = runner.materialize(items, rows, tmp_path)
    predictions = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [row["uuid"] for row in predictions] == ["u1", "u2"]
    assert summary == {
        "expected": 2,
        "completed": 2,
        "pending": 0,
        "successful": 1,
        "failures": 1,
        "empty_product_lists": 1,
        "search_nodes": 2,
        "retrieval_calls": 4,
        "failure_uuids": ["u2"],
    }


def test_rosa_trace_becomes_official_prediction(monkeypatch):
    runner = load_module(
        "scripts/run_shoppingcomp_predictions.py", "shoppingcomp_request_runner_test"
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "protocol_profile": "shoppingcomp",
                "final_answer": (
                    "<candidate_product_list><product>"
                    "<product_name>Sony Alpha 6700</product_name>"
                    "<check_item><demand>4K video</demand>"
                    "<reason>Supported by https://example.com/spec</reason>"
                    "<is_satisfied>true</is_satisfied></check_item>"
                    "</product></candidate_product_list>"
                ),
                "nodes_execution": [
                    {
                        "id": "1",
                        "type": "search",
                        "attempted_queries": ["camera specs"],
                        "search_context": "URL: https://example.com/spec",
                    },
                    {"id": "2", "type": "reason"},
                ],
            }

    monkeypatch.setattr(runner.requests, "post", lambda *args, **kwargs: FakeResponse())
    row = runner.request_prediction(
        runner.ShoppingItem(uuid="u1", question="Find a camera", position=0),
        api_url="http://localhost/rosa/chat",
        model_name="DAGENT-PSC",
        timeout=10,
        retries=0,
    )
    model_result = row["prediction"]["model_results"][0]
    assert model_result["products"] == ["Sony Alpha 6700"]
    assert model_result["success"] is True
    assert row["trajectory"]["search_nodes"] == 1
    assert row["trajectory"]["retrieval_calls"] == 1
    assert row["trajectory"]["raw_trace"]["protocol_profile"] == "shoppingcomp"


def test_search_cache_prevents_duplicate_live_search(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_test"
    )
    cache = retriever_module.SearchCache(tmp_path / "isolated.sqlite3")
    retriever = retriever_module.FreeWebRetriever(cache)
    calls = []

    def fake_search(query, topk):
        calls.append((query, topk))
        return [{"title": "Camera", "url": "https://example.com/camera", "snippet": "Specs"}]

    retriever._live_search = fake_search
    retriever._page_text = lambda url: "Official specification text"
    first = retriever.search("  camera   specs ", 1)
    second = retriever.search("camera specs", 1)
    assert calls == [("camera specs", 1)]
    assert first == second
    assert "URL: https://example.com/camera" in first[0]["document"]["contents"]


def test_live_search_prefers_duckduckgo_html(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_fallback_test"
    )
    retriever = retriever_module.FreeWebRetriever(
        retriever_module.SearchCache(tmp_path / "fallback.sqlite3")
    )

    html_rows = [
        {"title": "Example", "url": "https://example.com/item", "snippet": "Details"}
    ]
    monkeypatch.setattr(
        retriever, "_duckduckgo_html_search", lambda query, topk: html_rows
    )
    monkeypatch.setattr(
        retriever,
        "_ddgs_search",
        lambda query, topk: pytest.fail("ddgs should not run after HTML succeeds"),
    )

    assert retriever._live_search("example item", 1) == html_rows


def test_live_search_uses_searxng_json_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    monkeypatch.setenv("SHOPPING_SEARCH_BACKEND", "searxng")
    monkeypatch.setenv("SHOPPING_SEARXNG_URL", "http://10.0.0.8:18888")
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_searxng_test"
    )
    retriever = retriever_module.FreeWebRetriever(
        retriever_module.SearchCache(tmp_path / "searxng.sqlite3")
    )
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "Official Camera Specs",
                        "url": "https://example.com/specs",
                        "content": "Weather sealing and 4K video specifications.",
                    }
                ]
            }

    class FakeClient:
        def get(self, url, *, params, timeout):
            seen.update({"url": url, "params": params, "timeout": timeout})
            return FakeResponse()

    retriever._searxng_client = FakeClient()
    monkeypatch.setattr(
        retriever,
        "_duckduckgo_html_search",
        lambda query, topk: pytest.fail("direct HTML search must not run"),
    )
    rows = retriever._live_search("official camera specifications", 1)
    assert seen["url"] == "http://10.0.0.8:18888/search"
    assert seen["params"]["format"] == "json"
    assert seen["params"]["engines"] == "google,bing,duckduckgo"
    assert rows == [
        {
            "title": "Official Camera Specs",
            "url": "https://example.com/specs",
            "snippet": "Weather sealing and 4K video specifications.",
            "search_query": "official camera specifications",
        }
    ]


def test_live_search_uses_ddgs_when_html_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_ddgs_fallback_test"
    )
    retriever = retriever_module.FreeWebRetriever(
        retriever_module.SearchCache(tmp_path / "fallback.sqlite3")
    )
    ddgs_rows = [
        {"title": "Backup", "url": "https://example.com/backup", "snippet": "Details"}
    ]
    monkeypatch.setattr(
        retriever,
        "_duckduckgo_html_search",
        lambda query, topk: (_ for _ in ()).throw(RuntimeError("HTML unavailable")),
    )
    monkeypatch.setattr(retriever, "_bing_html_search", lambda query, topk: [])
    monkeypatch.setattr(retriever, "_ddgs_search", lambda query, topk: ddgs_rows)

    assert retriever._live_search("example item", 1) == ddgs_rows


def test_live_search_uses_bing_when_duckduckgo_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_bing_test"
    )
    retriever = retriever_module.FreeWebRetriever(
        retriever_module.SearchCache(tmp_path / "bing.sqlite3")
    )
    bing_rows = [
        {"title": "Bing result", "url": "https://example.com/bing", "snippet": "Facts"}
    ]
    monkeypatch.setattr(retriever, "_duckduckgo_html_search", lambda query, topk: [])
    monkeypatch.setattr(retriever, "_bing_html_search", lambda query, topk: bing_rows)
    monkeypatch.setattr(
        retriever,
        "_ddgs_search",
        lambda query, topk: pytest.fail("ddgs should not run after Bing succeeds"),
    )
    assert retriever._live_search("example item", 1) == bing_rows


def test_bing_redirect_is_unwrapped(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_bing_url_test"
    )
    target = "https://example.com/product?a=1"
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    wrapped = f"https://www.bing.com/ck/a?u=a1{encoded}&ntb=1"
    assert retriever_module.FreeWebRetriever._unwrap_bing_url(wrapped) == target


def test_bing_relevance_filter_rejects_unrelated_results(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_relevance_test"
    )
    relevant = retriever_module.FreeWebRetriever._is_relevant_result(
        "mirrorless travel camera 4K stabilization",
        "Best mirrorless travel cameras",
        "Compare 4K video and image stabilization.",
    )
    unrelated = retriever_module.FreeWebRetriever._is_relevant_result(
        "mirrorless travel camera 4K stabilization",
        "Bing weekly news quiz",
        "Test your knowledge and earn rewards.",
    )
    assert relevant is True
    assert unrelated is False


def test_live_search_compacts_long_queries(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_compact_test"
    )
    retriever = retriever_module.FreeWebRetriever(
        retriever_module.SearchCache(tmp_path / "compact.sqlite3")
    )
    seen = []

    def html_search(query, topk):
        seen.append(query)
        return [{"title": "Result", "url": "https://example.com", "snippet": "Facts"}]

    monkeypatch.setattr(retriever, "_duckduckgo_html_search", html_search)
    long_query = " ".join(f"term{index}" for index in range(100))
    rows = retriever._live_search(long_query, 1)
    assert len(seen[0].split()) <= retriever_module.QUERY_MAX_WORDS
    assert len(seen[0]) <= retriever_module.QUERY_MAX_CHARS
    assert rows[0]["search_query"] == seen[0]


def test_no_results_are_not_reported_as_server_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_empty_test"
    )
    retriever = retriever_module.FreeWebRetriever(
        retriever_module.SearchCache(tmp_path / "empty.sqlite3")
    )
    monkeypatch.setattr(retriever, "_duckduckgo_html_search", lambda query, topk: [])
    monkeypatch.setattr(retriever, "_bing_html_search", lambda query, topk: [])
    monkeypatch.setattr(
        retriever,
        "_ddgs_search",
        lambda query, topk: (_ for _ in ()).throw(RuntimeError("backend unavailable")),
    )
    assert retriever._live_search("no matching product", 2) == []


def test_all_empty_backends_return_clean_empty_result(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_all_empty_test"
    )
    retriever = retriever_module.FreeWebRetriever(
        retriever_module.SearchCache(tmp_path / "all-empty.sqlite3")
    )
    monkeypatch.setattr(retriever, "_duckduckgo_html_search", lambda query, topk: [])
    monkeypatch.setattr(retriever, "_bing_html_search", lambda query, topk: [])
    monkeypatch.setattr(retriever, "_ddgs_search", lambda query, topk: [])
    assert retriever._live_search("no matching product", 2) == []


def test_duckduckgo_redirect_is_unwrapped(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_SEARCH_CACHE", str(tmp_path / "cache.sqlite3"))
    retriever_module = load_module(
        "exp/shoppingcomp_free_retriever.py", "shoppingcomp_retriever_redirect_test"
    )
    wrapped = (
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fproduct%3Fa%3D1"
        "&rut=unused"
    )
    assert (
        retriever_module.FreeWebRetriever._unwrap_duckduckgo_url(wrapped)
        == "https://example.com/product?a=1"
    )


def test_launcher_freezes_free_search_and_no_judge_contract():
    source = (ROOT / "exp" / "evaluate_dagent_shoppingcomp_free_search.sh").read_text(
        encoding="utf-8"
    )
    required = (
        "SHOPPING_SEARCH_CACHE",
        "SHOPPING_SEARCH_BACKEND",
        "SHOPPING_SEARXNG_URL",
        "SHOPPING_PROXY_URL",
        "NO_PROXY",
        "TRITON_CACHE_DIR",
        "EXTERNAL_RETRIEVER_URL",
        "SHOPPING_DEP_DIR",
        "import bs4",
        "ROSA_PROTOCOL_PROFILE=shoppingcomp",
        "MAX_NODES_PER_TURN",
        "PLANNER_PROMPT_MAX_NODES",
        "FINAL_GEN_MAX_TOKENS",
        "run_shoppingcomp_predictions.py",
        "judge: not run",
        "--query-compute-apps=pid",
    )
    for value in required:
        assert value in source
    assert "ShoppingCompJudge" not in source
    for forbidden in ("ray stop", "pkill", "rm -", "nvidia-smi --gpu-reset"):
        assert forbidden not in source


def test_shopping_protocol_rejects_retrieval_failure_contexts():
    from qa_manager.ROSA import shoppingcomp_protocol as protocol

    assert protocol.is_retrieval_failure_context("[Search API Error]: timed out")
    assert protocol.is_retrieval_failure_context("[No Documents Found] for 'camera'")
    assert protocol.is_retrieval_failure_context("[No query provided]")
    assert not protocol.is_retrieval_failure_context(
        "Official Camera Specs\nURL: https://example.com/specs"
    )


def test_windows_bridge_uses_local_cache_and_loopback_reverse_tunnel():
    source = (ROOT / "scripts" / "run_shoppingcomp_free_search_bridge.ps1").read_text(
        encoding="utf-8"
    )
    assert "SHOPPING_SEARCH_BACKEND = \"duckduckgo\"" in source
    assert "SHOPPING_SEARCH_CACHE = $CachePath" in source
    assert '"-R", "127.0.0.1:${RemotePort}:127.0.0.1:${LocalPort}"' in source
    assert "ExitOnForwardFailure=yes" in source
    assert "ServerAliveInterval=30" in source
    assert "-WindowStyle Hidden" in source
    assert "http://127.0.0.1:$RemotePort/health" in source
