from qa_manager.ROSA.contract_protocol import parse_contract_plan
from qa_manager.ROSA.shoppingbench_protocol import (
    classify_shopping_request,
    compile_shoppingbench_plan,
    extract_atomic_search_queries,
    annotate_product_selection,
    build_contract_planner_system_message,
    build_planner_system_message,
    compile_contract_search_query,
    execute_artifact_reason,
    extract_product_ids,
    extract_visible_product_queries,
    parse_product_set,
    product_set_conclusion,
    select_supported_products,
    serialize_product_sink,
    validate_final_answer,
)


def test_request_classifier_recognizes_shared_merchant_semantics():
    shop_requests = (
        "I need one seller for several items.",
        "Please find one store for all requested items.",
        "I want these products from a single seller.",
        "Find a common shop for the following items.",
        "Show me a store that carries everything.",
        "Find a shop that sells all these products.",
    )
    assert all(classify_shopping_request(question) == "shop" for question in shop_requests)


def test_request_classifier_keeps_voucher_priority_and_product_default():
    assert classify_shopping_request(
        "Find several products. Purchase terms: budget 500 with a platform voucher discount."
    ) == "voucher"
    assert classify_shopping_request(
        "Find a product sold by the Acme store."
    ) == "product"


def test_compiler_owns_shop_topology_while_preserving_model_queries():
    raw_plan = """<plan sink="sink1">
    <search id="1">q=red belt&amp;q=black belt&amp;q=brown belt</search>
    <reason id="2" op="invented">free text</reason>
    </plan>"""
    plan, metadata = compile_shoppingbench_plan(
        "Find a shop that sells all these products: red, black, and brown belts.",
        raw_plan,
    )
    parsed = parse_contract_plan(plan)
    assert [node["id"] for node in parsed["nodes"]] == ["1", "2", "3", "4"]
    assert parsed["nodes"][-1]["op"] == "join_shop"
    assert parsed["nodes"][-1]["input_ids"] == ["1", "2", "3"]
    assert parsed["attrs"]["sink"] == "#4#"
    assert metadata["queries"] == ["red belt", "black belt", "brown belt"]
    assert metadata["gold_used"] is False


def test_compiler_ignores_model_ids_sink_operator_and_invented_filters():
    raw_plan = """<plan sink="1">
    <search id="9">q: oil crayons shop_id:1 sort:priceasc</search>
    <reason id="9" op="merge_vouchers" inputs="9">12345</reason>
    </plan>"""
    plan, metadata = compile_shoppingbench_plan(
        "Looking for oil crayons. My budget is 425 and I have a voucher.",
        raw_plan,
    )
    parsed = parse_contract_plan(plan)
    assert [node["id"] for node in parsed["nodes"]] == ["1", "2"]
    assert parsed["nodes"][0]["task"] == "oil crayons"
    assert parsed["nodes"][1]["op"] == "apply_voucher"
    assert "12345" not in plan
    assert metadata["request_kind"] == "voucher"


def test_compiler_preserves_multiple_voucher_queries_with_bounded_topology():
    raw_plan = """<plan>
    <search id="9">q=first item</search>
    <search id="4">q=second item</search>
    <search id="7">q=third item</search>
    <search id="2">q=fourth item</search>
    <search id="1">q=fifth item</search>
    </plan>"""

    plan, metadata = compile_shoppingbench_plan(
        "Find these products with a platform voucher and a fixed budget.",
        raw_plan,
        max_nodes=5,
    )
    parsed = parse_contract_plan(plan)

    assert metadata["queries"] == ["first item", "second item", "third item", "fourth item"]
    assert metadata["compiler"] == "shoppingbench-system-topology-v2-voucher-multisearch"
    assert [node["id"] for node in parsed["nodes"]] == ["1", "2", "3", "4", "5"]
    assert parsed["nodes"][-1]["op"] == "apply_voucher"
    assert parsed["nodes"][-1]["input_ids"] == ["1", "2", "3", "4"]
    assert parsed["attrs"]["sink"] == "#5#"


def test_product_compiler_still_keeps_only_one_search():
    raw_plan = "<plan><search>q=first item</search><search>q=second item</search></plan>"

    plan, metadata = compile_shoppingbench_plan("Find a product matching this request.", raw_plan)
    parsed = parse_contract_plan(plan)

    assert metadata["queries"] == ["first item"]
    assert len(parsed["nodes"]) == 1


def test_atomic_query_extraction_supports_q_lines_and_encoded_query_lists():
    raw_plan = """<plan>
    <search id="1">
    q=first item shop_id=1
    q=second%20item
    </search>
    <search id="2">q=third%20item&amp;q=fourth%20item</search>
    </plan>"""
    assert extract_atomic_search_queries(raw_plan) == [
        "first item",
        "second item",
        "third item",
        "fourth item",
    ]


def test_atomic_query_extraction_drops_nonsemantic_query_parameters():
    raw_plan = """<plan><search id="1">
    q=oil%20crayons%2050%20shades&amp;page=1
    </search></plan>"""
    assert extract_atomic_search_queries(raw_plan) == ["oil crayons 50 shades"]


def test_visible_product_query_extraction_requires_explicit_numbering():
    question = (
        "Find one seller. Product details: (1) red belt; it should have size large; "
        "(2) blue shoes; it should have size 42."
    )
    assert extract_visible_product_queries(question) == [
        "red belt; it should have size large",
        "blue shoes; it should have size 42",
    ]
    assert extract_visible_product_queries("Find one seller for red belt and blue shoes.") == []


def test_shop_compiler_falls_back_to_visible_numbered_products_only_when_needed():
    question = (
        "Find one shop for all these items. Product details: "
        "(1) red belt; it should have size large; (2) blue shoes; it should have size 42."
    )
    raw_plan = "<plan><search id=\"1\">red belt and blue shoes</search></plan>"

    plan, metadata = compile_shoppingbench_plan(question, raw_plan)
    parsed = parse_contract_plan(plan)

    assert metadata["queries"] == [
        "red belt; it should have size large",
        "blue shoes; it should have size 42",
    ]
    assert [node["type"] for node in parsed["nodes"]] == ["search", "search", "reason"]
    assert parsed["nodes"][-1]["op"] == "join_shop"


def test_shop_compiler_preserves_sufficient_model_atomic_queries():
    question = (
        "Find one shop for all these items. Product details: "
        "(1) red belt; (2) blue shoes."
    )
    raw_plan = "<plan><search>q=model belt</search><search>q=model shoes</search></plan>"

    _, metadata = compile_shoppingbench_plan(question, raw_plan)

    assert metadata["queries"] == ["model belt", "model shoes"]


def test_unlimited_planner_prompt_has_no_node_cap():
    prompt = build_planner_system_message(0)
    assert "at most" not in prompt.lower()
    assert "HIGH-LOW" not in prompt
    assert "at most 8 nodes" in build_planner_system_message(8)


def test_generation_prompts_do_not_contain_copyable_fake_content():
    from qa_manager.ROSA import shoppingbench_protocol as protocol

    prompts = protocol.SEARCH_SYSTEM_MSG + protocol.REASON_SYSTEM_MSG + protocol.FINAL_SYSTEM_MSG
    assert "CATALOG_EVIDENCE" not in prompts
    assert "12345" not in prompts
    assert "Do not create product tags" in protocol.SEARCH_SYSTEM_MSG
    assert "exactly one plain-text search node" in protocol.PLANNER_BASE_SYSTEM_MSG


def test_product_id_answer_validation_and_extraction():
    answer = "<product_ids>591486855,4410867766</product_ids>"
    assert validate_final_answer(answer) is None
    assert extract_product_ids(answer) == ["591486855", "4410867766"]


def test_empty_product_selection_is_valid():
    answer = "<product_ids></product_ids>"
    assert validate_final_answer(answer) is None
    assert extract_product_ids(answer) == []


def test_invalid_product_id_answer_is_rejected():
    assert validate_final_answer("<product_ids>abc,123</product_ids>") is not None
    assert validate_final_answer("<product_ids>123</product_ids><note>x</note>") is not None


def test_contract_prompt_keeps_cp_shape_without_declared_out():
    prompt = build_contract_planner_system_message(7)
    assert 'answer_type="product_ids"' in prompt
    assert "Use at most 7 nodes" in prompt
    assert "product_ids" in prompt
    assert "out field" in prompt
    assert ' out="' not in prompt


def test_contract_root_query_validates_json_fields():
    query, binding = compile_contract_search_query(
        {
            "task": '{"service":["COD"],"q":"rubber plugs","price":"75-95"}',
            "input_ids": [],
        },
        {},
        {},
    )
    assert query == '{"price":"75-95","q":"rubber plugs","service":["COD"]}'
    assert binding == {}

    query, _ = compile_contract_search_query(
        {"task": '{"q":"rubber plugs","invented_filter":true}', "input_ids": []},
        {},
        {},
    )
    assert query == ""


def test_contract_dependent_query_binds_runtime_product_ids():
    query, binding = compile_contract_search_query(
        {"task": "inspect selected products", "input_ids": ["2"]},
        {
            "2": (
                "Product ID: 591486855; supported. "
                "Product ID: 4410867766; supported. Product ID: 591486855"
            )
        },
        {},
    )
    assert query == '{"product_ids":"591486855,4410867766"}'
    assert binding == {}


def test_rosa_server_routes_cp_queries_through_active_profile():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "exp" / "rosa_server.py"
    ).read_text(encoding="utf-8")
    assert "active_planner_system_message" in source
    assert "compile_active_contract_search_query" in source


CATALOG_CONTEXT = """[1_1]: (Title: Product ID: 11) Product ID: 11
Shop ID: 7
Title: Product A
Price: 100.0
Services: official, COD
SKU Options: {"1": {"color": "red"}}
Attributes: {"brand": ["acme"]}
[1_2]: (Title: Product ID: 22) Product ID: 22
Shop ID: 8
Title: Product B
Price: 80.0
Services: freeShipping
"""


def test_product_set_is_parsed_from_retriever_context_with_provenance():
    products = parse_product_set(CATALOG_CONTEXT, query="products", source_node="1")
    assert [item["product_id"] for item in products] == ["11", "22"]
    assert products[0]["shop_id"] == "7"
    assert products[0]["price"] == 100.0
    assert products[0]["services"] == ["official", "COD"]
    assert products[0]["retrieval_rank"] == 1
    assert products[0]["source_node"] == "1"


def test_model_selection_cannot_introduce_an_unretrieved_product_id():
    products = parse_product_set(CATALOG_CONTEXT)
    selected = select_supported_products(
        products,
        '<product id="999">invented</product> Product ID: 11',
    )
    assert [item["product_id"] for item in selected] == ["11"]
    assert "999" not in product_set_conclusion(selected)
    assert serialize_product_sink(selected) == "<product_ids>11</product_ids>"


def test_numeric_only_model_selection_is_still_provenance_checked():
    products = parse_product_set(CATALOG_CONTEXT)
    selected = select_supported_products(products, "11\n999\n22")
    assert [item["product_id"] for item in selected] == ["11", "22"]


def test_model_annotation_does_not_delete_retrieved_products():
    products = parse_product_set(CATALOG_CONTEXT)
    annotated = annotate_product_selection(products, "Insufficient information")
    assert [item["product_id"] for item in annotated] == ["11", "22"]
    assert not any(item["model_selected"] for item in annotated)


def test_search_sink_prefers_one_selected_or_top_retrieved_product():
    products = annotate_product_selection(
        parse_product_set(CATALOG_CONTEXT),
        "Product ID: 22\nProduct ID: 11",
    )
    assert serialize_product_sink(
        products, sink_node={"type": "search"}
    ) == "<product_ids>11</product_ids>"
    unselected = annotate_product_selection(parse_product_set(CATALOG_CONTEXT), "")
    assert serialize_product_sink(
        unselected, sink_node={"type": "search"}
    ) == "<product_ids>11</product_ids>"


def test_deterministic_shop_join_preserves_source_order():
    first = [
        {"product_id": "11", "shop_id": "7", "retrieval_rank": 2},
        {"product_id": "12", "shop_id": "8", "retrieval_rank": 1},
    ]
    second = [
        {"product_id": "21", "shop_id": "7", "retrieval_rank": 1},
        {"product_id": "22", "shop_id": "8", "retrieval_rank": 4},
    ]
    result, method = execute_artifact_reason(
        {"op": "join_shop"}, [first, second], "find products from one shop"
    )
    assert method == "deterministic_shop_join"
    assert [item["product_id"] for item in result] == ["11", "21"]


def test_deterministic_fixed_voucher_rejects_over_budget_combinations():
    first = [{"product_id": "11", "shop_id": "7", "price": 100.0, "retrieval_rank": 1}]
    second = [
        {"product_id": "21", "shop_id": "7", "price": 60.0, "retrieval_rank": 1},
        {"product_id": "22", "shop_id": "8", "price": 120.0, "retrieval_rank": 2},
    ]
    question = """My budget is only `140`, but I have a voucher.
It is valid only when the total price of the products exceeds `100`.
It provides a fixed discount of `25`."""
    result, method = execute_artifact_reason(
        {"op": "apply_voucher"}, [first, second], question
    )
    assert method == "deterministic_voucher"
    assert [item["product_id"] for item in result] == ["11", "21"]
