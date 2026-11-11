from qa_manager.ROSA.protocol import (
    FINAL_SYSTEM_MSG,
    PLANNER_SYSTEM_MSG,
    REASON_SYSTEM_MSG,
    SEARCH_SYSTEM_MSG,
    build_answer_generation_messages,
    build_locked_query_rewrite_messages,
    build_planning_messages,
    build_provenance_preserving_query,
    build_query_rewrite_messages,
    build_reasoner_messages,
    build_summarization_messages,
    build_typed_reason_messages,
    find_reference_ids,
    is_degenerate_text,
    is_invalid_conclusion,
    is_supported_answer,
    make_typed_reason_binding,
    parse_typed_reason_selection,
    validate_locked_query_rewrite,
    validate_plan_semantics,
    validate_plan_structure,
    validate_typed_reason_selection,
)


def test_node_limit_is_validated_instead_of_truncated():
    nodes = [("search", str(index), "fact") for index in range(1, 7)]
    errors = validate_plan_structure(nodes, max_nodes=5)
    assert any("maximum is 5" in error for error in errors)
    assert validate_plan_structure(nodes, max_nodes=0) == []


def test_only_canonical_references_are_recognized():
    assert find_reference_ids("compare #1# and #2#") == ["1", "2"]
    assert find_reference_ids("compare #1 and #2") == []


def test_step310_copied_placeholders_are_rejected():
    nodes = [
        {"id": "1", "type": "search", "query": "one atomic retrieval question"},
        {"id": "2", "type": "reason", "task": "filtering over #1# results"},
    ]
    assert validate_plan_semantics("Who is Wisigard's father-in-law?", nodes)
    assert is_invalid_conclusion("if the answer is present in the context")
    assert is_invalid_conclusion("...")


def test_prompt_metavariables_are_not_valid_outputs():
    assert is_degenerate_text("{CONCRETE_QUESTION}")
    assert is_degenerate_text("{SUPPORTED_SHORT_ANSWER}")
    assert is_degenerate_text("{DERIVED_SHORT_ANSWER}")
    assert is_degenerate_text("{SUPPORTED_FINAL_ANSWER}")


def test_question_grounded_plan_is_accepted():
    nodes = [
        {"id": "1", "type": "search", "query": "Who was Ada Lovelace's husband?"},
        {"id": "2", "type": "search", "query": "Who was the father of #1#?"},
    ]
    assert validate_plan_semantics("Who is the father of Ada Lovelace's husband?", nodes) == []


def test_unrelated_root_search_is_rejected():
    nodes = [{"id": "1", "type": "search", "query": "What is atomic semantics?"}]
    errors = validate_plan_semantics("Where was Alan Turing born?", nodes)
    assert any("not grounded" in error for error in errors)


def test_final_answer_must_be_supported_by_a_node_conclusion():
    conclusions = ["Ada Lovelace", "London"]
    assert is_supported_answer("London", conclusions)
    assert not is_supported_answer("Paris", conclusions)
    assert not is_supported_answer("no", ["..."])


def test_prompts_do_not_offer_step310_failure_strings_as_targets():
    prompt_text = "\n".join([
        PLANNER_SYSTEM_MSG,
        SEARCH_SYSTEM_MSG,
        REASON_SYSTEM_MSG,
        FINAL_SYSTEM_MSG,
    ])
    assert "one atomic retrieval question" not in prompt_text
    assert "<conclusion>...</conclusion>" not in prompt_text
    assert "<answer>...</answer>" not in prompt_text


def test_planner_has_no_topology_label():
    assert "<type>" not in PLANNER_SYSTEM_MSG
    assert "chain" not in PLANNER_SYSTEM_MSG
    assert "parallel" not in PLANNER_SYSTEM_MSG
    assert "hybrid" not in PLANNER_SYSTEM_MSG


def test_canonical_message_builders_cover_all_runtime_roles():
    assert build_planning_messages("Q")[1]["content"] == "Question: Q\nGenerate the execution plan."
    assert build_query_rewrite_messages("Q", "#3# occupation", {"3": "Ada"})[1]["content"] == (
        "Original Goal: Q\nDraft: #3# occupation\n# 3 = Ada\nRewritten query:"
    )
    assert "**Search Context**:\nD1\nD2" in build_answer_generation_messages("SQ", ["D1", "D2"])[1]["content"]
    assert build_reasoner_messages("Compare A and B")[1]["content"] == (
        "**Current Logic Task**: Compare A and B"
    )
    summary = build_summarization_messages("Q", ["S1", "S2"], ["A1", "A2"])
    assert summary[1]["content"].endswith("Task 1: S1\nConclusion: A1\n\nTask 2: S2\nConclusion: A2")


def test_planner_prompt_can_disable_runtime_node_limit():
    limited_prompt = build_planning_messages("Q", max_nodes=5)[0]["content"]
    unlimited_prompt = build_planning_messages("Q", max_nodes=0)[0]["content"]

    assert "Use at most 5 nodes." in limited_prompt
    assert "Use at most" not in unlimited_prompt


def _selection_sources():
    return [
        {"id": "1", "task": "Zackary Medeiros birth year", "value": "1990"},
        {"id": "2", "task": "Rajid Baransi birth year", "value": "1979"},
    ]


def test_typed_reason_selection_preserves_source_identity_and_value():
    sources = _selection_sources()
    messages = build_typed_reason_messages(
        "Whose occupation is requested?",
        "Which of #1# and #2# is closer to 1975?",
        sources,
    )
    assert "#1# | task=Zackary Medeiros birth year | value=1990" in messages[1]["content"]

    selection = parse_typed_reason_selection(
        '<selection ref="#2#"><value>1979</value></selection>'
    )
    assert validate_typed_reason_selection(selection, sources) is None
    binding = make_typed_reason_binding(selection, sources)
    assert binding["selected_ref"] == "2"
    assert binding["selected_task"] == "Rajid Baransi birth year"


def test_typed_reason_selection_rejects_non_parent_and_mismatched_value():
    sources = _selection_sources()
    wrong_ref = parse_typed_reason_selection(
        '<selection ref="#3#"><value>1979</value></selection>'
    )
    wrong_value = parse_typed_reason_selection(
        '<selection ref="#2#"><value>1990</value></selection>'
    )
    assert "not a declared parent" in validate_typed_reason_selection(wrong_ref, sources)
    assert "does not match" in validate_typed_reason_selection(wrong_value, sources)


def test_locked_rewrite_rejects_sibling_switch_and_has_safe_fallback():
    selection = parse_typed_reason_selection(
        '<selection ref="#2#"><value>1979</value></selection>'
    )
    binding = make_typed_reason_binding(selection, _selection_sources())
    binding["reason_node_id"] = "3"

    messages = build_locked_query_rewrite_messages(
        "What was the occupation of the closer person?",
        "Occupation of #3#",
        binding,
    )
    assert "Selected Source Task: Rajid Baransi birth year" in messages[1]["content"]
    assert validate_locked_query_rewrite("Rajid Baransi occupation", binding) is None
    assert "rejected source" in validate_locked_query_rewrite(
        "Zackary Medeiros occupation", binding
    )

    fallback = build_provenance_preserving_query("Occupation of #3#", binding)
    assert "Rajid Baransi birth year" in fallback
    assert "Zackary" not in fallback
