from qa_manager.ROSA.contract_protocol import (
    compile_search_query,
    contract_reason_sources,
    execute_deterministic_reason,
    make_contract_reason_binding,
    parse_contract_plan,
    validate_contract_plan,
)


CHAIN = """<plan sink="#2#" answer_type="entity">
<search id="1" subject="Ada Lovelace" relation="husband">Who was Ada Lovelace's husband?</search>
<search id="2" subject="#1#" relation="father">Who was the father of #1#?</search>
</plan>"""


COMPARISON = """<plan sink="#3#" answer_type="entity">
<search id="1" subject="Film A" relation="director_birth_year" candidate="Film A">When was the director of Film A born?</search>
<search id="2" subject="Film B" relation="director_birth_year" candidate="Film B">When was the director of Film B born?</search>
<reason id="3" op="later" inputs="#1#,#2#" return="candidate">Select the film whose director was born later using #1# and #2#.</reason>
</plan>"""


def test_contract_chain_has_one_typed_sink_and_compiles_dependency():
    plan = parse_contract_plan(CHAIN)
    assert validate_contract_plan(
        "Who is the father of Ada Lovelace's husband?", plan, max_nodes=5
    ) == []
    query, binding = compile_search_query(plan["nodes"][1], {"1": "William King-Noel"}, {})
    assert query == "William King-Noel father"
    assert binding == {}


def test_contract_discards_legacy_model_declared_out():
    legacy = CHAIN.replace('relation="husband"', 'relation="husband" out="banana"')
    plan = parse_contract_plan(legacy)
    assert "out" not in plan["nodes"][0]
    assert validate_contract_plan(
        "Who is the father of Ada Lovelace's husband?", plan, max_nodes=5
    ) == []


def test_contract_comparison_preserves_candidate_and_executes_scalar_operator():
    plan = parse_contract_plan(COMPARISON)
    assert validate_contract_plan(
        "Which film has the director born later, Film A or Film B?", plan, max_nodes=5
    ) == []
    sources = contract_reason_sources(
        plan["nodes"][2], plan["nodes"], {"1": "1960", "2": "1974"}
    )
    answer, binding, method = execute_deterministic_reason(plan["nodes"][2], sources)
    assert answer == "Film B"
    assert binding["selected_ref"] == "2"
    assert method == "deterministic_selection"


def test_contract_rejects_wrong_comparison_relation_before_execution():
    plan = parse_contract_plan(COMPARISON.replace("director_birth_year", "director", 1))
    errors = validate_contract_plan(
        "Which film has the director born later, Film A or Film B?", plan
    )
    assert any("mixes source relations" in error for error in errors)


def test_contract_rejects_consistently_wrong_comparison_dimension():
    plan = parse_contract_plan(
        COMPARISON.replace("director_birth_year", "director_occupation")
    )
    errors = validate_contract_plan(
        "Which film has the director born later, Film A or Film B?", plan
    )
    assert any("outside the question's birth dimension" in error for error in errors)


def test_contract_rejects_dead_branch_and_unconsumed_reason():
    text = """<plan sink="#3#" answer_type="answer">
<search id="1" subject="A" relation="birth_year" candidate="A">A birth year?</search>
<search id="2" subject="B" relation="birth_year" candidate="B">B birth year?</search>
<reason id="3" op="later" inputs="#1#" return="candidate">Use #1#.</reason>
</plan>"""
    errors = validate_contract_plan("Who was born later, A or B?", parse_contract_plan(text))
    assert any("one terminal sink" in error for error in errors)


def test_contract_compiler_uses_selected_candidate_not_selected_scalar():
    plan = parse_contract_plan(
        COMPARISON.replace('sink="#3#"', 'sink="#4#"').replace(
            "</plan>",
            '<search id="4" subject="#3#" relation="occupation">Occupation of #3#?</search>\n</plan>',
        )
    )
    binding = {
        "3": {
            "selected_candidate": "Film B",
            "selected_value": "1974",
            "selected_subject": "Film B",
        }
    }
    query, selected = compile_search_query(plan["nodes"][3], {"3": "Film B"}, binding)
    assert query == "Film B occupation"
    assert selected["selected_value"] == "1974"


def test_contract_model_selection_binding_uses_executed_source_not_model_value():
    sources = [
        {"id": "1", "subject": "Film A", "candidate": "Film A", "value": "1960"},
        {"id": "2", "subject": "Film B", "candidate": "Film B", "value": "1974"},
    ]
    binding = make_contract_reason_binding("2", sources)
    assert binding is not None
    assert binding["selected_candidate"] == "Film B"
    assert binding["selected_value"] == "1974"
    assert make_contract_reason_binding("3", sources) is None
