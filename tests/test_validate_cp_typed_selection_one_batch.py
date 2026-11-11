from scripts.validate_cp_typed_selection_one_batch import (
    collect_action_prompts,
    parse_selection,
    validate_generated_action,
)


def _row():
    return {
        "question": "Which film was earlier?",
        "gold_answer": ["Film B"],
        "raw_trace": {
            "nodes_execution": [
                {
                    "id": "3",
                    "reason_execution_method": "model_typed_selection",
                    "typed_handoff": {
                        "selected_ref": "1",
                        "selected_task": "year A",
                        "selected_value": "2000",
                        "selected_candidate": "Film A",
                        "rejected_sources": [
                            {
                                "id": "2",
                                "task": "year B",
                                "value": "1990",
                                "candidate": "Film B",
                            }
                        ],
                    },
                }
            ],
            "sft_events": [
                {
                    "agent_name": "ReasonerAgent",
                    "messages": [{"role": "user", "content": "choose"}],
                    "metadata": {"node_id": "3"},
                }
            ],
        },
    }


def test_parse_and_validate_generated_selection_against_declared_sources():
    action = collect_action_prompts([_row()])[0]
    result = validate_generated_action(
        action, '<selection ref="#2#"><value>1990</value></selection>'
    )
    assert result["valid"] is True
    assert result["factual_em"] == 1
    assert result["alternative_em"] == 0
    assert result["decisive"] is True


def test_value_must_match_selected_ref():
    action = collect_action_prompts([_row()])[0]
    result = validate_generated_action(
        action, '<selection ref="#2#"><value>2000</value></selection>'
    )
    assert result == {
        "valid": False,
        "error": "value_ref_mismatch",
        "response": '<selection ref="#2#"><value>2000</value></selection>',
    }


def test_parser_accepts_runtime_hash_variants():
    assert parse_selection('<selection ref="2"><value>x</value></selection>') == {
        "ref": "2",
        "value": "x",
    }
