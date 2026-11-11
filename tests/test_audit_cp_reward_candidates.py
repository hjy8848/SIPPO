from scripts.audit_cp_reward_candidates import (
    audit_legacy_onpolicy,
    audit_process_valid_psc,
    audit_typed_selection,
    exact_match,
)


def _cp_row(selected, alternative, gold, em):
    return {
        "question": "Which candidate wins?",
        "gold_answer": [gold],
        "em": float(em),
        "raw_trace": {
            "answer_sink_id": "3",
            "final_answer": selected,
            "nodes_execution": [
                {"id": "1", "type": "search"},
                {"id": "2", "type": "search"},
                {
                    "id": "3",
                    "type": "reason",
                    "conclusion": selected,
                    "reason_execution_method": "model_typed_selection",
                    "contract_node": {"op": "earlier"},
                    "typed_handoff": {
                        "selected_candidate": selected,
                        "rejected_sources": [{"candidate": alternative}],
                    },
                },
            ],
        },
    }


def test_exact_match_uses_qa_normalization():
    assert exact_match("The Last Face", ["Last Face", "The Last Face"]) == 1
    assert exact_match("Other", ["Last Face"]) == 0


def test_typed_selection_requires_exact_terminal_identity_and_both_directions():
    rows = []
    for index in range(8):
        rows.append(_cp_row(f"Winner {index}", f"Loser {index}", f"Winner {index}", 1))
        rows.append(_cp_row(f"Loser B{index}", f"Winner B{index}", f"Winner B{index}", 0))
    result = audit_typed_selection(rows)
    assert result["identity_mismatches"] == 0
    assert result["decisive_cases"] == 16
    assert result["factual_wins"] == result["alternative_wins"] == 8
    assert result["passed"] is True


def test_invalid_typed_selection_is_gated_before_identity_control():
    rows = [_cp_row("", "", "Winner", 0)]
    rows[0]["raw_trace"]["nodes_execution"][-1]["typed_handoff"] = {}
    result = audit_typed_selection(rows)
    assert result["model_typed_selection_calls"] == 1
    assert result["ineligible_invalid_selection_calls"] == 1
    assert result["eligible_typed_selection_cases"] == 0
    assert result["identity_mismatches"] == 0


def test_process_proxy_does_not_promote_contract_or_topology_to_reward():
    rows = [
        {
            "em": "1",
            "semantic_shortcut": "True",
            "correct_reason_consumed": "False",
            "identity_binding_correct": "False",
            "downstream_conclusion_has_gold": "False",
            "canonical_plan": "True",
            "reason_selects_winner": "False",
        },
        {
            "em": "1",
            "semantic_shortcut": "False",
            "correct_reason_consumed": "True",
            "identity_binding_correct": "True",
            "downstream_conclusion_has_gold": "True",
            "canonical_plan": "False",
            "reason_selects_winner": "True",
        },
    ]
    result = audit_process_valid_psc(rows)
    assert result["em_positive_semantic_shortcuts"] == 1
    assert result["em_positive_grounded_reason_to_search"] == 1
    assert result["canonical_but_wrong_reason_em_positive"] == 1
    assert result["safety_gate_is_not_a_reward"] is True
    assert result["passed_as_trainable_reward"] is False


def test_legacy_replacement_fails_on_identity_control():
    result = audit_legacy_onpolicy(
        {
            "overall": {"informative_estimate_cases": 40},
            "replacement_statistics": {
                "replacement_samples": 100,
                "identity_response_samples": 70,
                "identity_response_terminal_em_changed_samples": 1,
            },
            "safety_invariants": {
                "identity_replacement_is_exact_negative_control": False
            },
        }
    )
    assert result["status"] == "invalid_estimator"
    assert result["passed"] is False
