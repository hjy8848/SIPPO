from scripts.analyze_reason_search_handoff import audit_workflow, summarize


METADATA = {
    "sample_id": "sample-1",
    "question": "Which candidate is closer?",
    "operation": "closer_to_anchor",
    "winner": "Candidate B",
    "candidate_a": "Candidate A",
    "candidate_b": "Candidate B",
}


def test_unique_reason_value_recovers_winner_and_flags_bad_rewrite():
    nodes = [
        {"id": "1", "type": "search", "query": "Birth year of Candidate A", "conclusion": "1990"},
        {"id": "2", "type": "search", "query": "Birth year of Candidate B", "conclusion": "1979"},
        {
            "id": "3",
            "type": "reason",
            "query": "Which is closer, #1# or #2#?",
            "conclusion": "1979",
        },
        {
            "id": "4",
            "type": "search",
            "query": "Occupation of #3#",
            "resolved_query": "Occupation of 1979",
            "executed_query": "Occupation of Candidate A",
        },
    ]

    rows = audit_workflow(nodes, METADATA)

    assert len(rows) == 1
    assert rows[0]["unique_value_match"] is True
    assert rows[0]["recovered_candidate"] == "Candidate B"
    assert rows[0]["binding_correct"] is True
    assert rows[0]["current_query_mentions_winner"] is False
    assert rows[0]["current_query_binding_correct"] is False
    assert rows[0]["correction_opportunity"] is True


def test_duplicate_parent_values_abstain_instead_of_guessing():
    nodes = [
        {"id": "1", "type": "search", "query": "Birth year of Candidate A", "conclusion": "1979"},
        {"id": "2", "type": "search", "query": "Birth year of Candidate B", "conclusion": "1979"},
        {"id": "3", "type": "reason", "query": "Compare #1# and #2#", "conclusion": "1979"},
        {"id": "4", "type": "search", "query": "Occupation of #3#", "executed_query": "1979"},
    ]

    row = audit_workflow(nodes, METADATA)[0]

    assert row["value_match_count"] == 2
    assert row["unique_candidate_binding"] is False
    assert row["binding_wrong"] is False


def test_summary_requires_safe_coverage_precision_and_correction_mass():
    rows = []
    for index in range(30):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "operation": "closer_to_anchor",
                "unique_value_match": True,
                "unique_candidate_binding": True,
                "binding_correct": True,
                "binding_wrong": False,
                "current_query_mentions_winner": index >= 15,
                "current_query_unique_candidate": True,
                "current_query_binding_correct": index >= 15,
                "current_query_ambiguous": False,
                "correction_opportunity": index < 15,
            }
        )

    report = summarize(rows, workflow_count=1000)

    assert report["passed"] is True
    assert report["binding_coverage"] == 1.0
    assert report["binding_precision"] == 1.0
    assert report["correction_opportunities"] == 15
