import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_dagent_reason_search_evidence_first.py"
SPEC = importlib.util.spec_from_file_location("reason_search_evidence_first", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def document(title, contents, document_id):
    return {"title": title, "contents": contents, "score": 1.0, "document_id": document_id}


def record(index, candidate_a, candidate_b, year_a, year_b, answer_a, answer_b, relation="occupation"):
    winner = candidate_a if year_a < year_b else candidate_b
    loser = candidate_b if winner == candidate_a else candidate_a
    winner_answer = answer_a if winner == candidate_a else answer_b
    loser_answer = answer_b if winner == candidate_a else answer_a
    comparison_a = f"{candidate_a} was born in {year_a}."
    comparison_b = f"{candidate_b} was born in {year_b}."
    downstream_a = f"{candidate_a} was a {answer_a}."
    downstream_b = f"{candidate_b} was a {answer_b}."
    docs_a = [document(candidate_a, comparison_a, f"{index}-ca")]
    docs_b = [document(candidate_b, comparison_b, f"{index}-cb")]
    final_docs = [document(winner, downstream_a if winner == candidate_a else downstream_b, f"{index}-dw")]
    counterfactual_docs = [document(loser, downstream_b if loser == candidate_b else downstream_a, f"{index}-dl")]
    evidence_a = {"answer": str(year_a), "comparison_key": float(year_a), "evidence_quote": comparison_a,
                  "doc_index": 1, "supporting_title": candidate_a, "supporting_document_id": f"{index}-ca"}
    evidence_b = {"answer": str(year_b), "comparison_key": float(year_b), "evidence_quote": comparison_b,
                  "doc_index": 1, "supporting_title": candidate_b, "supporting_document_id": f"{index}-cb"}
    winner_quote = downstream_a if winner == candidate_a else downstream_b
    loser_quote = downstream_b if loser == candidate_b else downstream_a
    return {
        "trajectory_id": f"trajectory-{index}",
        "winner": winner,
        "counterfactual_entity": loser,
        "source": {"source_id": f"source-{index}"},
        "blueprint": {
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "operation": "earlier",
            "search_query_a": f"{candidate_a} date of birth",
            "search_query_b": f"{candidate_b} date of birth",
            "downstream": [{"relation": relation, "query_template": "{entity} occupation"}],
        },
        "comparison_evidence": {"candidate_a": evidence_a, "candidate_b": evidence_b},
        "nodes_execution": [
            {"id": "1", "documents": docs_a},
            {"id": "2", "documents": docs_b},
            {"id": "4", "task": "#3# occupation", "executed_query": f"{winner} occupation", "documents": final_docs},
        ],
        "downstream_evidence": {
            "winner": {"answer": winner_answer, "evidence_quote": winner_quote},
            "counterfactual": {"answer": loser_answer, "evidence_quote": loser_quote},
        },
        "counterfactual": {"documents": counterfactual_docs},
    }


class FakeRetriever:
    def __init__(self, contents, direct_contents="No direct answer is available."):
        self.contents = contents
        self.direct_contents = direct_contents

    def search(self, query):
        content = self.contents.get(query, self.direct_contents)
        return [MODULE.legacy.Document(query, content, 1.0, query)]


def test_extracts_both_sides_and_recovers_selected_downstream_from_old_schema():
    source = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    cards = MODULE.extract_fact_cards(source)
    assert len(cards) == 2
    assert {card.entity for card in cards} == {"Ada Alpha", "Bea Beta"}
    assert {card.downstream_answer for card in cards} == {"chemist", "painter"}
    assert all(card.downstream_relation == "occupation" for card in cards)
    assert all(card.comparison_family == "birth_date" for card in cards)


def test_join_uses_different_trajectories_and_excludes_original_pairs():
    first = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    second = record(2, "Cara Gamma", "Dora Delta", 1920, 1930, "writer", "singer")
    cards = [card for item in (first, second) for card in MODULE.extract_fact_cards(item)]
    pairs, stats = MODULE.join_fact_cards(
        cards, MODULE.original_entity_pairs((first, second)), seed=7,
        max_candidates=20, max_entity_uses=10,
    )
    assert stats["eligible_join_pairs"] == 4
    assert len(pairs) == 4
    assert all(pair.left.source_trajectory_id != pair.right.source_trajectory_id for pair in pairs)


def test_join_can_freeze_downstream_relation_scope_before_sampling():
    first = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    second = record(2, "Cara Gamma", "Dora Delta", 1920, 1930, "writer", "singer")
    cards = [card for item in (first, second) for card in MODULE.extract_fact_cards(item)]

    pairs, stats = MODULE.join_fact_cards(
        cards,
        MODULE.original_entity_pairs((first, second)),
        seed=7,
        max_candidates=20,
        max_entity_uses=10,
        allowed_downstream_relations={"place of birth"},
    )

    assert not pairs
    assert stats["join_rejections"]["downstream_relation_not_allowed"] == 4


def test_retrieval_verification_builds_full_four_node_record():
    first = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    second = record(2, "Cara Gamma", "Dora Delta", 1920, 1930, "writer", "singer")
    left = MODULE.extract_fact_cards(first)[0]
    right = MODULE.extract_fact_cards(second)[0]
    contents = {
        left.comparison_query: left.comparison_evidence_quote,
        right.comparison_query: right.comparison_evidence_quote,
        "Ada Alpha occupation": left.downstream_evidence_quote,
        "Cara Gamma occupation": right.downstream_evidence_quote,
    }
    cached = MODULE.CachedRetriever(FakeRetriever(contents))
    built = MODULE.build_verified_record(MODULE.CardPair(left, right), cached, 0)
    assert [node["type"] for node in built["nodes_execution"]] == ["search", "search", "reason", "search"]
    assert "#3#" in built["nodes_execution"][3]["task"]
    assert built["answer"] == "chemist"
    assert built["counterfactual_answer"] == "writer"
    assert len(built["sft_events"]) == 7


def test_retrieval_verification_rejects_missing_quote_and_direct_leak():
    first = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    second = record(2, "Cara Gamma", "Dora Delta", 1920, 1930, "writer", "singer")
    left = MODULE.extract_fact_cards(first)[0]
    right = MODULE.extract_fact_cards(second)[0]
    pair = MODULE.CardPair(left, right)
    missing = FakeRetriever({left.comparison_query: "Ada Alpha existed."})
    try:
        MODULE.build_verified_record(pair, MODULE.CachedRetriever(missing), 0)
    except MODULE.legacy.Rejection as exc:
        assert "absent from current top3" in str(exc)
    else:
        raise AssertionError("missing evidence quote passed retrieval verification")

    contents = {
        left.comparison_query: left.comparison_evidence_quote,
        right.comparison_query: right.comparison_evidence_quote,
        "Ada Alpha occupation": left.downstream_evidence_quote,
        "Cara Gamma occupation": right.downstream_evidence_quote,
    }
    leaking = FakeRetriever(contents, direct_contents="The answer is chemist.")
    try:
        MODULE.build_verified_record(pair, MODULE.CachedRetriever(leaking), 0)
    except MODULE.legacy.Rejection as exc:
        assert "direct full-question retrieval" in str(exc)
    else:
        raise AssertionError("direct answer leakage passed verification")


def test_direct_leak_is_rejected_before_atomic_evidence_replay():
    first = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    second = record(2, "Cara Gamma", "Dora Delta", 1920, 1930, "writer", "singer")
    pair = MODULE.CardPair(MODULE.extract_fact_cards(first)[0], MODULE.extract_fact_cards(second)[0])
    cached = MODULE.CachedRetriever(FakeRetriever({}, direct_contents="The answer is chemist."))
    try:
        MODULE.build_verified_record(pair, cached, 0)
    except MODULE.legacy.Rejection as exc:
        assert "direct full-question retrieval" in str(exc)
        assert cached.requests == 1
    else:
        raise AssertionError("direct answer leakage passed verification")


def test_anchor_distance_adds_deterministic_non_retrieval_reasoning():
    first = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    second = record(2, "Cara Gamma", "Dora Delta", 1920, 1930, "writer", "singer")
    left = MODULE.extract_fact_cards(first)[0]
    right = MODULE.extract_fact_cards(second)[0]
    pair = MODULE.CardPair(left, right)
    spec = MODULE.reasoning_spec(pair, "anchor_distance")
    contents = {
        left.comparison_query: left.comparison_evidence_quote,
        right.comparison_query: right.comparison_evidence_quote,
        "Ada Alpha occupation": left.downstream_evidence_quote,
        "Cara Gamma occupation": right.downstream_evidence_quote,
    }
    built = MODULE.build_verified_record(
        pair, MODULE.CachedRetriever(FakeRetriever(contents)), 0, "anchor_distance"
    )
    assert spec.anchor_year is not None
    assert f"closer to {spec.anchor_year}" in built["question"]
    assert "compute each absolute distance" in built["nodes_execution"][2]["task"]
    assert built["reasoning"] == {
        "mode": "anchor_distance",
        "operation": "closer_to_anchor",
        "anchor_year": spec.anchor_year,
        "requires_additional_retrieval": False,
    }
    assert built["blueprint"]["operation"] == "closer_to_anchor"
    assert built["nodes_execution"][2]["conclusion"] == built["winner"]
    assert "#3#" in built["nodes_execution"][3]["task"]


def test_mixed_reasoning_assignment_is_stable():
    first = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    second = record(2, "Cara Gamma", "Dora Delta", 1920, 1930, "writer", "singer")
    pair = MODULE.CardPair(MODULE.extract_fact_cards(first)[0], MODULE.extract_fact_cards(second)[0])
    assert MODULE.reasoning_spec(pair, "mixed", 1.0).mode == "anchor_distance"
    assert MODULE.reasoning_spec(pair, "mixed", 0.0).mode == "direct"


def test_anchor_distance_rejects_same_year_pairs_during_join():
    first = record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter")
    second = record(2, "Cara Gamma", "Dora Delta", 1900, 1930, "writer", "singer")
    cards = [MODULE.extract_fact_cards(first)[0], MODULE.extract_fact_cards(second)[0]]
    pairs, stats = MODULE.join_fact_cards(
        cards, set(), seed=7, max_candidates=10, max_entity_uses=2,
        reasoning_mode="anchor_distance",
    )
    assert pairs == []
    assert stats["join_rejections"]["anchor_distance_same_year"] == 1
    assert MODULE.reasoning_spec(MODULE.CardPair(*cards), "mixed", 1.0).mode == "direct"


def test_parent_split_filter_uses_existing_trajectory_assignment():
    records = [
        {"trajectory_id": "train-id"},
        {"trajectory_id": "val-id"},
    ]
    temp_root = SCRIPT.parents[1] / ".test-tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        root = Path(directory)
        input_path = root / "accepted_trajectories.jsonl"
        input_path.write_text("", encoding="utf-8")
        MODULE.legacy.pd.DataFrame({"trajectory_id": ["train-id"]}).to_parquet(
            root / "sft_train.parquet", index=False
        )
        filtered, split_path = MODULE.filter_parent_split(records, input_path, "train")
    assert filtered == [{"trajectory_id": "train-id"}]
    assert split_path.name == "sft_train.parquet"


def test_parent_split_can_be_reconstructed_only_when_explicitly_requested():
    records = [
        {
            "trajectory_id": f"trajectory-{index}",
            "winner": f"Winner {index}",
            "counterfactual_entity": f"Loser {index}",
            "supporting_titles": [f"Title {index}"],
        }
        for index in range(10)
    ]
    temp_root = SCRIPT.parents[1] / ".test-tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        input_path = Path(directory) / "accepted_trajectories.jsonl"
        input_path.write_text("", encoding="utf-8")
        train, split_path = MODULE.filter_parent_split(
            records, input_path, "train", reconstruct=True, split_seed=42, val_ratio=0.2
        )
        val, _ = MODULE.filter_parent_split(
            records, input_path, "val", reconstruct=True, split_seed=42, val_ratio=0.2
        )
    train_ids = {item["trajectory_id"] for item in train}
    val_ids = {item["trajectory_id"] for item in val}
    assert split_path is None
    assert train_ids | val_ids == {item["trajectory_id"] for item in records}
    assert not train_ids & val_ids
    assert len(val_ids) == 2


def test_resume_ledger_appends_new_unique_pairs_without_repeating_work():
    records = [
        record(1, "Ada Alpha", "Bea Beta", 1900, 1910, "chemist", "painter"),
        record(2, "Cara Gamma", "Dora Delta", 1920, 1930, "writer", "singer"),
        record(3, "Eva Epsilon", "Faye Phi", 1940, 1950, "doctor", "editor"),
    ]
    cards = [card for item in records for card in MODULE.extract_fact_cards(item)]
    contents = {}
    for card in cards:
        contents[card.comparison_query] = card.comparison_evidence_quote
        contents[f"{card.entity} occupation"] = card.downstream_evidence_quote

    temp_root = SCRIPT.parents[1] / ".test-tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        root = Path(directory)
        input_path = root / "accepted_trajectories.jsonl"
        output_dir = root / "output"
        MODULE.write_jsonl(input_path, records)
        args = MODULE.build_parser().parse_args([
            "--input", str(input_path),
            "--output-dir", str(output_dir),
            "--verify-retrieval",
            "--max-candidates", "20",
            "--max-entity-uses", "10",
            "--target-accepted", "1",
        ])
        original_client = MODULE.legacy.RetrieverClient
        MODULE.legacy.RetrieverClient = lambda unused_args: FakeRetriever(contents)
        try:
            first_report = MODULE.run(args)
            args.resume = True
            args.target_accepted = 2
            second_report = MODULE.run(args)
        finally:
            MODULE.legacy.RetrieverClient = original_client
        accepted = MODULE.load_jsonl(output_dir / "accepted_trajectories.jsonl")
        rejected = MODULE.load_jsonl(output_dir / "rejected_candidates.jsonl")
        derived = MODULE.write_derived_outputs(accepted, output_dir, "train")
        sft_frame = MODULE.legacy.pd.read_parquet(output_dir / "sft_train.parquet")
        rl_frame = MODULE.legacy.pd.read_parquet(output_dir / "rl_train.parquet")
        assert derived == {"sft_rows": 14, "rl_rows": 2}
        assert len(sft_frame) == 14
        assert len(rl_frame) == 2
        assert set(rl_frame["extra_info"].map(lambda value: value["split"])) == {"train"}
    assert first_report["accepted_trajectories"] == 1
    assert second_report["accepted_trajectories"] == 2
    assert len(accepted) == 2
    assert len({item["evidence_join"]["pair_id"] for item in accepted}) == 2
    assert len({MODULE.legacy.normalize(item["question"]) for item in accepted}) == 2
    assert not rejected


def test_derived_test_split_is_explicit_and_preserved(tmp_path):
    record = {
        "trajectory_id": "trajectory-test",
        "question": "Which candidate is closer to 1950?",
        "answer": "Candidate A",
        "gold_answers": ["Candidate A"],
        "supporting_titles": ["Candidate A"],
        "plan_output": "<plan></plan>",
        "counterfactual_entity": "Candidate B",
        "counterfactual_answer": "Candidate B answer",
        "sft_events": [],
        "source": {"source_index": 0, "source_dataset": "unit", "source_id": "unit-0"},
    }
    derived = MODULE.write_derived_outputs([record], tmp_path, "test")
    rl_frame = MODULE.legacy.pd.read_parquet(tmp_path / "rl_test.parquet")

    assert derived == {"sft_rows": 0, "rl_rows": 1}
    assert set(rl_frame["extra_info"].map(lambda value: value["split"])) == {"test"}


def test_all_parent_records_require_explicit_derived_split():
    parser = MODULE.build_parser()
    args = parser.parse_args(["--write-derived", "--parent-split", "all"])
    with pytest.raises(SystemExit, match="requires --derived-split"):
        MODULE.validate_args(args)

    valid_args = parser.parse_args([
        "--write-derived", "--parent-split", "all", "--derived-split", "test"
    ])
    MODULE.validate_args(valid_args)
