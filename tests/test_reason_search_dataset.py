import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import pandas as pd
from qa_manager.ROSA import protocol


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_dagent_reason_search_dataset.py"
SPEC = importlib.util.spec_from_file_location("reason_search_generator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeLLM:
    def chat(self, system, user, max_tokens=0, temperature=None):
        if system == MODULE.BLUEPRINT_SYSTEM_MSG:
            return json.dumps({
                "candidate_a": "Ada Alpha",
                "candidate_b": "Bea Beta",
                "selected_candidate": "Ada Alpha",
                "operation": "earlier",
                "search_query_a": "When was Ada Alpha born?",
                "search_query_b": "When was Bea Beta born?",
                "comparison_instruction": "Select the person born earlier.",
                "downstream": [{
                    "relation": "birthplace",
                    "query_template": "Where was {entity} born?",
                    "question_template": "Where was the person born earlier from {entity}?",
                }],
            })
        if system == MODULE.EXTRACT_COMPARISON_SYSTEM_MSG:
            return json.dumps({
                "candidate_a": {
                    "supported": True,
                    "answer": "1 January 1900",
                    "comparison_key": 19000101,
                    "doc_index": 1,
                    "evidence_quote": "Ada Alpha was born on 1 January 1900.",
                },
                "candidate_b": {
                    "supported": True,
                    "answer": "2 February 1910",
                    "comparison_key": 19100202,
                    "doc_index": 1,
                    "evidence_quote": "Bea Beta was born on 2 February 1910.",
                },
            })
        if system == MODULE.EXTRACT_FACT_SYSTEM_MSG:
            return json.dumps({
                "winner": {
                    "supported": True,
                    "answer": "London",
                    "doc_index": 1,
                    "evidence_quote": "Ada Alpha was born in London.",
                },
                "counterfactual": {
                    "supported": True,
                    "answer": "Paris",
                    "doc_index": 1,
                    "evidence_quote": "Bea Beta was born in Paris.",
                },
            })
        if system == MODULE.QUESTION_SYSTEM_MSG:
            return json.dumps({
                "question": "Where was the earlier-born person, Ada Alpha or Bea Beta, born?"
            })
        raise AssertionError("unexpected prompt")


class FakeRetriever:
    CONTENT = {
        "When was Ada Alpha born?": "Ada Alpha was born on 1 January 1900.",
        "When was Bea Beta born?": "Bea Beta was born on 2 February 1910.",
        "Where was Ada Alpha born?": "Ada Alpha was born in London.",
        "Where was Bea Beta born?": "Bea Beta was born in Paris.",
    }

    def search(self, query):
        contents = self.CONTENT.get(query, "This comparison requires multiple retrieval steps.")
        return [MODULE.Document(query, contents, 1.0, query)]


class MultiVariantLLM(FakeLLM):
    def chat(self, system, user, max_tokens=0, temperature=None):
        if system == MODULE.BLUEPRINT_SYSTEM_MSG:
            value = json.loads(super().chat(system, user, max_tokens, temperature))
            value["downstream"].append({
                "relation": "award",
                "query_template": "What award did {entity} win?",
                "question_template": "What award did the earlier-born person from {entity} win?",
            })
            return json.dumps(value)
        if system == MODULE.EXTRACT_FACT_SYSTEM_MSG and "award" in user.lower():
            return json.dumps({
                "winner": {
                    "supported": True,
                    "answer": "Prize A",
                    "doc_index": 1,
                    "evidence_quote": "Ada Alpha won Prize A.",
                },
                "counterfactual": {
                    "supported": True,
                    "answer": "Prize B",
                    "doc_index": 1,
                    "evidence_quote": "Bea Beta won Prize B.",
                },
            })
        if system == MODULE.QUESTION_SYSTEM_MSG and "award" in user.lower():
            return json.dumps({
                "question": "What award did the earlier-born person, Ada Alpha or Bea Beta, win?"
            })
        return super().chat(system, user, max_tokens, temperature)


class MultiVariantRetriever(FakeRetriever):
    CONTENT = {
        **FakeRetriever.CONTENT,
        "What award did Ada Alpha win?": "Ada Alpha won Prize A.",
        "What award did Bea Beta win?": "Bea Beta won Prize B.",
    }


def seed():
    return {
        "source_index": 1,
        "source_dataset": "test",
        "source_id": "seed-1",
        "source_question_type": "comparison",
        "source_question": "Who was born earlier, Ada Alpha or Bea Beta?",
        "source_answer": "Ada Alpha",
    }


def test_build_one_requires_reason_for_followup_search():
    record = MODULE.build_one(seed(), FakeLLM(), FakeRetriever())
    assert [node["type"] for node in record["nodes_execution"]] == [
        "search", "search", "reason", "search"
    ]
    assert "#3#" in record["nodes_execution"][3]["task"]
    assert record["answer"] == "London"
    assert record["counterfactual_answer"] == "Paris"
    assert record["comparison_evidence"]["candidate_a"]["comparison_key"] == 19000101
    assert record["verification"]["direct_question_does_not_retrieve_answer"] is True


def test_one_seed_can_produce_distinct_grounded_variants():
    records = MODULE.build_seed_variants(
        seed(), MultiVariantLLM(), MultiVariantRetriever(), max_variants=2, blueprint_attempts=1
    )
    assert len(records) == 2
    assert len({record["question"] for record in records}) == 2
    assert {record["downstream"]["relation"] for record in records} == {"birthplace", "award"}
    assert all("#3#" in record["nodes_execution"][3]["task"] for record in records)


def test_sft_and_rl_outputs():
    record = MODULE.build_one(seed(), FakeLLM(), FakeRetriever())
    temp_root = SCRIPT.parents[1] / ".test-tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        output_dir = Path(directory)
        stats = MODULE.rebuild_outputs([record], output_dir, val_ratio=0.1, seed=42)
        sft = pd.read_parquet(output_dir / "sft_all.parquet")
        rl = pd.read_parquet(output_dir / "rl_train.parquet")
        assert stats["sft_rows"] == 7
        assert set(sft["agent_name"]) == {
            "PlanningAgent", "AnswerGenerationAgent", "ReasonerAgent",
            "QueryRewriteAgent", "AnswerSummarizationAgent",
        }
        assert all(
            sum(message.get("loss_mask") or 0 for message in messages) == 1
            for messages in sft["messages"]
        )
        assert all(
            isinstance(message.get("loss_mask"), int)
            for messages in sft["messages"]
            for message in messages
        )
        assert rl.iloc[0]["reward_model"]["ground_truth"]["target"] == ["London"]


def test_all_generated_agent_messages_use_canonical_protocol_builders():
    record = MODULE.build_one(seed(), FakeLLM(), FakeRetriever())
    events = record["sft_events"]
    expected_system = {
        "PlanningAgent": protocol.PLANNER_SYSTEM_MSG,
        "AnswerGenerationAgent": protocol.ANSWER_GENERATION_SYSTEM_MSG,
        "ReasonerAgent": protocol.REASON_SYSTEM_MSG,
        "QueryRewriteAgent": protocol.QUERY_REWRITE_SYSTEM_MSG,
        "AnswerSummarizationAgent": protocol.FINAL_SYSTEM_MSG,
    }
    assert all(event["messages"][0]["content"] == expected_system[event["agent_name"]] for event in events)
    assert events == MODULE.build_events(record)
    final = next(event for event in events if event["agent_name"] == "AnswerSummarizationAgent")
    assert "Task 1:" in final["messages"][1]["content"]
    assert "[SEARCH]" not in final["messages"][1]["content"]


def test_rebuild_replaces_stale_stored_events_with_canonical_events():
    record = MODULE.build_one(seed(), FakeLLM(), FakeRetriever())
    record["sft_events"][0]["messages"][0]["content"] = "stale planner prompt"
    temp_root = SCRIPT.parents[1] / ".test-tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        output_dir = Path(directory)
        MODULE.rebuild_outputs([record], output_dir, val_ratio=0.1, seed=42)
        sft = pd.read_parquet(output_dir / "sft_all.parquet")
        planner = sft[sft["agent_name"] == "PlanningAgent"].iloc[0]
        assert planner["messages"][0]["content"] == protocol.PLANNER_SYSTEM_MSG


def test_ungrounded_quote_is_rejected():
    documents = [MODULE.Document("Ada", "Ada was born in London.", 1.0, "1")]
    value = {
        "supported": True,
        "answer": "Paris",
        "doc_index": 1,
        "evidence_quote": "Ada was born in Paris.",
    }
    try:
        MODULE.grounded_extraction(value, documents, "test", numeric=False)
    except MODULE.Rejection:
        return
    raise AssertionError("an ungrounded quote passed validation")


def test_compound_winner_and_downstream_question_is_rejected():
    class CompoundQuestionLLM(FakeLLM):
        def chat(self, system, user, max_tokens=0, temperature=None):
            if system == MODULE.QUESTION_SYSTEM_MSG:
                return json.dumps({
                    "question": "Which person was born earlier, Ada Alpha or Bea Beta, and where were they born?"
                })
            return super().chat(system, user, max_tokens, temperature)

    try:
        MODULE.build_one(seed(), CompoundQuestionLLM(), FakeRetriever())
    except MODULE.Rejection as exc:
        assert "requests both" in str(exc)
        return
    raise AssertionError("a compound winner-and-downstream question passed validation")


def test_fatal_api_error_is_not_recorded_as_a_seed_rejection():
    class PaymentRequiredLLM(FakeLLM):
        def chat(self, system, user, max_tokens=0, temperature=None):
            raise MODULE.FatalApiError("DeepSeek API returned HTTP 402")

    try:
        MODULE.build_seed_variants(
            seed(), PaymentRequiredLLM(), FakeRetriever(), max_variants=3, blueprint_attempts=4
        )
    except MODULE.FatalApiError as exc:
        assert "402" in str(exc)
        return
    raise AssertionError("a fatal API error was swallowed as a normal rejection")


def test_entity_disjoint_three_way_split_and_exclusion():
    base = MODULE.build_one(seed(), FakeLLM(), FakeRetriever())
    records = []
    for index in range(12):
        record = deepcopy(base)
        record["trajectory_id"] = f"trajectory-{index}"
        record["winner"] = f"Winner {index}"
        record["counterfactual_entity"] = f"Loser {index}"
        record["question"] = f"Synthetic question {index}?"
        record["supporting_titles"] = [f"Title {index} A", f"Title {index} B"]
        record["source"] = {**record["source"], "source_id": f"seed-{index}"}
        records.append(record)
    train_ids, val_ids, test_ids = MODULE.entity_disjoint_three_way_split(
        records, val_ratio=0.25, test_ratio=0.25, seed=42
    )
    assert train_ids | val_ids | test_ids == {record["trajectory_id"] for record in records}
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)

    excluded = MODULE.exclusion_index([records[0]])
    assert MODULE.exclusion_reason(records[0], excluded) is not None
    assert MODULE.exclusion_reason(records[1], excluded) is None
