from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.evaluate_external_baseline_full_answerable import (
    DATASETS,
    DEFAULT_PROTOCOL_ROOT,
    close_truncated_tag,
    first_event,
    format_initial_prompt,
    load_protocol,
    run_one,
)


def test_first_event_prefers_the_earliest_closed_tag() -> None:
    text = "<think>plan</think><search>who?</search>"
    assert first_event(text) == ("search", "who?")

    answer = "<think>done</think><answer>Beijing</answer>"
    assert first_event(answer) == ("answer", "Beijing")

    tool_call = '<tool_call>{"name":"search","arguments":{"query":"who?"}}</tool_call>'
    tool_config = {"prompt_family": "search_r1_tool"}
    assert first_event(tool_call, tool_config) == ("search", "who?")
    boxed = r"\boxed{Beijing}"
    assert first_event(boxed, tool_config) == ("boxed_answer", "Beijing")


def test_close_truncated_tag_makes_vllm_stop_compatible() -> None:
    assert close_truncated_tag("<search>who") == "<search>who</search>"
    assert close_truncated_tag("<answer>Beijing") == "<answer>Beijing</answer>"


@pytest.mark.parametrize("baseline", ["search_r1", "zerosearch", "hiprag", "reopd", "opd", "rag", "ircot"])
def test_protocols_are_pinned_and_have_required_fields(baseline: str) -> None:
    path = DEFAULT_PROTOCOL_ROOT / f"{baseline}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["baseline"] == baseline
    for key in ("prompt_family", "retriever_path", "retrieval_format", "retrieval_topk", "stop", "continuation"):
        assert key in config


def test_prompt_family_uses_the_expected_protocol_prefix() -> None:
    class Tokenizer:
        chat_template = None

    for baseline in ("search_r1", "zerosearch"):
        config = json.loads((DEFAULT_PROTOCOL_ROOT / f"{baseline}.json").read_text(encoding="utf-8"))
        prompt = format_initial_prompt("Who?", config, Tokenizer())
        assert "<search>" in prompt and "<answer>" in prompt

    rag = json.loads((DEFAULT_PROTOCOL_ROOT / "rag.json").read_text(encoding="utf-8"))
    prompt = format_initial_prompt("Who?", rag, Tokenizer(), "Doc 1: evidence")
    assert "Retrieved passages" in prompt and "Doc 1: evidence" in prompt

    ircot = json.loads((DEFAULT_PROTOCOL_ROOT / "ircot.json").read_text(encoding="utf-8"))
    prompt = format_initial_prompt("Who?", ircot, Tokenizer())
    assert "one sentence at a time" in prompt

    config = json.loads((DEFAULT_PROTOCOL_ROOT / "hiprag.json").read_text(encoding="utf-8"))
    prompt = format_initial_prompt("Who?", config, Tokenizer())
    assert "<think>" in prompt and "<reasoning>" in prompt

    config = json.loads((DEFAULT_PROTOCOL_ROOT / "reopd.json").read_text(encoding="utf-8"))
    prompt = format_initial_prompt("Who?", config, Tokenizer())
    assert "<tool_call>" in prompt and "<tool_response>" in prompt and "\\boxed" in prompt


def test_protocol_configs_keep_vllm_stop_output_explicit() -> None:
    for baseline in ("search_r1", "zerosearch", "hiprag", "reopd", "opd", "rag", "ircot"):
        config = json.loads((DEFAULT_PROTOCOL_ROOT / f"{baseline}.json").read_text(encoding="utf-8"))
        assert config["include_stop_str_in_output"] is True
        assert "</answer>" in config["stop"]
    for baseline in ("search_r1", "zerosearch", "hiprag", "reopd", "opd"):
        config = json.loads((DEFAULT_PROTOCOL_ROOT / f"{baseline}.json").read_text(encoding="utf-8"))
        assert "</search>" in config["stop"]


def test_rag_retrieves_once_then_returns_the_tagged_answer() -> None:
    class Completion:
        async def generate(self, prompt, config):
            assert "Doc 1: evidence" in prompt
            return "<answer>Paris</answer>"

    class Retriever:
        def __init__(self):
            self.queries = []

        async def retrieve(self, query):
            self.queries.append(query)
            return "Doc 1: evidence"

    config = json.loads((DEFAULT_PROTOCOL_ROOT / "rag.json").read_text(encoding="utf-8"))
    retriever = Retriever()
    result = asyncio.run(run_one({"question": "Capital of France?"}, config, None, Completion(), retriever))
    assert result["answer"] == "Paris"
    assert result["search_calls"] == 1
    assert retriever.queries == ["Capital of France?"]


def test_ircot_retrieves_from_the_generated_reasoning_sentence() -> None:
    class Completion:
        def __init__(self):
            self.outputs = iter(["<think>France's capital is requested.</think>", "<answer>Paris</answer>"])

        async def generate(self, prompt, config):
            return next(self.outputs)

    class Retriever:
        def __init__(self):
            self.queries = []

        async def retrieve(self, query):
            self.queries.append(query)
            return "Doc 1: Paris is the capital of France."

    config = json.loads((DEFAULT_PROTOCOL_ROOT / "ircot.json").read_text(encoding="utf-8"))
    retriever = Retriever()
    result = asyncio.run(run_one({"question": "Capital of France?"}, config, None, Completion(), retriever))
    assert result["answer"] == "Paris"
    assert result["search_calls"] == 1
    assert retriever.queries == ["France's capital is requested."]


def test_ircot_uses_an_explicit_final_answer_turn_after_retrieval_budget() -> None:
    class Completion:
        def __init__(self):
            self.outputs = iter(["<think>Find the capital.</think>", "<answer>Paris</answer>"])
            self.prompts = []

        async def generate(self, prompt, config):
            self.prompts.append(prompt)
            return next(self.outputs)

    class Retriever:
        async def retrieve(self, query):
            return "Doc 1: Paris is the capital of France."

    config = json.loads((DEFAULT_PROTOCOL_ROOT / "ircot.json").read_text(encoding="utf-8"))
    config["max_retrieval_calls"] = 1
    completion = Completion()
    result = asyncio.run(run_one({"question": "Capital of France?"}, config, None, completion, Retriever()))
    assert result["answer"] == "Paris"
    assert "retrieval budget is exhausted" in completion.prompts[1]
