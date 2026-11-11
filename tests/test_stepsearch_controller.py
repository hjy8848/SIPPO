from __future__ import annotations

import asyncio
from io import StringIO
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_stepsearch_full_answerable import (  # noqa: E402
    StepSearchRetriever,
    FINAL_SEARCH_WARNING,
    format_prompt,
    official_answer_metrics,
    protocol,
    run_config,
    run_one,
    stepsearch_first_event,
    stepsearch_official_first_event,
    truncate_text_tokens,
    write_compact_json,
)


def test_stepsearch_uses_official_prompt_and_chat_template() -> None:
    class Tokenizer:
        chat_template = "qwen"

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is False
            assert messages[0]["role"] == "user"
            assert messages[0]["content"].endswith(" Question: Who is this?\n")
            return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"

    prompt = format_prompt("Who is this", Tokenizer())
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "for EVERY searched document" in prompt
    assert protocol()["max_turns"] == 5
    assert protocol()["max_tokens"] == 800
    assert protocol()["max_prompt_tokens"] == 4096
    assert protocol()["max_observation_tokens"] == 800
    assert protocol()["temperature"] == 1.0
    assert protocol()["top_p"] == 1.0


def test_stepsearch_run_config_uses_requested_sampling_seed() -> None:
    config = run_config(seed=43, parser_mode="strict")
    assert config["seed"] == 43
    assert config["parser_mode"] == "strict"


def test_stepsearch_token_clipping_matches_official_directions() -> None:
    class CharTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(text)}

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    tokenizer = CharTokenizer()
    assert truncate_text_tokens("abcdefgh", tokenizer, 5, keep="head") == "abcde"
    assert truncate_text_tokens("abcdefgh", tokenizer, 5, keep="tail") == "defgh"


def test_stepsearch_loop_slides_prompt_and_truncates_observation() -> None:
    class CharTokenizer:
        chat_template = None

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(text)}

        def decode(self, token_ids, **kwargs):
            return "".join(token_ids)

    class Completion:
        def __init__(self):
            self.prompts = []
            self.configs = []
            self.outputs = ["<search>query</search>", "<answer>ok</answer>"]

        async def generate(self, prompt, config):
            self.prompts.append(prompt)
            self.configs.append(config)
            return self.outputs.pop(0)

    class Retriever:
        async def retrieve(self, query):
            assert query == "query"
            return "x" * 100

    tokenizer = CharTokenizer()
    completion = Completion()
    config = protocol()
    config.update({"max_turns": 1, "max_prompt_tokens": 40, "max_observation_tokens": 24})
    result = asyncio.run(run_one(
        {"question": "q"}, config, tokenizer, completion, Retriever()
    ))

    assert result["answer"] == "ok"
    assert all(len(prompt) <= 40 for prompt in completion.prompts)
    assert len(result["trace"][0]["observation"]) <= 24
    assert result["trace"][0]["retrieval_results"] == "x" * 100
    assert all(item["max_tokens"] == 800 for item in completion.configs)
    assert all(item["temperature"] == 1.0 and item["top_p"] == 1.0 for item in completion.configs)


def test_stepsearch_official_metrics_retrieval_and_compact_io() -> None:
    assert official_answer_metrics("The New York", ["New York"]) == (1.0, 1.0)
    assert official_answer_metrics("New York city", ["New York"]) == (0.0, 0.8)
    retriever = object.__new__(StepSearchRetriever)
    retriever.topk = 1
    formatted = retriever._format_documents({
        "result": [[{"document": {"contents": "Title\nFirst line\nSecond line"}}]]
    })
    assert formatted == "Doc 1<## Title: Title ##> First line\nSecond line\n"
    class CapturedWriter(StringIO):
        def close(self) -> None:
            self.flush()

    output = CapturedWriter()
    with patch.object(Path, "open", return_value=output):
        write_compact_json(Path("result.json"), {"answer": [1, 2]})
    assert output.getvalue() == '{"answer":[1,2]}\n'


def test_stepsearch_parser_prefers_canonical_xml() -> None:
    assert stepsearch_first_event("<plan>x</plan>\n<search>who?</search>") == (
        "search",
        "who?",
        "<plan>x</plan>\n<search>who?</search>",
    )


def test_stepsearch_official_parser_is_strict_xml() -> None:
    assert stepsearch_official_first_event("<plan>x</plan>\n<search>who?</search>") == (
        "search", "who?", "<plan>x</plan>\n<search>who?</search>"
    )
    assert stepsearch_official_first_event("1. **answer** Beijing") is None
    assert stepsearch_official_first_event("<ANSWER>Beijing</ANSWER>") is None


def test_stepsearch_official_loop_generates_final_answer_after_search_limit() -> None:
    class Completion:
        def __init__(self):
            self.prompts = []
            self.outputs = ["<search>capital of France</search>", "<answer>Paris</answer>"]

        async def generate(self, prompt, config):
            self.prompts.append(prompt)
            return self.outputs.pop(0)

    class Retriever:
        async def retrieve(self, query):
            assert query == "capital of France"
            return "Doc 1<## Title: France ##> Paris is the capital.\n"

    completion = Completion()
    config = protocol()
    config.update({"parser_mode": "strict", "max_turns": 1})
    result = asyncio.run(run_one(
        {"question": "What is the capital of France?"}, config, None,
        completion, Retriever(),
    ))
    assert result["answer"] == "Paris"
    assert result["turns"] == 2
    assert result["search_calls"] == 1
    assert FINAL_SEARCH_WARNING in completion.prompts[1]


def test_stepsearch_parser_recovers_observed_search_variants() -> None:
    assert stepsearch_first_event("plan: x\nsearch: who?\n<information>hallucinated</information>") == (
        "search",
        "who?",
        "plan: x\n<search>who?</search>",
    )
    assert stepsearch_first_event("plan: x\n(search) who? </search>") == (
        "search",
        "who?",
        "plan: x\n<search>who?</search>",
    )
    assert stepsearch_first_event("plan: x\nsearch: <search>who?</search>") == (
        "search",
        "who?",
        "plan: x\nsearch: <search>who?</search>",
    )


def test_stepsearch_parser_recovers_observed_answer_variants() -> None:
    for text in (
        "plan: done\n(answer) Beijing</answer>",
        "plan: done\nanswer: Beijing\n</answer>",
        "plan: done\n1answer: Beijing\n</answer>",
        "plan: done\n8. **answer** Beijing\n<warning>other text</warning>",
    ):
        assert stepsearch_first_event(text) == (
            "answer",
            "Beijing",
            "plan: done\n<answer>Beijing</answer>",
        )


def test_stepsearch_parser_stops_at_first_action() -> None:
    parsed = stepsearch_first_event(
        "plan: x\n(search) who? </search>\n(answer) stale answer </answer>"
    )
    assert parsed == ("search", "who?", "plan: x\n<search>who?</search>")
