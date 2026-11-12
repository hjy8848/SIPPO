import torch

from qa_manager.ROSA.rosa_agent_manager import ROSA_AgentManager


def _manager():
    return ROSA_AgentManager.__new__(ROSA_AgentManager)


class _PromptTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "rendered prompt"

    def __call__(self, text, **kwargs):
        assert text == "rendered prompt"
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


def test_prompt_rendering_forwards_explicit_thinking_mode():
    manager = _manager()
    manager.tokenizer = _PromptTokenizer()
    manager.model_enable_thinking = False
    manager.max_prompt_length = 8
    manager.truncation = "left"
    manager.return_raw_chat = True

    result = manager._messages_to_ids([{"role": "user", "content": "question"}])

    assert manager.tokenizer.template_kwargs == {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    assert result["raw_prompt_ids"] == [1, 2]


def test_cp_planner_uses_typed_system_prompt_instead_of_legacy_builder():
    manager = _manager()
    manager.contract_preserving_dag_v2 = True
    manager.Prompts = {"PlanningAgent": "TYPED CONTRACT PROMPT"}

    messages = manager._build_messages_by_agent("PlanningAgent", "Who won?")

    assert messages == [
        {"role": "system", "content": "TYPED CONTRACT PROMPT"},
        {
            "role": "user",
            "content": "Question: Who won?\nGenerate the execution plan.",
        },
    ]


def test_agent_protocol_rejects_text_outside_required_output():
    manager = _manager()
    cases = [
        ("AnswerGenerationAgent", "explanation <conclusion>John Smith</conclusion>"),
        ("ReasonerAgent", "because facts <conclusion>yes</conclusion>"),
        ("AnswerSummarizationAgent", "The answer is <answer>Paris</answer>"),
    ]
    for agent, response in cases:
        assert manager.parse_response(agent, response)["is_valid"] is False


def test_planner_protocol_rejects_text_outside_plan():
    result = _manager().parse_response(
        "PlanningAgent",
        'explanation <plan><search id="1">Who directed Film A?</search></plan>',
        max_nodes=5,
        question="Who directed Film A?",
    )
    assert result["is_valid"] is False


def test_query_rewrite_protocol_rejects_extra_lines_and_tags():
    manager = _manager()
    assert manager.parse_response("QueryRewriteAgent", "Ada Lovelace husband")["is_valid"] is True
    assert manager.parse_response("QueryRewriteAgent", "Ada Lovelace husband\nextra")["is_valid"] is False
    assert manager.parse_response("QueryRewriteAgent", "<query>Ada Lovelace husband</query>")["is_valid"] is False


def test_cp_planner_parser_accepts_typed_plan_and_discards_legacy_out():
    manager = _manager()
    manager.contract_preserving_dag_v2 = True
    result = manager.parse_response(
        "PlanningAgent",
        '<plan sink="#1#" answer_type="entity">\n'
        '<search id="1" subject="Ada Lovelace" relation="husband" out="entity">'
        "Who was Ada Lovelace's husband?</search>\n"
        "</plan>",
        max_nodes=5,
        question="Who was Ada Lovelace's husband?",
    )
    assert result["is_valid"] is True
    assert result["answer_sink_id"] == "1"
    assert result["nodes"][0]["query"] == "Who was Ada Lovelace's husband?"
    assert "out" not in result["nodes"][0]


def test_cp_planner_parser_rejects_free_text_plan():
    manager = _manager()
    manager.contract_preserving_dag_v2 = True
    result = manager.parse_response(
        "PlanningAgent",
        '<plan><search id="1">Who was Ada Lovelace\'s husband?</search></plan>',
        max_nodes=5,
        question="Who was Ada Lovelace's husband?",
    )
    assert result["is_valid"] is False
    assert result["reason"] == "invalid_contract_plan"
