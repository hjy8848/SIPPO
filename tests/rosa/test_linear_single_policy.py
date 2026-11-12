import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qa_manager.ROSA.linear_protocol import (
    LinearActionKind,
    LinearTrajectoryState,
    build_linear_messages,
    exact_match_answer,
    parse_linear_action,
)

try:
    import omegaconf  # noqa: F401
    import ray  # noqa: F401
    import qa_manager.ROSA.linear_agent_loop as LINEAR_AGENT_LOOP_MODULE
except ModuleNotFoundError:
    LINEAR_AGENT_LOOP_MODULE = None

WORKFLOW_MODULE_PATH = (
    Path(__file__).parents[2] / "verl" / "trainer" / "ppo" / "workflow_dag_opsd_grpo.py"
)
WORKFLOW_SPEC = importlib.util.spec_from_file_location(
    "linear_workflow_batching_impl", WORKFLOW_MODULE_PATH
)
WORKFLOW_MODULE = importlib.util.module_from_spec(WORKFLOW_SPEC)
assert WORKFLOW_SPEC.loader is not None
WORKFLOW_SPEC.loader.exec_module(WORKFLOW_MODULE)
validate_linear_full_trajectory_minibatches = (
    WORKFLOW_MODULE.validate_linear_full_trajectory_minibatches
)


def test_reason_search_is_atomic_and_standalone_reason_is_rejected():
    action = parse_linear_action(
        "<reason_search><reason>Combine the two dates.</reason>"
        "<query>Which event happened first?</query></reason_search>"
    )

    assert action.kind == LinearActionKind.REASON_SEARCH
    assert action.reason == "Combine the two dates."
    assert action.query == "Which event happened first?"
    with pytest.raises(ValueError, match="standalone_reason"):
        parse_linear_action("<reason>Combine the two dates.</reason>")
    with pytest.raises(ValueError, match="empty_reason_search_query"):
        parse_linear_action(
            "<reason_search><reason>Combine facts.</reason><query> </query></reason_search>"
        )


def test_linear_state_allows_multiple_searches_then_answer():
    state = LinearTrajectoryState(max_tool_calls=3)
    assert state.apply(parse_linear_action("<search>first fact</search>"))
    assert state.apply(parse_linear_action("<search>second fact</search>"))
    assert state.apply(
        parse_linear_action(
            "<reason_search><reason>Bridge the facts.</reason>"
            "<query>third fact</query></reason_search>"
        )
    )
    assert not state.apply(parse_linear_action("<answer>target</answer>"))

    assert state.valid
    assert state.tool_calls == 3
    assert state.reason_search_count == 1
    assert exact_match_answer(state.answer, ["Target", "another alias"]) == 1.0


def test_linear_minibatches_count_complete_workflows():
    trajectory_ids = np.asarray(
        [f"workflow-{index}" for index in range(96)], dtype=object
    )
    assert validate_linear_full_trajectory_minibatches(trajectory_ids, 24) == 4
    with pytest.raises(ValueError, match="more than one row"):
        validate_linear_full_trajectory_minibatches(
            np.asarray(["workflow-0", "workflow-0"], dtype=object), 1
        )
    with pytest.raises(ValueError, match="not divisible"):
        validate_linear_full_trajectory_minibatches(trajectory_ids, 25)


class _FakeTokenizer:
    action_text = {
        1: "<search>first fact</search>",
        2: "<search>second fact</search>",
        3: (
            "<reason_search><reason>Combine both observations.</reason>"
            "<query>bridge fact</query></reason_search>"
        ),
        4: "<answer>Target</answer>",
    }

    def __init__(self):
        self.observation_count = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt in {True, False}
        assert tokenize is True
        if messages == [{}]:
            return [900]
        if len(messages) == 1 and messages[0].get("role") == "user":
            self.observation_count += 1
            return [900, 700 + self.observation_count]
        return [101, 102]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return self.action_text[int(token_ids[0])]


class _FakeRetriever:
    def __init__(self, tokenizer, config):
        del tokenizer, config
        self.queries = []

    def retrieval(self, query, top_k):
        self.queries.append((query, top_k))
        return [f"Document for {query}"]


class _ErrorRetriever:
    def __init__(self, tokenizer, config):
        del tokenizer, config

    def retrieval(self, query, top_k):
        del query, top_k
        return ["[Search API Error] timed out"]


class _FakeServerManager:
    def __init__(self, outputs=(1, 2, 3, 4)):
        self.outputs = iter(outputs)

    async def generate(
        self, request_id, *, prompt_ids, sampling_params, image_data=None
    ):
        del request_id, prompt_ids, sampling_params, image_data
        token_id = next(self.outputs)
        return SimpleNamespace(token_ids=[token_id], log_probs=[-0.1 * token_id])


def test_agent_loop_masks_observations_and_keeps_one_full_trajectory(monkeypatch):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    pytest.importorskip("ray")
    if LINEAR_AGENT_LOOP_MODULE is None:
        pytest.skip("linear AgentLoop runtime dependencies are unavailable")
    module = LINEAR_AGENT_LOOP_MODULE

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 32,
                    "response_length": 64,
                    "calculate_log_probs": True,
                }
            },
            "qa_manager": {
                "retrieval_topk": 3,
                "linear": {
                    "max_tool_calls": 3,
                    "max_actions": 4,
                    "max_action_tokens": 8,
                },
            },
        }
    )
    monkeypatch.setattr(module, "ROSA_AgentManager", _FakeRetriever)
    module.LinearRosaAgentLoop._class_initialized = False
    tokenizer = _FakeTokenizer()

    async def run_loop():
        loop = module.LinearRosaAgentLoop(
            trainer_config=SimpleNamespace(config=config),
            server_manager=_FakeServerManager(),
            tokenizer=tokenizer,
            processor=None,
        )
        return await loop.run(
            {"temperature": 0.8, "top_p": 0.95, "logprobs": True},
            raw_prompt=build_linear_messages("What is the target?"),
            linear_question="What is the target?",
            linear_answer=["Target"],
            linear_data_source="unit-test",
            linear_question_uid="question-0",
            linear_trajectory_id="workflow-0",
            linear_sample_id=0,
        )

    output = asyncio.run(run_loop())
    module.LinearRosaAgentLoop._class_initialized = False

    assert output.prompt_ids == [101, 102]
    assert output.response_ids == [1, 701, 2, 702, 3, 703, 4]
    assert output.response_mask == [1, 0, 1, 0, 1, 0, 1]
    assert output.response_logprobs == pytest.approx(
        [-0.1, 0.0, -0.2, 0.0, -0.3, 0.0, -0.4]
    )
    assert output.reward_score == 1.0
    assert output.extra_fields["workflow_valid"] is True
    assert output.extra_fields["linear_tool_calls"] == 3
    assert output.extra_fields["linear_reason_search_count"] == 1
    assert output.extra_fields["trajectory_id"] == "workflow-0"


def test_retrieval_error_invalidates_workflow_and_caps_reward_at_zero(monkeypatch):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    pytest.importorskip("ray")
    if LINEAR_AGENT_LOOP_MODULE is None:
        pytest.skip("linear AgentLoop runtime dependencies are unavailable")
    module = LINEAR_AGENT_LOOP_MODULE

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 32,
                    "response_length": 64,
                    "calculate_log_probs": True,
                }
            },
            "qa_manager": {
                "retrieval_topk": 3,
                "linear": {
                    "max_tool_calls": 3,
                    "max_actions": 4,
                    "max_action_tokens": 8,
                },
            },
        }
    )
    monkeypatch.setattr(module, "ROSA_AgentManager", _ErrorRetriever)
    module.LinearRosaAgentLoop._class_initialized = False

    async def run_loop():
        loop = module.LinearRosaAgentLoop(
            trainer_config=SimpleNamespace(config=config),
            server_manager=_FakeServerManager(outputs=(1,)),
            tokenizer=_FakeTokenizer(),
            processor=None,
        )
        return await loop.run(
            {"temperature": 0.8, "top_p": 0.95, "logprobs": True},
            raw_prompt=build_linear_messages("What is the target?"),
            linear_question="What is the target?",
            linear_answer=["Target"],
            linear_data_source="unit-test",
            linear_question_uid="question-0",
            linear_trajectory_id="workflow-error",
            linear_sample_id=0,
        )

    output = asyncio.run(run_loop())
    module.LinearRosaAgentLoop._class_initialized = False

    assert output.response_ids == [1]
    assert output.response_mask == [1]
    assert output.response_logprobs == pytest.approx([-0.1])
    assert output.reward_score == 0.0
    assert output.extra_fields["workflow_valid"] is False
    assert output.extra_fields["linear_pipeline_error"] is True
    assert output.extra_fields["linear_invalid_reason"] == "retrieval_error"
