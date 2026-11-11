"""Single-policy ROSA rollout with masked retrieval observations."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf

from qa_manager.ROSA.linear_protocol import (
    LinearTrajectoryState,
    build_linear_messages,
    exact_match_answer,
    format_search_observation,
    parse_linear_action,
)
from qa_manager.ROSA.rosa_agent_manager import ROSA_AgentManager
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.utils.profiler import simple_timer


class LinearRosaAgentLoop(AgentLoopBase):
    """Generate one complete causal trajectory for one question."""

    @classmethod
    def init_class(cls, config, tokenizer, processor=None, **kwargs):
        del processor, kwargs
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls.qa_manager = ROSA_AgentManager(tokenizer, config)
        cls.prompt_length = int(config.actor_rollout_ref.rollout.prompt_length)
        cls.response_length = int(config.actor_rollout_ref.rollout.response_length)
        cls.calculate_log_probs = bool(
            config.actor_rollout_ref.rollout.calculate_log_probs
        )
        cls.max_tool_calls = int(
            OmegaConf.select(config, "qa_manager.linear.max_tool_calls", default=4)
        )
        cls.max_actions = int(
            OmegaConf.select(
                config,
                "qa_manager.linear.max_actions",
                default=cls.max_tool_calls + 1,
            )
        )
        cls.max_action_tokens = int(
            OmegaConf.select(config, "qa_manager.linear.max_action_tokens", default=256)
        )
        if cls.max_tool_calls < 1:
            raise ValueError("qa_manager.linear.max_tool_calls must be positive")
        if cls.max_actions < 1:
            raise ValueError("qa_manager.linear.max_actions must be positive")
        if cls.max_action_tokens < 1:
            raise ValueError("qa_manager.linear.max_action_tokens must be positive")
        cls.chat_prefix = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True
        )

    @staticmethod
    def _unwrap(value):
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        try:
            if getattr(value, "ndim", 0) == 0:
                return value.item()
        except (AttributeError, ValueError):
            pass
        return value

    def _encode_initial_prompt(self, messages: list[dict[str, str]]) -> list[int]:
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"linear prompt has {len(prompt_ids)} tokens, exceeding {self.prompt_length}"
            )
        return list(prompt_ids)

    def _encode_observation(self, observation: str) -> list[int]:
        segment = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": observation}],
            add_generation_prompt=True,
            tokenize=True,
        )
        prefix_length = len(self.chat_prefix)
        if segment[:prefix_length] != self.chat_prefix:
            raise ValueError(
                "chat template prefix changed while encoding an observation"
            )
        return list(segment[prefix_length:])

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        question = str(self._unwrap(kwargs.get("linear_question", "")) or "").strip()
        references = self._unwrap(kwargs.get("linear_answer", ""))
        messages = kwargs.get("raw_prompt") or build_linear_messages(question)
        messages = list(messages)
        prompt_ids = self._encode_initial_prompt(messages)
        initial_prompt_ids = list(prompt_ids)

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        action_texts: list[str] = []
        metrics: dict[str, float] = {}
        request_id = uuid4().hex
        state = LinearTrajectoryState(max_tool_calls=self.max_tool_calls)

        for _ in range(self.max_actions):
            remaining = self.response_length - len(response_ids)
            if remaining < 1:
                state.invalidate("response_budget_exhausted")
                break
            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_tokens"] = min(self.max_action_tokens, remaining)
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=turn_sampling_params,
                )
            generated_ids = list(output.token_ids)
            if not generated_ids:
                state.invalidate("empty_generation")
                break
            if self.calculate_log_probs and output.log_probs is None:
                raise RuntimeError(
                    "linear rollout requested log-probs but the server returned none"
                )

            prompt_ids.extend(generated_ids)
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            if output.log_probs is not None:
                response_logprobs.extend(float(value) for value in output.log_probs)

            action_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            action_texts.append(action_text)
            try:
                action = parse_linear_action(action_text)
            except ValueError as exc:
                state.invalidate(str(exc))
                break

            should_retrieve = state.apply(action)
            if state.terminated:
                break
            if not should_retrieve:
                break

            with simple_timer("tool_calls", metrics):
                documents = await asyncio.to_thread(
                    self.qa_manager.retrieval,
                    action.query,
                    int(
                        OmegaConf.select(
                            self.config,
                            "qa_manager.retrieval_topk",
                            default=3,
                        )
                    ),
                )
            if any("[Search API Error]" in str(document) for document in documents):
                state.invalidate("retrieval_error")
                break
            observation = format_search_observation(documents, state.tool_calls)
            observation_ids = self._encode_observation(observation)
            if len(response_ids) + len(observation_ids) >= self.response_length:
                state.invalidate("observation_exceeds_response_budget")
                break
            prompt_ids.extend(observation_ids)
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            if output.log_probs is not None:
                response_logprobs.extend([0.0] * len(observation_ids))
        else:
            state.invalidate("action_budget_exhausted")

        exact = exact_match_answer(state.answer, references) if state.valid else 0.0
        data_source = str(
            self._unwrap(kwargs.get("linear_data_source", "unknown")) or "unknown"
        )
        extra_fields = {
            "uid": str(self._unwrap(kwargs.get("linear_question_uid", ""))),
            "trajectory_id": str(self._unwrap(kwargs.get("linear_trajectory_id", ""))),
            "workflow_sample": int(
                self._unwrap(kwargs.get("linear_sample_id", 0)) or 0
            ),
            "workflow_step": 0,
            "role": "LinearPolicy",
            "workflow_tool_steps": state.tool_calls,
            "outcome_reward": exact,
            "outcome_exact": exact,
            "workflow_valid": state.valid,
            "data_source": data_source,
            "linear_action_count": state.action_count,
            "linear_tool_calls": state.tool_calls,
            "linear_reason_search_count": state.reason_search_count,
            "linear_final_answer": state.answer,
            "linear_invalid_reason": state.invalid_reason,
            "linear_pipeline_error": state.invalid_reason == "retrieval_error",
            "linear_action_trace": action_texts,
        }
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if self.calculate_log_probs else None,
            reward_score=exact,
            num_turns=1 + state.action_count + state.tool_calls,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=extra_fields,
        )
