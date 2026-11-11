from copy import deepcopy
from unittest import mock

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import FSDPActorConfig


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(64, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=0,
        **kwargs,
    ):
        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        scores = query @ key.transpose(-1, -2) / hidden.shape[-1] ** 0.5
        causal = torch.triu(
            torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device), diagonal=1
        )
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(~attention_mask[:, None, :].bool(), torch.finfo(scores.dtype).min)
        hidden = torch.softmax(scores, dim=-1) @ value
        logits = self.output(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return CausalLMOutput(logits=logits)


def make_actor(model, *, trim):
    config = FSDPActorConfig(
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=2,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        response_only_logits=True,
        trim_padded_sequences=trim,
    )
    with mock.patch("torch.distributed.get_rank", return_value=0):
        return DataParallelPPOActor(config=config, actor_module=model)


def make_micro_batch():
    input_ids = torch.tensor(
        [
            [0, 0, 0, 4, 5, 6, 7, 8, 9, 0, 0, 0],
            [0, 0, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ]
    )
    position_ids = torch.tensor(
        [
            [0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0],
        ]
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": input_ids[:, -4:],
        "response_mask": attention_mask[:, -4:],
    }


def test_padded_trimming_preserves_valid_log_probs_and_gradients():
    torch.manual_seed(7)
    reference_model = TinyCausalLM()
    trimmed_model = deepcopy(reference_model)
    reference_actor = make_actor(reference_model, trim=False)
    trimmed_actor = make_actor(trimmed_model, trim=True)
    micro_batch = make_micro_batch()

    _, reference_log_probs = reference_actor._forward_micro_batch(
        micro_batch, temperature=1.0, preserve_fsdp_output_hook=True
    )
    _, trimmed_log_probs = trimmed_actor._forward_micro_batch(
        micro_batch, temperature=1.0, preserve_fsdp_output_hook=True
    )
    response_mask = micro_batch["response_mask"].bool()
    torch.testing.assert_close(trimmed_log_probs[response_mask], reference_log_probs[response_mask])
    assert torch.count_nonzero(trimmed_log_probs[~response_mask]) == 0

    reference_log_probs[response_mask].sum().backward()
    trimmed_log_probs[response_mask].sum().backward()
    for reference_parameter, trimmed_parameter in zip(
        reference_model.parameters(), trimmed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(trimmed_parameter.grad, reference_parameter.grad)


def test_non_prefix_response_mask_only_trims_common_prompt_padding():
    micro_batch = make_micro_batch()
    micro_batch["response_mask"][0] = torch.tensor([1, 0, 1, 0])
    trimmed = DataParallelPPOActor._trim_padded_micro_batch(
        micro_batch["input_ids"],
        micro_batch["attention_mask"],
        micro_batch["position_ids"],
        micro_batch["responses"],
        micro_batch["response_mask"],
    )
    input_ids, _, _, responses, response_length = trimmed
    assert input_ids.shape[-1] == 10
    assert responses.shape[-1] == 4
    assert response_length == 4
