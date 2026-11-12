"""SIPPO dataset: verl 0.9.1 ``RLHFDataset`` plus the tokenised prompt fields that the JADE workflow rollout expects.

verl 0.9.1 moved prompt tokenisation into the agent loop and its dataset only returns ``raw_prompt``. The JADE
workflow rollout (``sippo/workflow_rollout.py``, verbatim from JADE) uses a dataset row as the template for every
per-step input, so the row must carry ``input_ids / attention_mask / position_ids / raw_prompt_ids`` exactly as the
JADE dataset produced them (text-only path, copied from JADE ``RLHFDataset.__getitem__``).
"""

from __future__ import annotations

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask


class SippoRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        if self.processor is not None:
            raise NotImplementedError("SIPPO W/P runs are text-only")
        messages = row_dict["raw_prompt"]

        if self.apply_chat_template_kwargs.get("chat_template") is None:
            assert hasattr(self.tokenizer, "chat_template"), (
                "chat_template should be provided in apply_chat_template_kwargs or tokenizer config"
            )
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt
        return row_dict
