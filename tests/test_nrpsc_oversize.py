import torch

from verl.trainer.ppo.nrpsc_objective import MAX_ALIGNMENT_NODES, prepare_plan_views


class CharacterTokenizer:
    eos_token_id = None

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_oversize_plan_is_excluded_not_fatal():
    small = '<plan><search id="1">a</search></plan>'
    big = "<plan>" + "".join(f'<search id="{i}">x{i}</search>' for i in range(1, MAX_ALIGNMENT_NODES + 2)) + "</plan>"
    texts = [small, big]
    width = max(map(len, texts))
    responses = torch.zeros(2, width, dtype=torch.long)
    mask = torch.zeros_like(responses)
    for i, text in enumerate(texts):
        responses[i, :len(text)] = torch.tensor(list(map(ord, text)))
        mask[i, :len(text)] = 1
    prompts = torch.tensor([[1, 2]] * 2)
    attention = torch.ones(2, 2 + width, dtype=torch.long)
    views, counts = prepare_plan_views(CharacterTokenizer(), responses, mask, prompts, attention, ["PlanningAgent"] * 2)
    assert views[0] is not None and views[1] is None
    assert counts["oversize_plans"] == 1 and counts["usable_rows"] == 1