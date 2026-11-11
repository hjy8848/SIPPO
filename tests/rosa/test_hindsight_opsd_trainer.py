import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_dagent_hindsight_opsd.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hindsight_trainer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_search_query_mask_excludes_xml_and_reason_tokens():
    module = load_module()
    response = '<plan><search id="1">alpha</search><reason id="2">#1#</reason></plan>'
    _, mask = module.response_token_mask(CharacterTokenizer(), response, "search_query")
    selected = "".join(character for character, keep in zip(response, mask.tolist()) if keep)
    assert selected == "alpha"


def test_reason_structure_mask_selects_reason_and_dependent_search_only():
    module = load_module()
    response = (
        '<plan><search id="1">alpha</search><search id="2">beta</search>'
        '<reason id="3">compare #1# and #2#</reason>'
        '<search id="4">detail about #3#</search></plan>'
    )
    _, mask = module.reason_structure_token_mask(CharacterTokenizer(), response)
    selected = "".join(character for character, keep in zip(response, mask.tolist()) if keep)
    assert selected == (
        '<reason id="3">compare #1# and #2#</reason>'
        '<search id="4">detail about #3#</search>'
    )


def test_jsd_is_zero_for_identical_logits_and_positive_otherwise():
    module = load_module()
    left = torch.tensor([[1.0, 0.0, -1.0]])
    same = module.divergence_loss(left, left.clone(), "jsd", 1.0)
    shifted = module.divergence_loss(left, -left, "jsd", 1.0)
    assert torch.isclose(same, torch.tensor(0.0), atol=1e-7)
    assert shifted > 0


def test_forward_kl_has_gradient_on_student_only():
    module = load_module()
    student = torch.tensor([[0.2, -0.1]], requires_grad=True)
    teacher = torch.tensor([[-0.3, 0.4]])
    loss = module.divergence_loss(student, teacher, "forward_kl", 1.0)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_reason_can_distill_on_successful_reference_prefix():
    module = load_module()
    row = {
        "source_kind": "reason",
        "response": "failed plan",
        "positive_reference_response": "successful reason plan",
    }

    class Args:
        reason_distill_response = "positive_reference"

    assert module.distillation_response(row, Args()) == "successful reason plan"
    Args.reason_distill_response = "failed"
    assert module.distillation_response(row, Args()) == "failed plan"


def test_preference_loss_rewards_margin_above_frozen_reference():
    relative_margin = torch.tensor(0.0, requires_grad=True)
    beta = 2.0
    initial = -torch.nn.functional.logsigmoid(beta * relative_margin)
    improved = -torch.nn.functional.logsigmoid(beta * torch.tensor(1.0))
    initial.backward()
    assert torch.isclose(initial.detach(), torch.tensor(0.6931472), atol=1e-6)
    assert improved < initial.detach()
    assert relative_margin.grad < 0
