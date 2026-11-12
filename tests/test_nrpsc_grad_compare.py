"""CPU check of the frozen-batch gradient comparison (toy model; real-model numbers come from the NPU run)."""
import os
import numpy as np
import torch
import torch.nn as nn

import sippo
import verl
from sippo.actor_update import SippoActorConfig, SippoPolicyUpdater
from verl.trainer.ppo.nrpsc_objective import PlanView, frozen_tfidf, parse_plan


def test_runtime_imports_come_from_this_tree():
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.dirname(os.path.dirname(os.path.abspath(sippo.__file__))) == tree
    assert os.path.abspath(verl.__file__).startswith(tree + os.sep), verl.__file__
    from verl.trainer.ppo import nrpsc_objective
    assert os.path.abspath(nrpsc_objective.__file__).startswith(tree + os.sep)


def test_compare_gradients_reports_planner_only_support_and_keeps_parameters(monkeypatch):
    torch.manual_seed(0)
    model = nn.Embedding(64, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = SippoActorConfig(ppo_mini_batch_size=64, ppo_micro_batch_size_per_gpu=4, use_dynamic_bsz=False,
                           use_remove_padding=False, response_only_logits=False)
    updater = SippoPolicyUpdater(model, optimizer, cfg, device='cpu')
    monkeypatch.setattr('sippo.actor_update.forward_log_probs',
                        lambda model, tensors, temperature, cfg, device_type: model(tensors['responses']).squeeze(-1))
    rows = 16
    ids = torch.arange(rows * 8).reshape(rows, 8) % 64
    responses = ids[:, -4:]
    adv = torch.randn(rows, 4)
    tensors = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
               'position_ids': torch.arange(8).repeat(rows, 1), 'responses': responses,
               'response_mask': torch.ones(rows, 4), 'old_log_probs': torch.zeros(rows, 4),
               'advantages': adv, 'actor_dp_valid_rows': torch.ones(rows, dtype=torch.bool)}
    node_mask = torch.tensor([True, True, False, False])
    end_mask = torch.tensor([False, False, False, True])
    views = np.empty(rows, dtype=object)
    for i in range(8):
        views[i] = PlanView(parse_plan(f'<search id="1">entity{i}</search>'), (node_mask,),
                            (42,), tuple(responses[i].tolist()), end_mask)
    roles = np.asarray(['PlanningAgent'] * 8 + ['SearchAgent'] * 8, dtype=object)
    metadata = {'uid': np.asarray(['q'] * rows, dtype=object),
                'trajectory_id': np.asarray([f'w{i % 8}' for i in range(rows)], dtype=object),
                'role': roles, 'outcome_exact': np.asarray(([1.] + [0.] * 7) * 2),
                'workflow_step': np.asarray([0] * 8 + [1] * 8, dtype=np.int64), 'nrpsc_plan_view': views}
    meta = {'temperature': 1., 'planner_outcome_contrast_minibatch_mode': 'question_preserving',
            'planner_outcome_contrast_mode': 'nrpsc', 'planner_outcome_contrast_beta': 1.,
            'planner_outcome_contrast_loss_weight': 0.02, 'planner_outcome_contrast_data_parallel_size': 1,
            'nrpsc_idf': frozen_tfidf([f'entity{i}' for i in range(8)]).idf, 'nrpsc_similarity_threshold': 0.8}
    before = model.weight.detach().clone()
    result = updater.compare_gradients(tensors, metadata, meta, psc_weight=0.02)
    assert torch.equal(before, model.weight.detach())  # no optimizer step
    assert len(result['minibatches']) == 1
    rec = result['minibatches'][0]
    for name in ('psc', 'nrpsc'):
        assert rec[f'grad_norm/{name}'] > 0, name
        assert rec[f'{name}/nonzero_tokens'] > 0
        assert rec[f'{name}/nonzero_tokens_outside_planner_rows'] == 0
        assert rec[f'{name}/nonzero_tokens_outside_response_mask'] == 0
    assert rec['grad_norm/base'] > 0
    assert -1.0001 <= rec['cosine/psc__nrpsc'] <= 1.0001
    assert rec['matched_weight_norm/nrpsc'] > 0