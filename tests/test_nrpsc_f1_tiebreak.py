"""Actor-path check of the F1 tie-break: EM-tied plans with different final-answer F1 yield tie-break pairs."""
import numpy as np
import torch
import torch.nn as nn

from sippo.actor_update import SippoActorConfig, SippoPolicyUpdater
from verl.trainer.ppo.nrpsc_objective import PlanView, frozen_tfidf, parse_plan


def _run(source, decile):
    torch.manual_seed(0)
    model = nn.Embedding(64, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = SippoActorConfig(ppo_mini_batch_size=32, ppo_micro_batch_size_per_gpu=4, use_dynamic_bsz=False,
                           use_remove_padding=False, response_only_logits=False)
    updater = SippoPolicyUpdater(model, optimizer, cfg, device='cpu')
    import sippo.actor_update as au
    au.forward_log_probs = lambda m, t, temperature, cfg, device_type: m(t['responses']).squeeze(-1)
    ids = torch.arange(64).reshape(8, 8) % 64
    responses = ids[:, -4:]
    tensors = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'position_ids': torch.arange(8).repeat(8, 1),
               'responses': responses, 'response_mask': torch.ones(8, 4), 'old_log_probs': torch.zeros(8, 4),
               'advantages': torch.zeros(8, 4), 'actor_dp_valid_rows': torch.ones(8, dtype=torch.bool)}
    node, end = torch.tensor([True, True, False, False]), torch.tensor([False, False, False, True])
    views = np.empty(8, dtype=object)
    for i in range(8):
        views[i] = PlanView(parse_plan(f'<search id="1">entity{i}</search>'), (node,), (42,), tuple(responses[i].tolist()), end)
    non = {'uid': np.asarray(['q'] * 8, dtype=object), 'trajectory_id': np.asarray([f'w{i}' for i in range(8)], dtype=object),
           'role': np.asarray(['PlanningAgent'] * 8, dtype=object), 'outcome_exact': np.zeros(8),  # all EM-wrong
           'workflow_step': np.zeros(8, dtype=np.int64), 'nrpsc_plan_view': views,
           'nrpsc_tiebreak_f1_decile': np.asarray(decile, dtype=np.int64)}
    meta = {'temperature': 1., 'planner_outcome_contrast_minibatch_mode': 'question_preserving',
            'planner_outcome_contrast_mode': 'nrpsc', 'planner_outcome_contrast_beta': 1.,
            'planner_outcome_contrast_loss_weight': 0.12, 'planner_outcome_contrast_data_parallel_size': 1,
            'nrpsc_idf': frozen_tfidf([f'entity{i}' for i in range(8)]).idf, 'nrpsc_similarity_threshold': 0.8,
            'nrpsc_tiebreak_weight': 0.3, 'nrpsc_tiebreak_source': source}
    before = model.weight.detach().clone()
    result = updater.update(tensors, non, meta)
    return result, not torch.equal(before, model.weight.detach())


def test_f1_tiebreak_gives_signal_on_all_em_wrong_question():
    result, changed = _run('f1', [6, 0, 0, 0, 0, 0, 0, 0])
    assert result['aux/nrpsc/tiebreak_informative_class_pairs'][0] > 0
    assert result['aux/nrpsc/em_informative_class_pairs'][0] == 0
    assert changed


def test_equal_f1_gives_no_signal():
    result, _ = _run('f1', [3] * 8)
    assert result['aux/nrpsc/informative_class_pairs'][0] == 0


def test_retrieval_source_ignores_f1_field():
    result, _ = _run('retrieval', [6, 0, 0, 0, 0, 0, 0, 0])
    assert result['aux/nrpsc/informative_class_pairs'][0] == 0