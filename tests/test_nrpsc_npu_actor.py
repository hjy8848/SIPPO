"""CPU integration test of the NPU updater's actual n=8 optimizer path."""
import numpy as np
import torch
import torch.nn as nn
import hashlib
import json
import math
from omegaconf import OmegaConf
from tensordict import TensorDict

from sippo.actor_update import SippoActorConfig, SippoPolicyUpdater
from sippo.workflow_rollout import WorkflowRolloutMixin
from verl import DataProto
from verl.trainer.ppo.nrpsc_objective import PlanView, frozen_tfidf, parse_plan


def test_npu_updater_eight_siblings_changes_model_only_through_nrpsc(monkeypatch):
    model = nn.Embedding(64, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = SippoActorConfig(ppo_mini_batch_size=32, ppo_micro_batch_size_per_gpu=4,
                           use_dynamic_bsz=False, use_remove_padding=False,
                           response_only_logits=False)
    updater = SippoPolicyUpdater(model, optimizer, cfg, device='cpu')
    monkeypatch.setattr('sippo.actor_update.forward_log_probs',
                        lambda model, tensors, temperature, cfg, device_type:
                        model(tensors['responses']).squeeze(-1))
    ids = torch.arange(64).reshape(8, 8) % 64
    responses = ids[:, -4:]
    tensors = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
               'position_ids': torch.arange(8).repeat(8, 1), 'responses': responses,
               'response_mask': torch.ones(8, 4), 'old_log_probs': torch.zeros(8, 4),
               'advantages': torch.zeros(8, 4), 'actor_dp_valid_rows': torch.ones(8, dtype=torch.bool)}
    node_mask = torch.tensor([True, True, False, False])
    end_mask = torch.tensor([False, False, False, True])
    views = np.empty(8, dtype=object)
    for i in range(8):
        views[i] = PlanView(parse_plan(f'<search id="1">entity{i}</search>'), (node_mask,),
                            (42,), tuple(responses[i].tolist()), end_mask)
    metadata = {'uid': np.asarray(['q'] * 8, dtype=object),
                'trajectory_id': np.asarray([f'w{i}' for i in range(8)], dtype=object),
                'role': np.asarray(['PlanningAgent'] * 8, dtype=object),
                'outcome_exact': np.asarray([1.] + [0.] * 7),
                'workflow_step': np.zeros(8, dtype=np.int64), 'nrpsc_plan_view': views}
    meta = {'temperature': 1., 'planner_outcome_contrast_minibatch_mode': 'question_preserving',
            'planner_outcome_contrast_mode': 'nrpsc', 'planner_outcome_contrast_beta': 1.,
            'planner_outcome_contrast_loss_weight': 0.02,
            'planner_outcome_contrast_data_parallel_size': 1,
            'nrpsc_idf': frozen_tfidf([f'entity{i}' for i in range(8)]).idf,
            'nrpsc_similarity_threshold': 0.8}
    before = model.weight.detach().clone()
    result = updater.update(tensors, metadata, meta)
    assert result['actor/optimizer_steps_expected'] == [1.0]
    assert result['actor/optimizer_steps_executed'] == [1.0]
    assert result['aux/nrpsc/informative_class_pairs'][0] > 0
    assert not torch.equal(before, model.weight.detach())


def test_npu_trainer_to_actor_n8_with_frozen_idf(tmp_path, monkeypatch):
    class CharacterTokenizer:
        eos_token_id = None
        def decode(self, ids, **kwargs):
            return ''.join(map(chr, ids))
        def __call__(self, text, **kwargs):
            return {'input_ids': list(map(ord, text)),
                    'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    artifact = {'schema': 'nrpsc_tfidf_v1', 'document_count': 1,
                'corpus_sha256': 'a' * 64, 'unknown_idf': math.log(2) + 1,
                'idf': {'entity0': 1.0}}
    path = tmp_path / 'idf.json'
    path.write_text(json.dumps(artifact), encoding='utf8')
    config = OmegaConf.create({
        'algorithm': {'planner_outcome_contrast': {'mode': 'nrpsc', 'minibatch_mode': 'question_preserving',
            'loss_weight': 0.02, 'nrpsc_expected_siblings': 8, 'nrpsc_idf_path': str(path),
            'nrpsc_idf_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}},
        'actor_rollout_ref': {'actor': {'ppo_epochs': 1, 'loss_agg_mode': 'token-mean',
            'policy_loss': {'loss_mode': 'vanilla'}, 'use_dynamic_bsz': False}},
        'trainer': {'n_gpus_per_node': 1, 'nnodes': 1}, 'qa_manager': {'workflow_rollout_n': 8},
    })
    trainer = WorkflowRolloutMixin()
    trainer.config, trainer.tokenizer = config, CharacterTokenizer()
    texts = [f'<plan><search id="1">entity{i}</search></plan>' for i in range(8)]
    width = max(map(len, texts))
    response = torch.zeros(8, width, dtype=torch.long)
    response_mask = torch.zeros_like(response)
    for i, text in enumerate(texts):
        response[i, :len(text)] = torch.tensor(list(map(ord, text)))
        response_mask[i, :len(text)] = 1
    prompts = torch.tensor([[1, 2]] * 8)
    inputs = torch.cat((prompts, response), dim=-1)
    batch = TensorDict({'prompts': prompts, 'responses': response, 'response_mask': response_mask,
                        'input_ids': inputs, 'attention_mask': torch.cat((torch.ones_like(prompts), response_mask), dim=-1),
                        'position_ids': torch.arange(inputs.shape[-1]).repeat(8, 1),
                        'old_log_probs': torch.zeros(8, width), 'advantages': torch.zeros(8, width),
                        'actor_dp_valid_rows': torch.ones(8, dtype=torch.bool)}, batch_size=[8])
    metadata = {'uid': np.asarray(['q'] * 8, dtype=object),
                'trajectory_id': np.asarray([f'w{i}' for i in range(8)], dtype=object),
                'workflow_sample': np.arange(8), 'role': np.asarray(['PlanningAgent'] * 8, dtype=object),
                'outcome_exact': np.asarray([1.] + [0.] * 7), 'workflow_step': np.zeros(8, dtype=np.int64)}
    data = DataProto(batch=batch, non_tensor_batch=metadata, meta_info={'temperature': 1.0})
    data, metrics = trainer._apply_planner_outcome_contrast(data)
    assert metrics['aux/nrpsc/complete_sibling_questions'] == 1
    assert metrics['aux/nrpsc/usable_rows'] == 8
    model = nn.Embedding(128, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    updater = SippoPolicyUpdater(model, optimizer, SippoActorConfig(
        ppo_mini_batch_size=256, ppo_micro_batch_size_per_gpu=4, use_dynamic_bsz=False,
        use_remove_padding=False, response_only_logits=False), device='cpu')
    monkeypatch.setattr('sippo.actor_update.forward_log_probs',
                        lambda model, tensors, temperature, cfg, device_type:
                        model(tensors['responses']).squeeze(-1))
    keys = ('input_ids', 'attention_mask', 'position_ids', 'responses', 'response_mask',
            'old_log_probs', 'advantages', 'actor_dp_valid_rows')
    before = model.weight.detach().clone()
    result = updater.update({key: data.batch[key] for key in keys}, data.non_tensor_batch, data.meta_info)
    assert result['actor/optimizer_steps_executed'] == [1.0]
    assert result['aux/nrpsc/informative_class_pairs'][0] > 0
    assert not torch.equal(before, model.weight.detach())
