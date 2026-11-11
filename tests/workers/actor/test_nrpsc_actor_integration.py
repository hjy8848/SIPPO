"""Exercise the actor's real NR-PSC refresh without running an optimizer."""
from types import MethodType

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.nrpsc_objective import PlanView, frozen_tfidf, parse_plan
from verl.workers.actor.dp_actor import DataParallelPPOActor


def test_actor_refresh_consumes_node_views_and_ignores_other_roles(monkeypatch):
    mask = torch.tensor([True, False, False, False])
    end = torch.tensor([False, False, False, True])
    a = PlanView(parse_plan('<search id="1">alpha</search>'), (mask,), (9,), (1, 2, 3, 0), end)
    b = PlanView(parse_plan('<search id="1">beta</search>'), (mask,), (9,), (4, 5, 6, 0), end)
    views = np.empty(3, dtype=object)
    views[:] = [a, b, None]
    data = DataProto(
        batch=TensorDict({'response_mask': torch.ones(3, 4),
                          'actor_dp_valid_rows': torch.ones(3, dtype=torch.bool)}, batch_size=[3]),
        non_tensor_batch={'uid': np.asarray(['q', 'q', 'q2'], dtype=object),
                          'trajectory_id': np.asarray(['a', 'b', 'c'], dtype=object),
                          'role': np.asarray(['PlanningAgent', 'PlanningAgent', 'ReasonerAgent'], dtype=object),
                          'outcome_exact': np.asarray([1., 0., 0.]),
                          'nrpsc_plan_view': views},
        meta_info={'nrpsc_idf': frozen_tfidf(['alpha', 'beta']).idf,
                   'nrpsc_similarity_threshold': 0.8},
    )
    actor = object.__new__(DataParallelPPOActor)
    actor._current_minibatch_log_probs = MethodType(
        lambda self, mini_batch, **kwargs: torch.zeros(3, 4), actor)
    monkeypatch.setattr('verl.workers.actor.dp_actor.get_device_id', lambda: 'cpu')
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    metrics = actor._refresh_nrpsc(data, temperature=0.8, beta=1., data_parallel_size=1)
    coeff = data.batch['planner_outcome_contrast_coefficients']
    assert metrics['aux/nrpsc/G'] == 1
    assert coeff[0, 0] < 0 < coeff[1, 0]
    assert torch.count_nonzero(coeff[:, 1:]) == 0
    assert torch.count_nonzero(coeff[2]) == 0
