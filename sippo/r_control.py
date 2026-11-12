"""R arm: normalization-matched Planner outcome-loss reweighting (pointwise leave-one-out control for PSC).

Experiment E1 in the rebuttal plan asks whether the PSC gain over Workflow-RLOO (W) comes from its *pairwise,
margin-dependent* comparison or merely from putting more weight on the Planner's outcome signal. The control (R) adds
to W an auxiliary Planner loss that uses the same outcome labels and the same grouping/normalisation conventions as PSC
but is a plain pointwise RLOO objective:

    for every informative question q (>=1 successful and >=1 failed eligible Planner workflow, n = k + m workflows)
        A_i = y_i - mean_{j != i} y_j                       (RLOO advantage; y in {0, 1})
        s_i = mean response-token log-probability of workflow i's Planner action     (length-normalised score)
        L_q = - (1/n) * sum_i A_i * s_i
    L_R = (1 / Q_inf) * sum_q L_q                           (mean over informative questions, as in PSC)

so that  grad L_q = -(k m / (n (n-1))) * (mean_+ grad s_i - mean_- grad s_j), the quantity the PSC gradient reduces to in
the zero-margin limit (up to the factor beta/2). The loss is *linear* in the token log-probabilities, so it is represented
exactly as ``sum(log_prob * coefficient)`` -- the same carrier PSC uses -- and its coefficients do not depend on the
policy (no refresh needed).

The overall scale of the auxiliary term is NOT chosen here: the updater multiplies these coefficients by a per-minibatch
scalar so that ||grad of the aux term|| equals ||grad of the weighted PSC term|| on the very same trajectories (see
``SippoPolicyUpdater``).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from verl.trainer.ppo.planner_outcome_contrast import PLANNER_ROLE


def build_planner_pointwise_rloo_coefficients(
    *,
    response_mask: torch.Tensor,
    question_ids,
    trajectory_ids,
    roles,
    outcomes,
    workflow_steps,
    chosen_eligibility=None,
    valid_rows=None,
    data_parallel_size: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token coefficients ``c`` such that ``L_R = sum(log_prob * c)`` (see module docstring). Returns (c, metrics)."""
    if response_mask.ndim != 2:
        raise ValueError("expects a rank-2 response_mask")
    if data_parallel_size < 1:
        raise ValueError("data_parallel_size must be positive")
    row_count = response_mask.shape[0]
    q_ids = np.asarray(question_ids, dtype=object)
    t_ids = np.asarray(trajectory_ids, dtype=object)
    role_arr = np.asarray(roles, dtype=object)
    y = np.asarray(outcomes, dtype=np.float64)
    steps = np.asarray(workflow_steps, dtype=np.int64)
    eligible = (
        np.ones(row_count, dtype=np.bool_) if chosen_eligibility is None else np.asarray(chosen_eligibility, dtype=np.bool_)
    )
    for name, arr in (
        ("question_ids", q_ids),
        ("trajectory_ids", t_ids),
        ("roles", role_arr),
        ("outcomes", y),
        ("workflow_steps", steps),
        ("chosen_eligibility", eligible),
    ):
        if arr.ndim != 1 or arr.shape[0] != row_count:
            raise ValueError(f"{name} must contain one value per batch row")
    if not np.all(np.isclose(y, 0.0, atol=1e-6) | np.isclose(y, 1.0, atol=1e-6)):
        raise ValueError("requires binary exact-match outcomes")
    if valid_rows is None:
        valid = np.ones(row_count, dtype=np.bool_)
    elif isinstance(valid_rows, torch.Tensor):
        valid = valid_rows.detach().cpu().numpy().astype(np.bool_, copy=False).reshape(-1)
    else:
        valid = np.asarray(valid_rows, dtype=np.bool_).reshape(-1)
    mask = response_mask.bool()

    workflow_rows: dict = defaultdict(list)
    w_question: dict = {}
    w_outcome: dict = {}
    w_eligible: dict = {}
    for row in range(row_count):
        if not valid[row]:
            continue
        t = t_ids[row]
        if w_question.setdefault(t, q_ids[row]) != q_ids[row]:
            raise ValueError(f"trajectory {t!r} belongs to multiple questions")
        if not np.isclose(w_outcome.setdefault(t, float(y[row])), y[row], atol=1e-6, rtol=0.0):
            raise ValueError(f"trajectory {t!r} has inconsistent outcomes")
        if w_eligible.setdefault(t, bool(eligible[row])) != bool(eligible[row]):
            raise ValueError(f"trajectory {t!r} has inconsistent chosen eligibility")
        workflow_rows[t].append(row)

    planner_rows: dict = {}
    for t, rows in workflow_rows.items():
        by_step: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            if str(role_arr[row]) == PLANNER_ROLE:
                by_step[int(steps[row])].append(row)
        if not by_step:
            continue
        if len(by_step) != 1:
            raise ValueError(f"trajectory {t!r} has {len(by_step)} logical Planner actions")
        dups = next(iter(by_step.values()))
        counts = [int(mask[r].sum().item()) for r in dups]
        if not counts or counts[0] < 1 or any(c != counts[0] for c in counts[1:]):
            raise ValueError(f"trajectory {t!r}: empty or inconsistent Planner response length")
        planner_rows[t] = dups
    # same eligibility rule as PSC: successful workflows that are not eligible "chosen" examples are dropped
    for t in [t for t in planner_rows if w_outcome[t] > 0.5 and not w_eligible[t]]:
        planner_rows.pop(t)

    by_question: dict = defaultdict(list)
    for t in planner_rows:
        by_question[w_question[t]].append(t)
    informative = []
    for q, wfs in by_question.items():
        wfs.sort(key=str)
        k = sum(w_outcome[t] > 0.5 for t in wfs)
        m = sum(w_outcome[t] < 0.5 for t in wfs)
        if k >= 1 and m >= 1:
            informative.append((q, wfs, k, m))

    coefficients = torch.zeros(response_mask.shape, dtype=torch.float32)
    q_inf = len(informative)
    pos_mass = neg_mass = 0.0
    for _q, wfs, k, m in informative:
        n = k + m
        for t in wfs:
            positive = w_outcome[t] > 0.5
            advantage = (m / (n - 1)) if positive else -(k / (n - 1))
            dups = planner_rows[t]
            tokens = int(mask[dups[0]].sum().item())
            value = -(advantage / n) / q_inf * data_parallel_size / (tokens * len(dups))
            for row in dups:
                coefficients[row][mask[row]] += value
            if positive:
                pos_mass += abs(value) * tokens * len(dups)
            else:
                neg_mass += abs(value) * tokens * len(dups)
    metrics = {
        "aux/planner_control/informative_question_count": float(q_inf),
        "aux/planner_control/planner_workflow_count": float(len(planner_rows)),
        "aux/planner_control/coefficient_l1": float(coefficients.abs().sum().item()),
        "aux/planner_control/coefficient_sum": float(coefficients.sum().item()),
        "aux/planner_control/positive_coefficient_mass": float(pos_mass),
        "aux/planner_control/negative_coefficient_mass": float(neg_mass),
    }
    return coefficients, metrics
