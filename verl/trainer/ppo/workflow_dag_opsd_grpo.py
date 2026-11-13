"""Outcome-only workflow GRPO with sign-preserving DAG-aware OPSD shaping."""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch


_COMPLETE_WORKFLOW_ESTIMATORS = frozenset({"workflow_grpo", "workflow_rloo"})


def uses_complete_workflow_actor_batches(advantage_estimator: object) -> bool:
    """Return whether actor batching must preserve complete workflow groups."""

    value = getattr(advantage_estimator, "value", advantage_estimator)
    return str(value) in _COMPLETE_WORKFLOW_ESTIMATORS


def compute_actor_data_parallel_size(
    world_size: int, ulysses_sequence_parallel_size: int = 1
) -> int:
    """Return the number of distinct actor data shards used by dispatch."""

    if world_size < 1:
        raise ValueError("actor world_size must be positive")
    if ulysses_sequence_parallel_size < 1:
        raise ValueError("ulysses_sequence_parallel_size must be positive")
    if world_size % ulysses_sequence_parallel_size != 0:
        raise ValueError(
            "actor world_size must be divisible by ulysses_sequence_parallel_size"
        )
    return world_size // ulysses_sequence_parallel_size


def build_complete_workflow_minibatch_indices(
    trajectory_ids: np.ndarray,
    workflows_per_minibatch: int,
) -> list[np.ndarray]:
    """Group every row of a workflow into the same actor minibatch."""

    ids = np.asarray(trajectory_ids, dtype=object)
    if ids.ndim != 1 or ids.size < 1:
        raise ValueError("workflow trajectory ids must be a non-empty rank-1 array")
    if workflows_per_minibatch < 1:
        raise ValueError("workflow PPO minibatch size must be positive")

    ordered_ids = list(dict.fromkeys(ids.tolist()))
    if len(ordered_ids) % workflows_per_minibatch != 0:
        raise ValueError(
            f"{len(ordered_ids)} workflows are not divisible by PPO minibatch size "
            f"{workflows_per_minibatch}"
        )

    batches = []
    for start in range(0, len(ordered_ids), workflows_per_minibatch):
        selected = set(ordered_ids[start : start + workflows_per_minibatch])
        row_indices = np.asarray(
            [row for row, trajectory_id in enumerate(ids) if trajectory_id in selected],
            dtype=np.int64,
        )
        if row_indices.size < workflows_per_minibatch:
            raise ValueError("workflow minibatch lost one or more complete trajectories")
        batches.append(row_indices)
    return batches


def _group_question_rows(
    question_ids: np.ndarray,
    trajectory_ids: np.ndarray,
) -> list[np.ndarray]:
    """Return complete question groups after validating trajectory ownership."""

    questions = np.asarray(question_ids, dtype=object)
    trajectories = np.asarray(trajectory_ids, dtype=object)
    if questions.ndim != 1 or questions.size < 1:
        raise ValueError("question ids must be a non-empty rank-1 array")
    if trajectories.shape != questions.shape:
        raise ValueError("trajectory ids must match question ids")

    trajectory_question: dict[object, object] = {}
    grouped_rows: dict[object, list[int]] = {}
    for row, (question_id, trajectory_id) in enumerate(
        zip(questions, trajectories, strict=True)
    ):
        previous = trajectory_question.setdefault(trajectory_id, question_id)
        if previous != question_id:
            raise ValueError(
                f"trajectory {trajectory_id!r} belongs to multiple questions"
            )
        grouped_rows.setdefault(question_id, []).append(row)
    return [np.asarray(rows, dtype=np.int64) for rows in grouped_rows.values()]


def _pack_groups_into_fixed_bins(
    groups: list[np.ndarray],
    bin_count: int,
) -> list[np.ndarray]:
    """Balance indivisible row groups across a fixed number of non-empty bins."""

    if bin_count < 1:
        raise ValueError("bin count must be positive")
    if len(groups) < bin_count:
        raise ValueError(
            f"cannot place {len(groups)} question groups into {bin_count} non-empty bins"
        )

    bins: list[list[np.ndarray]] = [[] for _ in range(bin_count)]
    loads = [0] * bin_count
    ordered = sorted(
        enumerate(groups),
        key=lambda item: (-item[1].size, int(item[1][0]), item[0]),
    )
    for _, group in ordered:
        target = min(range(bin_count), key=lambda index: (loads[index], index))
        bins[target].append(group)
        loads[target] += int(group.size)

    return [
        np.sort(np.concatenate(items)).astype(np.int64, copy=False)
        for items in bins
    ]


def build_question_preserving_minibatch_indices(
    question_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    target_action_rows: int,
) -> list[np.ndarray]:
    """Pack complete sibling-question groups into action-row PPO minibatches.

    The number of bins follows ordinary row slicing, ``ceil(rows / target)``.
    Groups are then load-balanced across exactly that many bins so every actor
    data-parallel rank executes the same number of collectives.
    """

    if target_action_rows < 1:
        raise ValueError("target action rows must be positive")
    groups = _group_question_rows(question_ids, trajectory_ids)
    row_count = int(sum(group.size for group in groups))
    minibatch_count = int(np.ceil(row_count / target_action_rows))
    return _pack_groups_into_fixed_bins(groups, minibatch_count)


def build_complete_question_dp_partitions(
    question_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    data_parallel_size: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Pack complete sibling-question groups into equal-row DP shards.

    Shorter shards are padded by duplicating one local row. Callers must mask
    rows after each returned real-row count out of actor losses.
    """

    if data_parallel_size < 1:
        raise ValueError("actor data parallel size must be positive")
    groups = _group_question_rows(question_ids, trajectory_ids)
    partitions = _pack_groups_into_fixed_bins(groups, data_parallel_size)
    real_row_counts = [int(partition.size) for partition in partitions]
    max_rows = max(real_row_counts)
    padded = []
    for partition in partitions:
        padding = np.resize(partition[:1], max_rows - partition.size)
        padded.append(np.concatenate([partition, padding]))
    return padded, real_row_counts


def build_complete_workflow_dp_partitions(
    trajectory_ids: np.ndarray,
    data_parallel_size: int,
) -> list[np.ndarray]:
    """Pack complete workflows into equal-row DP shards using duplicate padding."""

    ids = np.asarray(trajectory_ids, dtype=object)
    if ids.ndim != 1 or ids.size < 1:
        raise ValueError("workflow trajectory ids must be a non-empty rank-1 array")
    if data_parallel_size < 1:
        raise ValueError("actor data parallel size must be positive")
    ordered_ids = list(dict.fromkeys(ids.tolist()))
    if len(ordered_ids) % data_parallel_size != 0:
        raise ValueError(
            f"{len(ordered_ids)} workflows are not divisible by actor DP size "
            f"{data_parallel_size}"
        )

    workflows_per_rank = len(ordered_ids) // data_parallel_size
    partitions = []
    for rank in range(data_parallel_size):
        selected = set(
            ordered_ids[
                rank * workflows_per_rank : (rank + 1) * workflows_per_rank
            ]
        )
        partitions.append(
            np.asarray(
                [row for row, trajectory_id in enumerate(ids) if trajectory_id in selected],
                dtype=np.int64,
            )
        )

    max_rows = max(partition.size for partition in partitions)
    padded = []
    for partition in partitions:
        if partition.size < 1:
            raise ValueError("actor DP workflow partition is empty")
        padding = np.resize(partition[:1], max_rows - partition.size)
        padded.append(np.concatenate([partition, padding]))
    return padded


def compute_workflow_token_loss_weights(
    response_mask: torch.Tensor,
    trajectory_ids: np.ndarray,
    workflow_steps: np.ndarray,
    *,
    data_parallel_size: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalize policy-gradient tokens so every complete workflow has equal mass."""

    if response_mask.ndim != 2:
        raise ValueError("workflow loss normalization expects a rank-2 response mask")
    if data_parallel_size < 1:
        raise ValueError("data_parallel_size must be positive")
    trajectory_ids = np.asarray(trajectory_ids, dtype=object)
    workflow_steps = np.asarray(workflow_steps, dtype=np.int64)
    if trajectory_ids.shape[0] != response_mask.shape[0]:
        raise ValueError("trajectory id count does not match response mask rows")
    if workflow_steps.shape[0] != response_mask.shape[0]:
        raise ValueError("workflow step count does not match response mask rows")

    workflow_actions: dict[object, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row_idx, (trajectory_id, workflow_step) in enumerate(
        zip(trajectory_ids, workflow_steps, strict=True)
    ):
        workflow_actions[trajectory_id][int(workflow_step)].append(row_idx)
    if not workflow_actions:
        raise ValueError("workflow loss normalization received no workflows")

    mask = response_mask.to(dtype=torch.bool)
    weights = torch.zeros_like(response_mask, dtype=torch.float32)
    workflow_count = len(workflow_actions)
    token_counts: list[int] = []
    logical_action_count = 0
    duplicate_row_count = 0
    workflow_mass = float(data_parallel_size) / workflow_count
    for actions in workflow_actions.values():
        action_token_counts: dict[int, int] = {}
        for workflow_step, rows in actions.items():
            row_token_counts = [int(mask[row].sum().item()) for row in rows]
            if not row_token_counts or row_token_counts[0] < 1:
                raise ValueError("workflow loss normalization found an empty action")
            if any(count != row_token_counts[0] for count in row_token_counts[1:]):
                raise ValueError(
                    f"duplicate workflow step {workflow_step} has inconsistent token counts"
                )
            action_token_counts[workflow_step] = row_token_counts[0]
            logical_action_count += 1
            duplicate_row_count += len(rows) - 1

        token_count = sum(action_token_counts.values())
        if token_count < 1:
            raise ValueError("workflow loss normalization found an empty workflow")
        token_counts.append(token_count)
        per_token_weight = workflow_mass / token_count
        for workflow_step, rows in actions.items():
            duplicate_scale = 1.0 / len(rows)
            for row in rows:
                weights[row] = (
                    mask[row].to(dtype=torch.float32)
                    * per_token_weight
                    * duplicate_scale
                )

    metrics = {
        "actor/workflow_loss_normalization_enabled": 1.0,
        "actor/workflow_loss_data_parallel_size": float(data_parallel_size),
        "actor/workflow_loss_workflow_count": float(workflow_count),
        "actor/workflow_loss_logical_action_count": float(logical_action_count),
        "actor/workflow_loss_duplicate_row_count": float(duplicate_row_count),
        "actor/workflow_loss_total_mass": float(weights.sum().item()),
        "actor/workflow_loss_mass_per_workflow": float(workflow_mass),
        "actor/workflow_loss_tokens_min": float(min(token_counts)),
        "actor/workflow_loss_tokens_max": float(max(token_counts)),
        "actor/workflow_loss_tokens_mean": float(np.mean(token_counts)),
    }
    return weights, metrics


def _config_value(config: Optional[Any], key: str, default: Any) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _workflow_grpo_advantages(
    question_ids: np.ndarray,
    workflow_ids: np.ndarray,
    outcomes: np.ndarray,
    normalize_by_std: bool,
    epsilon: float,
) -> np.ndarray:
    """Compute one GRPO advantage per workflow and broadcast it to role rows."""

    workflow_rows: dict[object, list[int]] = defaultdict(list)
    workflow_question: dict[object, object] = {}
    for row_idx, (question_id, workflow_id) in enumerate(
        zip(question_ids, workflow_ids, strict=True)
    ):
        previous = workflow_question.setdefault(workflow_id, question_id)
        if previous != question_id:
            raise ValueError(f"trajectory {workflow_id!r} is assigned to multiple questions")
        workflow_rows[workflow_id].append(row_idx)

    workflow_outcomes: dict[object, float] = {}
    question_workflows: dict[object, list[object]] = defaultdict(list)
    for workflow_id, rows in workflow_rows.items():
        values = outcomes[rows]
        if not np.allclose(values, values[0], atol=1e-6, rtol=0.0):
            raise ValueError(
                f"trajectory {workflow_id!r} has inconsistent outcome rewards across roles"
            )
        workflow_outcomes[workflow_id] = float(values[0])
        question_workflows[workflow_question[workflow_id]].append(workflow_id)

    row_advantages = np.zeros(len(question_ids), dtype=np.float64)
    for workflows in question_workflows.values():
        if len(workflows) < 2:
            continue
        values = np.asarray([workflow_outcomes[item] for item in workflows], dtype=np.float64)
        centered = values - float(values.mean())
        if normalize_by_std:
            std = float(values.std(ddof=1))
            centered = centered / (std + epsilon) if std > epsilon else np.zeros_like(centered)
        for position, workflow_id in enumerate(workflows):
            row_advantages[workflow_rows[workflow_id]] = centered[position]
    return row_advantages


def compute_workflow_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_ids: np.ndarray,
    outcome_rewards: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[Any] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pure outcome GRPO over complete workflows, not role rows."""
    del token_level_rewards, kwargs
    question_ids = np.asarray(index, dtype=object)
    workflow_ids = np.asarray(trajectory_ids, dtype=object)
    outcomes = np.asarray(outcome_rewards, dtype=np.float64)
    row_count = response_mask.shape[0]
    for name, values in {
        "question ids": question_ids,
        "trajectory ids": workflow_ids,
        "outcome rewards": outcomes,
    }.items():
        if values.shape[0] != row_count:
            raise ValueError(f"{name} count does not match reward batch size")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("outcome rewards contain a non-finite value")
    row_advantages = _workflow_grpo_advantages(
        question_ids,
        workflow_ids,
        outcomes,
        bool(_config_value(config, "norm_adv_by_std_in_grpo", True)),
        epsilon,
    )
    advantages = torch.as_tensor(
        row_advantages, dtype=torch.float32, device=response_mask.device
    ).unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_workflow_dag_opsd_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_ids: np.ndarray,
    outcome_rewards: np.ndarray,
    outcome_exact: np.ndarray,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    opsd_token_mask: torch.Tensor,
    roles: np.ndarray,
    workflow_steps: np.ndarray,
    peer_available: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[Any] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape outcome-only workflow GRPO with successful-peer hindsight.

    The environment reward remains exact match. The teacher/student probability
    gap only rescales the magnitude of the original workflow advantage on
    selected DAG action tokens, so it can never create or flip a reward signal.
    """

    del token_level_rewards, outcome_exact, kwargs
    question_ids = np.asarray(index, dtype=object)
    workflow_ids = np.asarray(trajectory_ids, dtype=object)
    outcomes = np.asarray(outcome_rewards, dtype=np.float64)
    row_roles = np.asarray(roles, dtype=object)
    row_steps = np.asarray(workflow_steps, dtype=np.int64)
    has_peer = np.asarray(peer_available, dtype=np.bool_)
    row_count = response_mask.shape[0]

    for name, values in {
        "question ids": question_ids,
        "trajectory ids": workflow_ids,
        "outcome rewards": outcomes,
        "roles": row_roles,
        "workflow steps": row_steps,
        "peer availability": has_peer,
    }.items():
        if values.shape[0] != row_count:
            raise ValueError(f"{name} count does not match reward batch size")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("outcome rewards contain a non-finite value")
    for name, tensor in {
        "student log probabilities": student_log_probs,
        "teacher log probabilities": teacher_log_probs,
        "OPSD token mask": opsd_token_mask,
    }.items():
        if tensor.shape != response_mask.shape:
            raise ValueError(f"{name} shape must match response_mask")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains a non-finite value")

    dag_config = _config_value(config, "dag_opsd", {}) or {}
    mix_strength = float(_config_value(dag_config, "mix_strength", 0.2))
    alpha_clip = float(_config_value(dag_config, "alpha_clip", 0.05))
    gap_temperature = float(_config_value(dag_config, "gap_temperature", 1.0))
    role_normalize = bool(_config_value(dag_config, "role_normalize", True))
    max_role_scale = float(_config_value(dag_config, "max_role_scale", 4.0))
    if not 0.0 <= mix_strength <= 1.0:
        raise ValueError("algorithm.dag_opsd.mix_strength must be in [0, 1]")
    if not 0.0 <= alpha_clip < 1.0:
        raise ValueError("algorithm.dag_opsd.alpha_clip must be in [0, 1)")
    if gap_temperature <= 0.0:
        raise ValueError("algorithm.dag_opsd.gap_temperature must be positive")
    if max_role_scale < 1.0:
        raise ValueError("algorithm.dag_opsd.max_role_scale must be at least 1")

    row_advantages = _workflow_grpo_advantages(
        question_ids,
        workflow_ids,
        outcomes,
        bool(_config_value(config, "norm_adv_by_std_in_grpo", True)),
        epsilon,
    )
    base = torch.as_tensor(
        row_advantages, dtype=student_log_probs.dtype, device=response_mask.device
    ).unsqueeze(-1)
    selected = (
        response_mask.bool()
        & opsd_token_mask.to(device=response_mask.device).bool()
        & torch.as_tensor(has_peer, device=response_mask.device).unsqueeze(-1)
        & (base != 0)
    )

    delta = teacher_log_probs.to(response_mask.device) - student_log_probs.to(response_mask.device)
    signed_gap = torch.sign(base) * delta / gap_temperature
    raw_weight = 2.0 * torch.sigmoid(signed_gap)
    weight = raw_weight.clamp(1.0 - alpha_clip, 1.0 + alpha_clip)
    deviation = mix_strength * (weight - 1.0)

    if role_normalize and selected.any():
        role_scale = torch.ones_like(deviation)
        for workflow_id in dict.fromkeys(workflow_ids.tolist()):
            rows = np.flatnonzero(workflow_ids == workflow_id).tolist()
            counts: dict[object, int] = defaultdict(int)
            seen_steps = set()
            for row in rows:
                step_key = (row_roles[row], int(row_steps[row]))
                if step_key in seen_steps:
                    continue
                seen_steps.add(step_key)
                counts[row_roles[row]] += int(selected[row].sum().item())
            active_counts = [count for count in counts.values() if count > 0]
            if not active_counts:
                continue
            target = float(np.mean(active_counts))
            for row in rows:
                count = counts[row_roles[row]]
                if count <= 0:
                    continue
                scale = min(max(target / count, 1.0 / max_role_scale), max_role_scale)
                role_scale[row] = scale
        deviation = deviation * role_scale

    max_deviation = mix_strength * alpha_clip
    deviation = deviation.clamp(-max_deviation, max_deviation)
    multiplier = torch.where(selected, 1.0 + deviation, torch.ones_like(deviation))
    advantages = base * multiplier * response_mask.to(dtype=base.dtype)
    return advantages, advantages
