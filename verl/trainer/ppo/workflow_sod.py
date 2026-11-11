"""Step-wise on-policy distillation for row-wise DAGENT workflows."""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Optional

import numpy as np
import torch


@dataclass
class WorkflowSODOutput:
    """Result of adding the SOD token signal to an existing RL advantage."""

    advantages: torch.Tensor
    weights: torch.Tensor
    row_divergences: torch.Tensor
    metrics: dict[str, float]


def normalize_reference_question(question: object) -> str:
    """Normalize harmless whitespace without changing question semantics."""

    return " ".join(str(question or "").split())


def workflow_action_match_key(role: object, node_id: object, occurrence: int) -> str:
    """Return the stable key shared by rollout actions and offline references."""

    return (
        f"role={str(role).strip()}|node={str(node_id).strip()}|"
        f"occurrence={int(occurrence)}"
    )


def build_reference_action_records(
    sft_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Convert one accepted trajectory into deterministic action records."""

    records = []
    occurrences: dict[tuple[str, str], int] = defaultdict(int)
    for event in sft_events:
        if not isinstance(event, dict):
            raise ValueError("reference-bank sft_events must contain objects")
        role = str(event.get("agent_name", "")).strip()
        response = str(event.get("response", "")).strip()
        metadata = event.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("reference-bank event metadata must be an object")
        if not role or not response:
            raise ValueError("reference-bank events require agent_name and response")
        if role == "PlanningAgent":
            node_id = "planning"
        elif role == "AnswerSummarizationAgent":
            node_id = "summary"
        else:
            node_id = str(metadata.get("node_id", "")).strip()
            if not node_id:
                raise ValueError(
                    f"reference-bank {role} event is missing metadata.node_id"
                )
        occurrence_key = (role, node_id)
        occurrence = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        records.append(
            {
                "step": len(records),
                "role": role,
                "node_id": node_id,
                "alignment_key": workflow_action_match_key(
                    role, node_id, occurrence
                ),
                "response": response,
            }
        )
    if not records:
        raise ValueError("reference-bank trajectory has no sft_events")
    return records


def load_privileged_reference_bank(path: str | Path) -> dict[str, dict[str, object]]:
    """Load one immutable accepted workflow per normalized question."""

    bank_path = Path(path)
    if not bank_path.is_file():
        raise FileNotFoundError(f"workflow SOD reference bank not found: {bank_path}")
    bank: dict[str, dict[str, object]] = {}
    with bank_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid reference-bank JSON at line {line_number}"
                ) from exc
            question = normalize_reference_question(record.get("question"))
            if not question:
                raise ValueError(
                    f"reference-bank line {line_number} is missing question"
                )
            if question in bank:
                raise ValueError(
                    f"reference-bank contains duplicate question at line {line_number}"
                )
            action_records = build_reference_action_records(
                record.get("sft_events") or []
            )
            bank[question] = {
                "trajectory_id": str(
                    record.get("trajectory_id") or f"bank-line-{line_number}"
                ),
                "action_records": action_records,
                "aligned_references": build_prefix_aligned_references(
                    action_records
                ),
            }
    if not bank:
        raise ValueError("workflow SOD reference bank is empty")
    return bank


_ROLE_OPENING_TAGS = {
    "PlanningAgent": ("plan",),
    "QueryRewriteAgent": (),
    "ReasonerAgent": ("conclusion",),
    "AnswerSummarizationAgent": ("answer",),
    "AnswerGenerationAgent": ("conclusion", "rewrite"),
}


def build_opening_tag_excluded_mask(
    tokenizer,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    role: object,
) -> tuple[torch.Tensor, bool, int]:
    """Exclude only a recognized leading protocol tag from an action loss."""

    full_mask = response_mask.detach().cpu().bool()
    tags = _ROLE_OPENING_TAGS.get(str(role))
    if tags == ():
        return full_mask, False, 0
    if tags is None:
        return full_mask, True, 0

    valid_positions = torch.nonzero(full_mask, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return full_mask, True, 0
    valid_ids = response_ids.detach().cpu()[valid_positions].tolist()
    text = tokenizer.decode(
        valid_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    alternatives = "|".join(re.escape(tag) for tag in tags)
    match = re.match(rf"^\s*<(?:{alternatives})>", text, flags=re.IGNORECASE)
    if match is None:
        return full_mask, True, 0

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != valid_ids:
        return full_mask, True, 0

    selected = full_mask.clone()
    for local_index, (start, end) in enumerate(encoded["offset_mapping"]):
        if start < match.end() and end > match.start():
            selected[valid_positions[local_index]] = False
    removed = int((full_mask & ~selected).sum().item())
    if removed == 0 or not selected.any():
        return full_mask, True, 0
    return selected, False, removed


def required_invalid_workflow_reward_cap(*, sod_only: bool) -> float:
    """Prevent invalid workflows from becoming successful sibling Teachers."""

    return 0.0 if sod_only else 1.0


def should_score_workflow_sod_target(
    *, target_success: bool, workflow_valid: bool, sod_only: bool
) -> bool:
    """Keep malformed failures trainable when SOD is the only recovery signal."""

    if target_success:
        return False
    return bool(workflow_valid or sod_only)


def _safe_role_name(role: object) -> str:
    value = str(role or "unknown")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def choose_distinct_successful_sibling(
    successful_workflow_ids: list[object],
    target_workflow_id: object,
    rotation: int = 0,
) -> Optional[object]:
    """Choose a successful reference without exposing the target trajectory itself."""

    candidates = sorted(
        [
            workflow_id
            for workflow_id in successful_workflow_ids
            if workflow_id != target_workflow_id
        ],
        key=str,
    )
    if not candidates:
        return None
    return candidates[int(rotation) % len(candidates)]


def _render_action_prefix(action_records: list[dict[str, object]]) -> str:
    lines = ["<successful_sibling_prefix>"]
    for record in action_records:
        lines.extend(
            [
                '<agent role="{}" step="{}" node_id="{}">'.format(
                    record.get("role", ""),
                    record.get("step", ""),
                    record.get("node_id", ""),
                ),
                str(record.get("response", "")),
                "</agent>",
            ]
        )
    lines.append("</successful_sibling_prefix>")
    return "\n".join(lines)


def render_full_privileged_reference(
    action_records: list[dict[str, object]],
) -> str:
    """Render the complete accepted workflow used by the privileged Teacher."""

    if not action_records:
        raise ValueError("privileged workflow reference has no actions")
    return _render_action_prefix(action_records).replace(
        "successful_sibling_prefix", "privileged_successful_workflow"
    )


def build_prefix_aligned_references(
    action_records: list[dict[str, object]],
) -> dict[str, str]:
    """Map each action key to a successful prefix ending at that action."""

    references = {}
    prefix = []
    for expected_step, record in enumerate(action_records):
        step = int(record.get("step", -1))
        if step != expected_step:
            raise ValueError("successful sibling action steps must be contiguous")
        prefix.append(record)
        key = str(record.get("alignment_key", "")).strip()
        if not key:
            raise ValueError("successful sibling action is missing an alignment key")
        if key in references:
            raise ValueError(f"duplicate successful sibling alignment key: {key}")
        references[key] = _render_action_prefix(prefix)
    return references


def count_target_future_actions_in_reference_prefix(
    reference_action_records: list[dict[str, object]],
    target_action_records: list[dict[str, object]],
    target_step: int,
    alignment_key: str,
) -> int:
    """Count reference-prefix actions that occur later in the target workflow."""

    target_steps = {
        str(record.get("alignment_key", "")): int(record.get("step", -1))
        for record in target_action_records
        if str(record.get("alignment_key", "")).strip()
    }
    matched_reference_steps = [
        int(record.get("step", -1))
        for record in reference_action_records
        if str(record.get("alignment_key", "")) == str(alignment_key)
    ]
    if len(matched_reference_steps) != 1:
        raise ValueError(
            "aligned reference keys must identify exactly one reference action"
        )
    matched_reference_step = matched_reference_steps[0]
    return sum(
        target_steps.get(str(record.get("alignment_key", "")), -1)
        > int(target_step)
        for record in reference_action_records
        if int(record.get("step", -1)) <= matched_reference_step
    )


def render_causally_aligned_reference(
    reference_action_records: list[dict[str, object]],
    target_action_records: list[dict[str, object]],
    target_step: int,
    alignment_key: str,
) -> str:
    """Render a bank prefix without exposing actions later in the Student trace."""

    target_keys = {
        str(record.get("alignment_key", ""))
        for record in target_action_records
        if int(record.get("step", -1)) <= int(target_step)
    }
    matching_indices = [
        index
        for index, record in enumerate(reference_action_records)
        if str(record.get("alignment_key", "")) == str(alignment_key)
    ]
    if len(matching_indices) != 1:
        raise ValueError(
            "causal reference keys must identify exactly one reference action"
        )
    reference_limit = matching_indices[0]
    prefix = [
        record
        for index, record in enumerate(reference_action_records)
        if index <= reference_limit
        and str(record.get("alignment_key", "")) in target_keys
    ]
    if not prefix or str(prefix[-1].get("alignment_key", "")) != str(
        alignment_key
    ):
        raise ValueError("causal reference omitted its aligned current action")
    return _render_action_prefix(prefix)


def render_missing_action_reference(
    action_records: list[dict[str, object]],
    target_step: int,
    alignment_key: str,
    target_action_records: Optional[list[dict[str, object]]] = None,
) -> str:
    """Render the causally aligned prefix before an explicitly absent action."""

    if target_action_records is None:
        prefix = [
            record
            for record in action_records
            if int(record.get("step", -1)) < int(target_step)
        ]
    else:
        prior_target_keys = {
            str(record.get("alignment_key", ""))
            for record in target_action_records
            if int(record.get("step", -1)) < int(target_step)
        }
        prior_target_keys.discard("")
        prefix = [
            record
            for record in action_records
            if str(record.get("alignment_key", "")) in prior_target_keys
        ]
    text = _render_action_prefix(prefix)
    marker = (
        f'<aligned_action_absent key="{alignment_key}">'
        "The successful sibling has no corresponding action at this position. "
        "Reassess whether the original action should exist."
        "</aligned_action_absent>"
    )
    return text.replace(
        "</successful_sibling_prefix>",
        marker + "\n</successful_sibling_prefix>",
    )


def choose_fixed_aligned_sibling(
    successful_workflow_ids: list[object],
    target_workflow_id: object,
    target_alignment_keys: set[str],
    references_by_workflow: dict[object, dict[str, str]],
    rotation: int = 0,
) -> Optional[object]:
    """Choose one sibling for an entire target, preferring maximal key overlap."""

    overlaps = {
        workflow_id: len(
            target_alignment_keys.intersection(references_by_workflow[workflow_id])
        )
        for workflow_id in successful_workflow_ids
        if workflow_id != target_workflow_id
        and workflow_id in references_by_workflow
    }
    if not overlaps:
        return None
    best = max(overlaps.values())
    candidates = sorted(
        [workflow_id for workflow_id, overlap in overlaps.items() if overlap == best],
        key=str,
    )
    return candidates[int(rotation) % len(candidates)]


def apply_workflow_sod_advantage(
    *,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    distillation_mask: Optional[torch.Tensor] = None,
    trajectory_ids: np.ndarray,
    workflow_steps: np.ndarray,
    roles: Optional[np.ndarray] = None,
    workflow_tool_steps: Optional[np.ndarray] = None,
    coefficient: float = 1.0,
    epsilon: float = 1e-6,
    delta: float = 0.2,
    sod_only: bool = False,
) -> WorkflowSODOutput:
    """Add SOD's sampled reverse-KL signal to a workflow-level RL advantage.

    DAGENT stores each Agent action in a separate batch row. Rows are grouped by
    trajectory and ordered by ``workflow_step`` before computing the SOD weights.
    Randomly duplicated rows used for distributed padding are collapsed into the
    same logical action and receive the same weight.
    """

    tensors = {
        "advantages": advantages,
        "old_log_probs": old_log_probs,
        "teacher_log_probs": teacher_log_probs,
        "response_mask": response_mask,
    }
    expected_shape = old_log_probs.shape
    if len(expected_shape) != 2:
        raise ValueError(f"workflow SOD expects rank-2 log probabilities, got {expected_shape}")
    for name, tensor in tensors.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match "
                f"old_log_probs shape {tuple(expected_shape)}"
            )
    if distillation_mask is not None and distillation_mask.shape != expected_shape:
        raise ValueError(
            f"distillation_mask shape {tuple(distillation_mask.shape)} does not match "
            f"old_log_probs shape {tuple(expected_shape)}"
        )
    if coefficient < 0:
        raise ValueError("workflow SOD coefficient must be non-negative")
    if epsilon <= 0:
        raise ValueError("workflow SOD epsilon must be positive")
    if delta < 0:
        raise ValueError("workflow SOD delta must be non-negative")

    row_count = expected_shape[0]
    trajectory_ids = np.asarray(trajectory_ids, dtype=object)
    workflow_steps = np.asarray(workflow_steps, dtype=np.int64)
    if roles is None:
        roles = np.full(row_count, "unknown", dtype=object)
    else:
        roles = np.asarray(roles, dtype=object)
    for name, values in {
        "trajectory_ids": trajectory_ids,
        "workflow_steps": workflow_steps,
        "roles": roles,
    }.items():
        if values.shape[0] != row_count:
            raise ValueError(f"{name} count does not match workflow SOD batch size")

    response_mask_bool = response_mask.to(dtype=torch.bool)
    mask = response_mask_bool
    if distillation_mask is not None:
        mask = mask & distillation_mask.to(device=mask.device, dtype=torch.bool)
    valid_counts = mask.sum(dim=-1)
    finite_tokens = torch.isfinite(old_log_probs) & torch.isfinite(teacher_log_probs)
    if not torch.all(finite_tokens | ~mask):
        raise ValueError("workflow SOD log probabilities contain non-finite response tokens")

    abs_gap = (old_log_probs.detach() - teacher_log_probs.detach()).abs()
    row_divergences = torch.zeros(row_count, dtype=torch.float32, device=old_log_probs.device)
    nonempty_rows = valid_counts > 0
    if nonempty_rows.any():
        row_divergences[nonempty_rows] = (
            (abs_gap * mask).sum(dim=-1)[nonempty_rows]
            / valid_counts[nonempty_rows].to(dtype=abs_gap.dtype)
        ).float()

    action_rows: dict[tuple[object, int], list[int]] = defaultdict(list)
    action_roles: dict[tuple[object, int], object] = {}
    trajectory_actions: dict[object, list[tuple[object, int]]] = defaultdict(list)
    for row_idx, (trajectory_id, step, role) in enumerate(
        zip(trajectory_ids, workflow_steps, roles, strict=True)
    ):
        if not bool(nonempty_rows[row_idx].item()):
            continue
        key = (trajectory_id, int(step))
        existing_role = action_roles.setdefault(key, role)
        if existing_role != role:
            raise ValueError(
                f"trajectory {trajectory_id!r} step {step} has inconsistent roles: "
                f"{existing_role!r} and {role!r}"
            )
        action_rows[key].append(row_idx)

    for key in action_rows:
        trajectory_actions[key[0]].append(key)
    for keys in trajectory_actions.values():
        keys.sort(key=lambda item: item[1])

    if workflow_tool_steps is None:
        workflow_tool_steps = np.zeros(row_count, dtype=np.int64)
    else:
        workflow_tool_steps = np.asarray(workflow_tool_steps, dtype=np.int64)
        if workflow_tool_steps.shape[0] != row_count:
            raise ValueError("workflow_tool_steps count does not match workflow SOD batch size")

    weights = torch.zeros_like(response_mask, dtype=torch.float32)
    action_d_values: list[float] = []
    action_w_values: list[float] = []
    first_d_values: list[float] = []
    last_d_values: list[float] = []
    divergence_increase: list[float] = []
    recovery_after_increase: list[float] = []
    duplicate_action_count = 0
    multi_action_same_role_stream_count = 0
    role_d_values: dict[str, list[float]] = defaultdict(list)
    role_w_values: dict[str, list[float]] = defaultdict(list)

    row_divergences_cpu = row_divergences.detach().cpu()
    for keys in trajectory_actions.values():
        d_values = [
            float(row_divergences_cpu[action_rows[key]].mean().item())
            for key in keys
        ]
        if not d_values:
            continue
        role_counts_in_trajectory: dict[object, int] = defaultdict(int)
        for key in keys:
            role_counts_in_trajectory[action_roles[key]] += 1
        multi_action_same_role_stream_count += sum(
            int(count > 1) for count in role_counts_in_trajectory.values()
        )

        w_values = [1.0]
        cumulative_ratio = 1.0
        for previous, current in zip(d_values[:-1], d_values[1:], strict=True):
            cumulative_ratio *= (previous + epsilon) / (current + epsilon)
            w_values.append(min(cumulative_ratio, 1.0 + delta))

        first_d_values.append(d_values[0])
        last_d_values.append(d_values[-1])
        divergence_increase.append(float(d_values[-1] > d_values[0] + epsilon))
        saw_increase = False
        recovered = False
        for previous, current in zip(d_values[:-1], d_values[1:], strict=True):
            if current > previous + epsilon:
                saw_increase = True
            elif saw_increase and current < previous - epsilon:
                recovered = True
        recovery_after_increase.append(float(recovered))

        for key, d_value, weight in zip(keys, d_values, w_values, strict=True):
            rows = action_rows[key]
            duplicate_action_count += max(0, len(rows) - 1)
            role_name = _safe_role_name(action_roles[key])
            action_d_values.append(d_value)
            action_w_values.append(weight)
            role_d_values[role_name].append(d_value)
            role_w_values[role_name].append(weight)
            for row_idx in rows:
                weights[row_idx] = mask[row_idx].to(dtype=torch.float32) * weight

    local_advantage = (
        teacher_log_probs.detach() - old_log_probs.detach()
    ).to(dtype=advantages.dtype) * mask
    added_advantage = coefficient * weights.to(dtype=advantages.dtype) * local_advantage
    task_advantages = torch.zeros_like(advantages) if sod_only else advantages
    updated_advantages = task_advantages + added_advantage

    token_mask = mask & (weights > 0)
    selected_trajectory_ids = {
        trajectory_ids[row_idx]
        for row_idx in range(row_count)
        if bool(token_mask[row_idx].any().item())
    }
    all_trajectory_ids = {
        trajectory_ids[row_idx]
        for row_idx in range(row_count)
        if bool(response_mask_bool[row_idx].any().item())
    }
    selected_multi_tool_trajectories = {
        trajectory_ids[row_idx]
        for row_idx in range(row_count)
        if trajectory_ids[row_idx] in selected_trajectory_ids
        and int(workflow_tool_steps[row_idx]) > 1
    }
    original_task_rms = torch.sqrt(
        ((advantages.detach().float() ** 2) * response_mask_bool).sum()
        / response_mask_bool.sum().clamp_min(1)
    )
    task_rms = torch.sqrt(
        ((task_advantages.detach().float() ** 2) * response_mask_bool).sum()
        / response_mask_bool.sum().clamp_min(1)
    )
    added_rms = torch.sqrt(
        ((added_advantage.detach().float() ** 2) * response_mask_bool).sum()
        / response_mask_bool.sum().clamp_min(1)
    )
    selected_task_rms = torch.sqrt(
        ((task_advantages.detach().float() ** 2) * token_mask).sum()
        / token_mask.sum().clamp_min(1)
    )
    valid_local = local_advantage.detach()[token_mask]
    valid_weights = weights.detach()[token_mask]

    metrics = {
        "aux/workflow_sod/enabled": 1.0,
        "aux/workflow_sod/sod_only": float(sod_only),
        "aux/workflow_sod/coefficient": float(coefficient),
        "aux/workflow_sod/trajectory_count": float(len(selected_trajectory_ids)),
        "aux/workflow_sod/multi_step_trajectory_count": float(
            len(selected_multi_tool_trajectories)
        ),
        "aux/workflow_sod/stepwise_stream_count": float(len(trajectory_actions)),
        "aux/workflow_sod/same_role_stream_count": float(
            sum(
                len({action_roles[key] for key in keys})
                for keys in trajectory_actions.values()
            )
        ),
        "aux/workflow_sod/multi_action_same_role_stream_count": float(
            multi_action_same_role_stream_count
        ),
        "aux/workflow_sod/action_count": float(len(action_d_values)),
        "aux/workflow_sod/duplicate_padding_rows": float(duplicate_action_count),
        "aux/workflow_sod/response_token_count": float(token_mask.sum().item()),
        "aux/workflow_sod/full_batch_response_token_count": float(
            response_mask_bool.sum().item()
        ),
        "aux/workflow_sod/distillation_token_coverage": float(
            token_mask.sum().item() / response_mask_bool.sum().clamp_min(1).item()
        ),
        "aux/workflow_sod/distillation_workflow_coverage": float(
            len(selected_trajectory_ids) / max(1, len(all_trajectory_ids))
        ),
        # Unselected tokens retain zero advantage while the actor loss keeps its
        # full-workflow denominator, so sparse pairs cannot be renormalized to
        # full-batch strength.
        "aux/workflow_sod/full_batch_coverage_scaling": 1.0,
        "aux/workflow_sod/divergence_mean": float(np.mean(action_d_values))
        if action_d_values
        else 0.0,
        "aux/workflow_sod/divergence_first_mean": float(np.mean(first_d_values))
        if first_d_values
        else 0.0,
        "aux/workflow_sod/divergence_last_mean": float(np.mean(last_d_values))
        if last_d_values
        else 0.0,
        "aux/workflow_sod/weight_mean": float(np.mean(action_w_values))
        if action_w_values
        else 0.0,
        "aux/workflow_sod/weight_min": float(np.min(action_w_values))
        if action_w_values
        else 0.0,
        "aux/workflow_sod/weight_max": float(np.max(action_w_values))
        if action_w_values
        else 0.0,
        "aux/workflow_sod/attenuated_action_fraction": float(
            np.mean(np.asarray(action_w_values) < 1.0 - epsilon)
        )
        if action_w_values
        else 0.0,
        "aux/workflow_sod/amplified_action_fraction": float(
            np.mean(np.asarray(action_w_values) > 1.0 + epsilon)
        )
        if action_w_values
        else 0.0,
        "aux/workflow_sod/divergence_increase_trajectory_fraction": float(
            np.mean(divergence_increase)
        )
        if divergence_increase
        else 0.0,
        "aux/workflow_sod/recovery_after_increase_trajectory_fraction": float(
            np.mean(recovery_after_increase)
        )
        if recovery_after_increase
        else 0.0,
        "aux/workflow_sod/local_advantage_mean": float(valid_local.float().mean().item())
        if valid_local.numel()
        else 0.0,
        "aux/workflow_sod/local_advantage_positive_fraction": float(
            (valid_local > 0).float().mean().item()
        )
        if valid_local.numel()
        else 0.0,
        "aux/workflow_sod/token_weight_mean": float(valid_weights.float().mean().item())
        if valid_weights.numel()
        else 0.0,
        "aux/workflow_sod/task_advantage_rms": float(task_rms.item()),
        "aux/workflow_sod/original_rl_advantage_rms": float(original_task_rms.item()),
        "aux/workflow_sod/selected_task_advantage_rms": float(selected_task_rms.item()),
        "aux/workflow_sod/added_advantage_rms": float(added_rms.item()),
        "aux/workflow_sod/added_to_task_rms_ratio": 0.0
        if sod_only
        else float((added_rms / task_rms.clamp_min(epsilon)).item()),
        "aux/workflow_sod/added_to_selected_task_rms_ratio": 0.0
        if sod_only
        else float((added_rms / selected_task_rms.clamp_min(epsilon)).item()),
    }
    for role_name, values in role_d_values.items():
        metrics[f"aux/workflow_sod/divergence_mean/{role_name}"] = float(np.mean(values))
        metrics[f"aux/workflow_sod/weight_mean/{role_name}"] = float(
            np.mean(role_w_values[role_name])
        )

    return WorkflowSODOutput(
        advantages=updated_advantages,
        weights=weights,
        row_divergences=row_divergences,
        metrics=metrics,
    )
