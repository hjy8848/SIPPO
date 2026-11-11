#!/usr/bin/env python3
"""Validate DAG-CCW joins and invariants on a recorded four-rollout batch."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

from scripts.audit_dagent_ccw_paired_suffixes import (
    attach_workflow_rloo_advantages,
    load_trajectories,
)
try:
    from verl.trainer.ppo.workflow_ccw import apply_workflow_ccw_weights
except ModuleNotFoundError as exc:
    if exc.name != "ray":
        raise
    module_path = Path(__file__).parents[1] / "verl" / "trainer" / "ppo" / "workflow_ccw.py"
    spec = importlib.util.spec_from_file_location("workflow_ccw_standalone", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load standalone CCW module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    apply_workflow_ccw_weights = module.apply_workflow_ccw_weights


def load_replays(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in rows:
            raise ValueError(f"duplicate paired replay trajectory: {trajectory_id}")
        rows[trajectory_id] = row
    return rows


def apply_by_complete_workflow_partition(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    credits: torch.Tensor,
    trajectory_ids: Sequence[str],
    world_size: int,
) -> tuple[torch.Tensor, list[int]]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    workflow_order = sorted(set(trajectory_ids))
    workflow_rank = {
        workflow_id: index % world_size
        for index, workflow_id in enumerate(workflow_order)
    }
    combined = torch.empty_like(advantages)
    rank_token_counts = []
    for rank in range(world_size):
        indices = [
            index
            for index, workflow_id in enumerate(trajectory_ids)
            if workflow_rank[workflow_id] == rank
        ]
        if not indices:
            rank_token_counts.append(0)
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long)
        output = apply_workflow_ccw_weights(
            advantages[index_tensor],
            response_mask[index_tensor],
            credits[index_tensor],
        )
        combined[index_tensor] = output.advantages
        rank_token_counts.append(int(response_mask[index_tensor].sum().item()))
    return combined, rank_token_counts


def validate(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    trajectories = load_trajectories(
        args.trajectory_files,
        datasets,
        args.max_trajectories_per_dataset,
        include_empty_nodes=True,
    )
    trajectories = attach_workflow_rloo_advantages(
        trajectories,
        args.expected_rollouts_per_question,
    )
    replays = load_replays(args.paired_replays)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )

    action_rows = []
    missing_trajectories = []
    missing_actions = []
    for trajectory in trajectories:
        replay = replays.get(trajectory.trajectory_id)
        if replay is None:
            missing_trajectories.append(trajectory.trajectory_id)
            continue
        replay_advantage = replay.get("workflow_rloo_advantage")
        if replay_advantage is None or not math.isclose(
            float(replay_advantage),
            float(trajectory.rloo_advantage),
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"RLOO mismatch for {trajectory.trajectory_id}: "
                f"trajectory={trajectory.rloo_advantage}, replay={replay_advantage}"
            )
        cases = {str(case["node_id"]): case for case in replay["cases"]}
        for node in trajectory.nodes:
            node_id = str(node["id"])
            case = cases.get(node_id)
            if case is None:
                missing_actions.append(f"{trajectory.trajectory_id}:node:{node_id}")
                continue
            response_text = str(node.get("raw_output") or node.get("conclusion") or "").strip()
            token_ids = tokenizer.encode(response_text, add_special_tokens=False)
            if not token_ids:
                raise ValueError(
                    f"eligible action has no response tokens: {trajectory.trajectory_id}:node:{node_id}"
                )
            action_rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "question_group_id": trajectory.question_group_id,
                    "rollout_index": trajectory.rollout_index,
                    "node_id": node_id,
                    "node_type": str(node["type"]),
                    "rloo_advantage": float(trajectory.rloo_advantage),
                    "signed_credit": float(case["signed_credit"]),
                    "token_count": len(token_ids),
                }
            )

    expected_actions = sum(len(trajectory.nodes) for trajectory in trajectories)
    maximum_tokens = max((row["token_count"] for row in action_rows), default=0)
    advantages = torch.zeros((len(action_rows), maximum_tokens), dtype=torch.float32)
    response_mask = torch.zeros_like(advantages, dtype=torch.bool)
    credits = torch.tensor(
        [row["signed_credit"] for row in action_rows],
        dtype=torch.float32,
        requires_grad=True,
    )
    for index, row in enumerate(action_rows):
        token_count = row["token_count"]
        response_mask[index, :token_count] = True
        advantages[index, :token_count] = row["rloo_advantage"]

    output = apply_workflow_ccw_weights(advantages, response_mask, credits)
    partitioned, rank_token_counts = apply_by_complete_workflow_partition(
        advantages,
        response_mask,
        credits,
        [row["trajectory_id"] for row in action_rows],
        args.dp_world_size,
    )
    group_sizes = Counter(trajectory.question_group_id for trajectory in trajectories)
    rloo_by_group: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        rloo_by_group[trajectory.question_group_id].append(
            float(trajectory.rloo_advantage)
        )

    hard_gates = {
        "complete_four_rollout_groups": bool(group_sizes)
        and all(size == args.expected_rollouts_per_question for size in group_sizes.values()),
        "rloo_group_sum_zero": all(
            math.isclose(sum(values), 0.0, abs_tol=1e-7)
            for values in rloo_by_group.values()
        ),
        "trajectory_join_100_percent": not missing_trajectories,
        "action_join_100_percent": len(action_rows) == expected_actions and not missing_actions,
        "weights_detached": not output.row_weights.requires_grad,
        "weights_positive_bounded": bool(
            output.row_weights.gt(0).all()
            and output.row_weights.ge(0.5).all()
            and output.row_weights.le(1.5).all()
        ),
        "token_advantage_sign_preserved": output.metrics[
            "aux/workflow_ccw/sign_flip_count"
        ]
        == 0,
        "response_token_mass_unchanged": output.metrics[
            "aux/workflow_ccw/token_mass_unchanged"
        ]
        == 1,
        "dp_partition_consistent": torch.equal(output.advantages, partitioned),
        "no_standalone_credit_gradient": not output.token_weights.requires_grad,
    }
    report = {
        "schema_version": 1,
        "method": "dag_ccw_recorded_one_batch_invariant_check",
        "questions": len(group_sizes),
        "workflows": len(trajectories),
        "datasets": dict(Counter(trajectory.dataset for trajectory in trajectories)),
        "workflow_em": {
            "mean": sum(trajectory.logged_em for trajectory in trajectories)
            / max(1, len(trajectories)),
            "nonzero_rloo_workflows": sum(
                not math.isclose(float(trajectory.rloo_advantage), 0.0, abs_tol=1e-9)
                for trajectory in trajectories
            ),
            "zero_rloo_workflows": sum(
                math.isclose(float(trajectory.rloo_advantage), 0.0, abs_tol=1e-9)
                for trajectory in trajectories
            ),
        },
        "joins": {
            "expected_action_rows": expected_actions,
            "joined_action_rows": len(action_rows),
            "missing_trajectories": missing_trajectories,
            "missing_actions": missing_actions,
        },
        "tokens": {
            "response_tokens_before": int(response_mask.sum().item()),
            "response_tokens_after": int(response_mask.sum().item()),
            "dp_world_size_simulated": args.dp_world_size,
            "dp_rank_token_counts": rank_token_counts,
        },
        "weights": {
            "minimum": float(output.row_weights.min().item()) if len(action_rows) else 0.0,
            "maximum": float(output.row_weights.max().item()) if len(action_rows) else 0.0,
            "mean": float(output.row_weights.mean().item()) if len(action_rows) else 0.0,
            "below_one": int(output.row_weights.lt(1).sum().item()),
            "equal_one": int(output.row_weights.eq(1).sum().item()),
            "above_one": int(output.row_weights.gt(1).sum().item()),
        },
        "hard_gates": hard_gates,
        "passed": bool(hard_gates and all(hard_gates.values())),
        "limitations": {
            "recorded_batch_not_optimizer_step": True,
            "eligible_roles": ["search", "reason"],
            "other_agent_rows_receive_weight_one": True,
            "dp_check_uses_complete_workflow_partition_simulation": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-files", type=Path, nargs="+", required=True)
    parser.add_argument("--paired-replays", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default="2wiki,hotpotqa,musique")
    parser.add_argument("--max-trajectories-per-dataset", type=int, default=32)
    parser.add_argument("--expected-rollouts-per-question", type=int, default=4)
    parser.add_argument("--dp-world-size", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    report = validate(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
