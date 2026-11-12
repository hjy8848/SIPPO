"""CPU self-test of data-parallel SIPPO updates: N gloo processes must reproduce the single-process update.

Each rank holds a full replica, updates on its own shard of complete sibling-question groups, backward runs without
gradient sync and the gradients are averaged once per mini-batch. Rank-specific micro-batch sizes prove that differing
micro-batch counts (the usual source of data-parallel deadlocks) are harmless.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch
import torch.nn as nn


def build_batch(questions: int = 2, siblings: int = 8, mode: str = "nrpsc"):
    from verl.trainer.ppo.nrpsc_objective import PlanView, frozen_tfidf, parse_plan

    rows = questions * siblings
    ids = (torch.arange(rows * 8).reshape(rows, 8) * 7 + 3) % 64
    responses = ids[:, -4:]
    generator = torch.Generator().manual_seed(1)
    tensors = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
               "position_ids": torch.arange(8).repeat(rows, 1), "responses": responses,
               "response_mask": torch.ones(rows, 4), "old_log_probs": torch.zeros(rows, 4),
               "advantages": torch.randn(rows, 4, generator=generator),
               "actor_dp_valid_rows": torch.ones(rows, dtype=torch.bool)}
    node_mask = torch.tensor([True, True, False, False])
    end_mask = torch.tensor([False, False, False, True])
    views = np.empty(rows, dtype=object)
    for row in range(rows):
        views[row] = PlanView(parse_plan(f'<search id="1">entity{row % siblings}</search>'), (node_mask,), (42,),
                              tuple(responses[row].tolist()), end_mask)
    outcome = np.asarray(([1.0] + [0.0] * (siblings - 1)) * questions)
    non_tensors = {"uid": np.asarray([f"q{row // siblings}" for row in range(rows)], dtype=object),
                   "trajectory_id": np.asarray([f"w{row}" for row in range(rows)], dtype=object),
                   "role": np.asarray(["PlanningAgent"] * rows, dtype=object),
                   "outcome_exact": outcome, "workflow_step": np.zeros(rows, dtype=np.int64),
                   "nrpsc_plan_view": views}
    meta = {"temperature": 1.0, "planner_outcome_contrast_minibatch_mode": "question_preserving",
            "planner_outcome_contrast_mode": mode, "planner_outcome_contrast_beta": 1.0,
            "planner_outcome_contrast_loss_weight": 0.04,
            "nrpsc_idf": frozen_tfidf([f"entity{i}" for i in range(siblings)]).idf, "nrpsc_similarity_threshold": 0.8}
    return tensors, non_tensors, meta


def run_rank(rank: int, world: int, port: int, micro_sizes, mode: str, queue) -> None:
    import sippo.actor_update as actor_update
    from sippo.actor_update import SippoActorConfig, SippoPolicyUpdater

    if world > 1:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = nn.Embedding(64, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    actor_update.forward_log_probs = lambda m, t, temperature, cfg, device_type: m(t["responses"]).squeeze(-1)
    tensors, non_tensors, meta = build_batch(mode=mode)
    rows = tensors["input_ids"].shape[0]
    if world > 1:
        per_rank = rows // world
        window = slice(rank * per_rank, (rank + 1) * per_rank)
        tensors = {k: v[window] for k, v in tensors.items()}
        non_tensors = {k: np.asarray(v)[window] for k, v in non_tensors.items()}
        meta["planner_outcome_contrast_data_parallel_size"] = world

    def average() -> None:
        for parameter in model.parameters():
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad.div_(world)

    cfg = SippoActorConfig(ppo_mini_batch_size=64, ppo_micro_batch_size_per_gpu=micro_sizes[rank], use_dynamic_bsz=False,
                           use_remove_padding=False, response_only_logits=False)
    extra = {"dp_size": world, "no_sync": contextlib.nullcontext, "allreduce_grads": average} if world > 1 else {}
    updater = SippoPolicyUpdater(model, optimizer, cfg, device="cpu", **extra)
    updater.update(tensors, non_tensors, meta)
    queue.put((rank, model.weight.detach().tolist()))
    if world > 1:
        torch.distributed.destroy_process_group()


def run(world: int, micro_sizes, mode: str = "nrpsc", port: int = 29613, timeout: float = 120.0):
    import multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.Queue()
    procs = [context.Process(target=run_rank, args=(rank, world, port, micro_sizes, mode, queue)) for rank in range(world)]
    for proc in procs:
        proc.start()
    results = {}
    for _ in range(world):
        rank, weight = queue.get(timeout=timeout)  # a deadlock shows up as a timeout here
        results[rank] = torch.tensor(weight)
    for proc in procs:
        proc.join(timeout=30)
    return [results[rank] for rank in range(world)]