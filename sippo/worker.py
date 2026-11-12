"""SIPPO actor/rollout/ref worker: the stock verl 0.9.1 engine worker plus ``update_actor_sippo``.

``update_actor_sippo`` runs the SIPPO (workflow-RLOO / PSC) policy update on the model and optimizer that the 0.9.1
FSDP engine owns. Model placement, optimizer offload, gradient clipping, the non-finite-gradient skip and the
learning-rate scheduler all stay with the engine; only the objective and the mini-batch schedule are SIPPO's.

Scope: a single data-parallel rank (one NPU per run, as in the original W/P runs). Multi-rank use would need the
question-preserving DP partitioning of the original trainer and is rejected explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_device_name
from verl.workers.engine_workers import ActorRolloutRefWorker

from sippo.actor_update import SippoActorConfig, SippoPolicyUpdater


class SippoActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        from sippo.seed import seed_process_from_env

        seed_process_from_env()
        super().__init__(*args, **kwargs)

    def _sippo_actor_config(self) -> SippoActorConfig:
        raw = OmegaConf.to_container(self.config.actor, resolve=True)
        # use_remove_padding lives on the model config in verl 0.9.1 (it was an actor field in the JADE fork)
        raw["use_remove_padding"] = bool(self.config.model.get("use_remove_padding", False))
        return SippoActorConfig.from_actor_config(raw)

    @staticmethod
    def _no_sync_factory(module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(module, FSDP):
            return module.no_sync
        raise NotImplementedError("data-parallel SIPPO updates need the FSDP1 engine (actor.strategy=fsdp)")

    @staticmethod
    def _grad_averager(module):
        def average() -> None:
            world = torch.distributed.get_world_size()
            for parameter in module.parameters():
                grad = parameter.grad
                if grad is None:
                    grad = torch.zeros_like(parameter)
                elif grad.dtype != parameter.dtype or grad.device != parameter.device:
                    # under no_sync FSDP leaves the low-precision (bf16) gradient in place; the optimizer needs the
                    # parameter dtype (fp32 master weights), as the synchronising backward would have produced
                    grad = grad.to(device=parameter.device, dtype=parameter.dtype)
                parameter.grad = grad
                torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                parameter.grad.div_(world)
        return average

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_actor_sippo(
        self,
        tensors: dict[str, torch.Tensor],
        non_tensors: dict[str, np.ndarray],
        meta: dict[str, Any],
    ) -> dict[str, list]:
        assert self._is_actor and self.actor is not None, "update_actor_sippo needs the actor role"
        engine = self.actor.engine
        dp_size = engine.get_data_parallel_size()
        cfg = self._sippo_actor_config()
        device = torch.device(get_device_name(), get_device_id())
        extra = {}
        if dp_size > 1:
            # Each rank holds a full model replica (fsdp_size=1) and updates on its own contiguous shard of complete
            # sibling-question groups (the trainer concatenated equal-row shards, padding rows are masked out).
            rows = int(tensors["input_ids"].shape[0])
            if rows % dp_size:
                raise ValueError(f"{rows} rows are not divisible by the actor data-parallel size {dp_size}")
            per_rank = rows // dp_size
            rank = engine.get_data_parallel_rank()
            window = slice(rank * per_rank, (rank + 1) * per_rank)
            tensors = {key: value[window] for key, value in tensors.items()}
            non_tensors = {key: np.asarray(value)[window] for key, value in non_tensors.items()}
            meta = dict(meta)
            meta["planner_outcome_contrast_data_parallel_size"] = dp_size
            extra = {"dp_size": dp_size, "no_sync": self._no_sync_factory(engine.module),
                     "allreduce_grads": self._grad_averager(engine.module)}
        with engine.train_mode():
            updater = SippoPolicyUpdater(
                engine.module,
                engine.optimizer,
                cfg,
                device=device,
                optimizer_step_fn=engine.optimizer_step,
                **extra,
            )
            metrics = updater.update(tensors, non_tensors, meta)
            lr = engine.lr_scheduler_step()
        metrics["actor/lr"] = [float(lr)]
        metrics["actor/lr_scheduler_step"] = [1.0]
        return metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def grad_compare_sippo(
        self,
        tensors: dict[str, torch.Tensor],
        non_tensors: dict[str, np.ndarray],
        meta: dict[str, Any],
        psc_weight: float = 0.02,
    ) -> dict:
        """Frozen-batch gradient comparison (base W loss, PSC, NR-PSC). Never steps the optimizer or the scheduler."""
        assert self._is_actor and self.actor is not None, "grad_compare_sippo needs the actor role"
        engine = self.actor.engine
        if engine.get_data_parallel_size() != 1:
            raise NotImplementedError("SIPPO gradient comparison supports a single data-parallel rank")
        cfg = self._sippo_actor_config()
        device = torch.device(get_device_name(), get_device_id())
        with engine.train_mode():
            updater = SippoPolicyUpdater(engine.module, engine.optimizer, cfg, device=device,
                                         optimizer_step_fn=engine.optimizer_step)
            return updater.compare_gradients(tensors, non_tensors, meta, psc_weight=psc_weight)
