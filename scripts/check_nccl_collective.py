#!/usr/bin/env python3
"""Exercise the NCCL collectives needed by FSDP before starting a long run."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        # A moderately large tensor catches failures that tiny scalar probes miss.
        tensor = torch.full(
            (16 * 1024 * 1024,),
            float(rank + 1),
            dtype=torch.float32,
            device=f"cuda:{local_rank}",
        )
        expected = world_size * (world_size + 1) / 2
        for _ in range(3):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if not torch.allclose(tensor[0], torch.tensor(expected, device=tensor.device)):
                raise RuntimeError(
                    f"all_reduce mismatch on rank {rank}: {tensor[0].item()} != {expected}"
                )
            tensor.fill_(float(rank + 1))
            dist.barrier()
        if rank == 0:
            print(
                f"[PASS] NCCL all_reduce/barrier succeeded for world_size={world_size}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
