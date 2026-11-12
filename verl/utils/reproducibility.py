"""Small reproducibility helpers shared by training entry points."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_process_rngs(seed: int, rank: int) -> int:
    """Seed Python, NumPy, PyTorch, and CUDA deterministically for one rank."""
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32))
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return process_seed
