from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "verl" / "utils" / "reproducibility.py"
SPEC = importlib.util.spec_from_file_location("sft_reproducibility", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sample_rngs() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(()))


def test_seed_sft_process_replays_all_rng_streams():
    assert MODULE.seed_process_rngs(17, 3) == 20
    first = sample_rngs()
    assert MODULE.seed_process_rngs(17, 3) == 20
    assert sample_rngs() == first


def test_seed_sft_process_separates_ranks():
    MODULE.seed_process_rngs(17, 0)
    rank_zero = sample_rngs()
    MODULE.seed_process_rngs(17, 1)
    assert sample_rngs() != rank_zero
