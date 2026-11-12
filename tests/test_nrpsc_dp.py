import torch

from sippo import dp_selftest


def test_two_ranks_match_single_process_for_nrpsc_and_psc_and_plain():
    for mode, port in (("nrpsc", 29613), ("real", 29614), ("off", 29615)):
        single = dp_selftest.run(1, [4], mode=mode, port=port)[0]
        ranks = dp_selftest.run(2, [4, 2], mode=mode, port=port + 10)  # different micro-batch counts per rank
        assert torch.allclose(ranks[0], ranks[1], atol=1e-7), mode  # replicas stay identical
        assert torch.allclose(ranks[0], single, atol=1e-6), (mode, (ranks[0] - single).abs().max())
        assert not torch.equal(single, torch.zeros_like(single))