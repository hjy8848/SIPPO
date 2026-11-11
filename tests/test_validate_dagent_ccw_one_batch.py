import torch

from scripts.validate_dagent_ccw_one_batch import apply_by_complete_workflow_partition
from tests.test_workflow_ccw import apply_workflow_ccw_weights


def test_complete_workflow_dp_partition_matches_global_weighting():
    advantages = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [-1.0, -1.0], [-1.0, 0.0]]
    )
    response_mask = advantages.ne(0)
    credits = torch.tensor([1.0, -0.5, 0.5, -1.0])
    trajectory_ids = ["w0", "w0", "w1", "w1"]

    global_output = apply_workflow_ccw_weights(
        advantages,
        response_mask,
        credits,
    )
    partitioned, rank_tokens = apply_by_complete_workflow_partition(
        advantages,
        response_mask,
        credits,
        trajectory_ids,
        world_size=2,
    )

    assert torch.equal(partitioned, global_output.advantages)
    assert rank_tokens == [3, 3]
