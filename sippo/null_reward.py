"""Placeholder reward for the verl 0.9.1 agent loop.

The 0.9.1 agent loop scores every generated step through a reward function. In the JADE/SIPPO workflow the real
reward is computed by ``qa_manager`` once the whole workflow has finished (``reward_tensor`` in the rollout batch), and the
per-step score of the agent loop is never read. This function therefore returns 0.0 so the agent loop does not fail
on the dataset names (e.g. ``2wiki``) that verl's built-in scorers do not know.
"""


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    return 0.0