from pathlib import Path

import pytest

from scripts.handoff_q25_typedmeta_gate_to_full468 import audit


def step_line(step: int, pair_count: float = 3.0) -> str:
    return (
        f"step:{step} - train/workflow/completed:128 - train/workflow/valid_rate:1.0 "
        f"- aux/planner_outcome_contrast/global_pair_count:{pair_count} "
        "- actor/optimizer_steps_expected:5.0 - actor/optimizer_steps_executed:5.0 "
        f"- actor/ppo_optimizer_step:1.0 - actor/grad_norm:1.2 - training/global_step:{step}"
    )


def test_two_complete_healthy_steps_pass(tmp_path: Path):
    log = tmp_path / "gate.log"
    log.write_text(step_line(235) + "\n" + step_line(236) + "\n", encoding="utf-8")

    report = audit(log)

    assert report["passed"] is True
    assert report["steps"]["236"]["completed"] == 128


@pytest.mark.parametrize(
    "text",
    [
        step_line(235),
        step_line(235) + "\n" + step_line(236, pair_count=0),
        step_line(235) + "\n" + step_line(236) + "\nTraceback",
    ],
)
def test_incomplete_empty_signal_or_fatal_gate_fails(tmp_path: Path, text: str):
    log = tmp_path / "gate.log"
    log.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        audit(log)
