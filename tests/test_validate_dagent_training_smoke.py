import pytest

from scripts.validate_dagent_training_smoke import validate


def smoke_log(
    *,
    poc_enabled: float = 0.0,
    grad_norm: str = "0.75",
    pair_count: str = "8",
) -> str:
    return (
        "step:1 - train/workflow/completed:128 "
        "- train/workflow/incomplete_question_groups:0 "
        "- actor/optimizer_steps_expected:6 "
        "- actor/optimizer_steps_executed:6 "
        "- actor/pg_loss:0.001 "
        f"- actor/grad_norm:{grad_norm} "
        f"- aux/planner_outcome_contrast/enabled:{poc_enabled} "
        f"- aux/planner_outcome_contrast/global_pair_count:{pair_count} "
        "- aux/planner_outcome_contrast/global_informative_question_count:3 "
        "- aux/planner_outcome_contrast/global_signal_required:1 "
        "- aux/planner_outcome_contrast/empty_refreshed_minibatch:0 "
        "- aux/planner_outcome_contrast/refreshed_weighted_poc_direct_gradient_norm:0.002 "
        "- training/global_step:1\n"
    )


def test_validate_accepts_complete_finite_smokes():
    validate(smoke_log(), expected_step=1, expected_workflows=128, poc_mode="off")
    validate(smoke_log(poc_enabled=1.0), expected_step=1, expected_workflows=128, poc_mode="real")


@pytest.mark.parametrize(
    "text, match",
    (
        (smoke_log(grad_norm="nan"), "non-finite"),
        (smoke_log() + "CUDA out of memory\n", "fatal pattern"),
        (smoke_log(poc_enabled=0.0), "PSC mode mismatch"),
        (smoke_log(poc_enabled=1.0, pair_count="0"), "no eligible sibling signal"),
    ),
)
def test_validate_rejects_invalid_real_psc_smokes(text: str, match: str):
    with pytest.raises(ValueError, match=match):
        validate(text, expected_step=1, expected_workflows=128, poc_mode="real")
