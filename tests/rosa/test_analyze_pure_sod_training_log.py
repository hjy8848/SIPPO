from pathlib import Path
import importlib.util

import pytest


MODULE_PATH = (
    Path(__file__).parents[2] / "scripts" / "analyze_pure_sod_training_log.py"
)
SPEC = importlib.util.spec_from_file_location("pure_sod_log_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def step_line(step: int, **overrides: float) -> str:
    metrics = {
        "aux/workflow_sod/sod_only": 1.0,
        "aux/workflow_sod/task_advantage_rms": 0.0,
        "aux/workflow_sod/reference_switch_count": 0.0,
        "aux/workflow_sod/future_reference_action_count": 0.0,
        "aux/workflow_sod/questions_with_successful_sibling": 5.0,
        "aux/workflow_sod/scored_row_count": 10.0,
        "aux/workflow_sod/response_token_count": 200.0,
        "aux/workflow_sod/empty_signal_step": 0.0,
        "aux/workflow_sod/actor_update_skipped_no_signal": 0.0,
        "actor/workflow_dp_size": 4.0,
        "actor/workflow_complete_workflows_per_minibatch": 6.0,
        "actor/optimizer_steps_expected": 4.0,
        "actor/workflow_loss_total_mass": 4.0,
        "actor/workflow_loss_workflow_count": 96.0,
        "actor/workflow_loss_logical_action_count": 480.0,
        "aux/workflow_sod/trajectory_count": 12.0,
        "aux/workflow_sod/fixed_reference_trajectory_count": 12.0,
        "aux/workflow_sod/missing_action_reference_count": 2.0,
        "aux/workflow_sod/topology_mismatch_trajectory_count": 3.0,
        "aux/workflow_sod/scored_invalid_target_row_count": 1.0,
        "aux/workflow_sod/missing_reason_trajectory_count": 1.0,
        "aux/workflow_sod/scored_row_count/PlanningAgent": 4.0,
        "aux/workflow_sod/generated_row_count/PlanningAgent": 96.0,
        "aux/workflow_sod/coverage/PlanningAgent": 4.0 / 96.0,
        "aux/workflow_sod/teacher_gap_mean/PlanningAgent": -0.25,
        "aux/workflow_sod/teacher_endorsement_rate/PlanningAgent": 0.2,
        "train/workflow/valid_rate": 0.9,
    }
    metrics.update(overrides)
    fields = " - ".join(f"{key}:{value}" for key, value in metrics.items())
    return f"\x1b[36m(TaskRunner pid=123)\x1b[0m step:{step} - {fields}\n"


def test_parse_step_line_strips_ansi_and_reads_metrics():
    parsed = MODULE.parse_step_line(step_line(3, **{"actor/entropy": 0.07}))
    assert parsed is not None
    step, metrics = parsed
    assert step == 3
    assert metrics["actor/entropy"] == 0.07
    assert metrics["aux/workflow_sod/sod_only"] == 1.0


def test_analyze_reports_trends_and_hard_gate_failures(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        step_line(1, **{"train/workflow/valid_rate": 0.9})
        + step_line(
            2,
            **{
                "train/workflow/valid_rate": 0.8,
                "aux/workflow_sod/task_advantage_rms": 0.25,
                "actor/ppo_kl": 0.002,
            },
        ),
        encoding="utf-8",
    )
    report = MODULE.analyze(
        log,
        window=1,
        expected_dp_size=4,
        expected_minibatch_workflows=6,
        expected_total_mass=4,
    )
    assert report["last_step"] == 2
    assert report["missing_steps"] == []
    assert report["trends"]["train/workflow/valid_rate"][
        "delta_last_minus_first_window"
    ] == pytest.approx(-0.1)
    assert report["trends"]["actor/ppo_kl"]["last_window_mean"] == 0.002
    assert report["empty_signal_steps"] == []
    assert report["effective_actor_update_step_count"] == 2
    assert report["optimizer_step_count"] == 8.0
    assert report["mechanism"]["derived_trends"][
        "derived/workflow_sod/active_trajectory_fraction"
    ]["last_window_mean"] == pytest.approx(12.0 / 96.0)
    assert report["mechanism"]["derived_trends"][
        "derived/workflow_sod/scored_action_fraction"
    ]["last_window_mean"] == pytest.approx(10.0 / 480.0)
    assert report["mechanism"]["count_totals"][
        "aux/workflow_sod/missing_reason_trajectory_count"
    ] == 2.0
    planning = report["mechanism"]["roles"]["PlanningAgent"]
    assert planning["scored_row_total"] == 8.0
    assert planning["generated_row_total"] == 192.0
    assert planning["aggregate_row_coverage"] == pytest.approx(4.0 / 96.0)
    assert planning["trends"]["teacher_gap_mean"]["last_window_mean"] == -0.25
    assert report["hard_gate_failures"] == [
        {
            "step": 2,
            "metric": "aux/workflow_sod/task_advantage_rms",
            "expected": 0.0,
            "actual": 0.25,
        }
    ]


def test_analyze_flags_zero_signal_and_missing_steps(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        step_line(1)
        + step_line(
            3,
            **{"aux/workflow_sod/questions_with_successful_sibling": 0.0},
        ),
        encoding="utf-8",
    )
    report = MODULE.analyze(log)
    assert report["missing_steps"] == [2]
    assert any(
        failure["metric"]
        == "aux/workflow_sod/questions_with_successful_sibling"
        for failure in report["hard_gate_failures"]
    )


def test_analyze_accepts_explicit_empty_signal_step_without_optimizer_update(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        step_line(
            25,
            **{
                "aux/workflow_sod/questions_with_successful_sibling": 0.0,
                "aux/workflow_sod/scored_row_count": 0.0,
                "aux/workflow_sod/response_token_count": 0.0,
                "aux/workflow_sod/empty_signal_step": 1.0,
                "aux/workflow_sod/actor_update_skipped_no_signal": 1.0,
                "actor/optimizer_steps_expected": 0.0,
            },
        ),
        encoding="utf-8",
    )

    report = MODULE.analyze(log, expected_minibatch_workflows=6)
    assert report["hard_gate_failures"] == []
    assert report["missing_hard_gate_metrics"] == {}
    assert report["empty_signal_steps"] == [25]
    assert report["empty_signal_step_fraction"] == 1.0
    assert report["longest_consecutive_empty_signal_steps"] == 1
    assert report["effective_actor_update_step_count"] == 0
    assert report["optimizer_step_count"] == 0.0


def test_analyze_expects_one_optimizer_step_for_standalone_forward_kl(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        step_line(
            1,
            **{
                "aux/workflow_sod/forward_kl_only": 1.0,
                "actor/optimizer_steps_expected": 1.0,
                "aux/workflow_sod_forward_kl/format_prefix_forward_kl": 0.6,
                "aux/workflow_sod_forward_kl/response_body_forward_kl": 0.05,
                "aux/workflow_sod_forward_kl/full_response_forward_kl": 0.3,
                "aux/workflow_sod_forward_kl/gate_forward_kl": 0.3,
                "aux/workflow_sod_forward_kl/gate_on_full_response_kl": 1.0,
                "aux/workflow_sod_forward_kl/exclude_opening_tag_from_loss": 1.0,
                "aux/workflow_sod_forward_kl/excluded_opening_tag_token_count": 3.0,
                "aux/workflow_sod_forward_kl/loss_token_start": 0.0,
                "aux/workflow_sod/opening_tag_mask_fallback_count": 1.0,
            },
        ),
        encoding="utf-8",
    )

    report = MODULE.analyze(log)
    assert report["hard_gate_failures"] == []
    assert report["optimizer_step_count"] == 1.0
    assert report["trends"][
        "aux/workflow_sod_forward_kl/format_prefix_forward_kl"
    ]["last_window_mean"] == pytest.approx(0.6)
    assert report["trends"][
        "aux/workflow_sod_forward_kl/gate_forward_kl"
    ]["last_window_mean"] == pytest.approx(0.3)
    assert report["trends"][
        "aux/workflow_sod_forward_kl/excluded_opening_tag_token_count"
    ]["last_window_mean"] == pytest.approx(3.0)


def test_analyze_flags_forward_kl_gate_mismatch(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        step_line(
            1,
            **{
                "aux/workflow_sod/forward_kl_only": 1.0,
                "actor/optimizer_steps_expected": 1.0,
                "aux/workflow_sod_forward_kl/full_response_forward_kl": 0.3,
                "aux/workflow_sod_forward_kl/gate_forward_kl": 0.2,
                "aux/workflow_sod_forward_kl/gate_on_full_response_kl": 1.0,
            },
        ),
        encoding="utf-8",
    )

    report = MODULE.analyze(log)
    assert report["hard_gate_failures"] == [
        {
            "step": 1,
            "metric": "aux/workflow_sod_forward_kl/gate_forward_kl",
            "expected": 0.3,
            "actual": 0.2,
        }
    ]


def test_longest_consecutive_empty_signal_run():
    assert MODULE.longest_consecutive_run([]) == 0
    assert MODULE.longest_consecutive_run([7, 3, 4, 9, 8]) == 3
