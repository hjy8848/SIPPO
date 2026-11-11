import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_psc_training_log", ROOT / "scripts" / "analyze_psc_training_log.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def metric_line(
    step=1,
    workflows=128,
    expected=6,
    executed=6,
    signal=1,
    informative=1.5,
    pairs=4,
    ratio=0.2,
    planner_norm=0.01,
    psc_norm=0.002,
):
    return (
        f"(TaskRunner pid=1) step:{step} - train/workflow/completed:{workflows} "
        "- train/workflow/incomplete_question_groups:0 - train/workflow/valid_rate:1 "
        "- train/rollout/em_aggregate:0.25 - train/rollout/best_of_n_em:0.5 "
        "- train/workflow/planned_reason_nodes_mean:0.2 - train/agent_calls/ReasonerAgent:0.2 "
        "- actor/question_preserving_minibatch_count:6 "
        f"- aux/planner_outcome_contrast/global_informative_question_count:{informative} "
        f"- aux/planner_outcome_contrast/global_pair_count:{pairs} "
        "- aux/planner_outcome_contrast/empty_refreshed_minibatch:0 "
        f"- aux/planner_outcome_contrast/refreshed_weighted_poc_to_planner_em_direct_gradient_ratio:{ratio} "
        "- aux/planner_outcome_contrast/refreshed_weighted_poc_planner_em_direct_gradient_cosine:0.7 "
        f"- aux/planner_outcome_contrast/refreshed_planner_em_direct_gradient_norm:{planner_norm} "
        f"- aux/planner_outcome_contrast/refreshed_weighted_poc_direct_gradient_norm:{psc_norm} "
        f"- actor/optimizer_steps_expected:{expected} - actor/optimizer_steps_executed:{executed} "
        f"- aux/planner_outcome_contrast/global_signal_required:{signal} "
        "- aux/planner_outcome_contrast/non_planner_nonzero_tokens:0 - actor/grad_norm:1 "
        f"- training/global_step:{step}"
    )


def test_parse_and_analyze_healthy_steps():
    rows = MODULE.parse_steps(metric_line(1) + "\n" + metric_line(2))
    report = MODULE.analyze(rows, expected_workflows=128, expected_global_signal=True)
    assert report["steps"] == 2
    assert report["summary"]["estimated_informative_questions"]["mean"] == 9.0
    assert report["summary"]["estimated_informative_questions"]["sum"] == 18.0
    assert report["summary"]["estimated_pairs"]["mean"] == 24.0
    assert report["summary"]["optimizer_steps_executed"]["sum"] == 12.0
    assert report["audit"]["all_engineering_invariants_pass"] is True
    assert report["audit"]["positive_alignment_rate"] == 1.0


def test_invariant_failures_are_recorded_by_step():
    rows = MODULE.parse_steps(metric_line(3, workflows=127, expected=6, executed=5, signal=0))
    report = MODULE.analyze(rows, expected_workflows=128, expected_global_signal=True)
    assert report["violations"]["workflow_count"] == [3]
    assert report["violations"]["optimizer_step_mismatch"] == [3]
    assert report["violations"]["global_signal_contract_mismatch"] == [3]
    assert report["audit"]["all_engineering_invariants_pass"] is False


def test_false_signal_contract_is_valid_when_explicit():
    rows = MODULE.parse_steps(metric_line(4, signal=0))
    report = MODULE.analyze(rows, expected_workflows=128, expected_global_signal=False)
    assert report["violations"]["global_signal_contract_mismatch"] == []
    assert report["audit"]["all_engineering_invariants_pass"] is True
    assert report["audit"]["signal_contract_pass"] is True


def test_robust_ratio_excludes_sparse_and_denominator_degenerate_steps():
    rows = MODULE.parse_steps(
        "\n".join(
            [
                metric_line(1, signal=0),
                metric_line(2, signal=0, ratio=100.0),
                metric_line(
                    3,
                    signal=0,
                    informative=0.0,
                    pairs=0.0,
                    ratio=0.0,
                    planner_norm=0.0,
                    psc_norm=0.0,
                ),
            ]
        )
    )
    report = MODULE.analyze(rows, expected_workflows=128, expected_global_signal=False)
    assert report["violations"]["denominator_degenerate_gradient_ratio"] == [2]
    assert report["summary"]["interpretable_gradient_ratio"]["count"] == 1
    assert report["summary"]["interpretable_gradient_ratio"]["mean"] == 0.2
    assert report["audit"]["interpretable_gradient_ratio_steps"] == 1
    assert report["audit"]["signal_present_every_step"] is False
    assert report["audit"]["signal_nonempty_step_rate"] == 2 / 3
    assert report["audit"]["signal_contract_pass"] is True


def test_merge_segments_selects_final_lineage_and_records_provenance(tmp_path):
    original = tmp_path / "original.log"
    resume = tmp_path / "resume.log"
    original.write_text("\n".join(metric_line(step) for step in range(1, 5)))
    resume.write_text("\n".join(metric_line(step) for step in range(3, 6)))

    rows, provenance = MODULE.merge_segments(
        [(original, 1, 2), (resume, 3, 5)]
    )

    assert [int(row["training/global_step"]) for row in rows] == [1, 2, 3, 4, 5]
    assert provenance[0]["step_start"] == 1
    assert provenance[1]["completed_steps"] == 3


def test_merge_segments_fails_closed_on_overlap_or_gap(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join(metric_line(step) for step in range(1, 6)))

    try:
        MODULE.merge_segments([(log, 1, 3), (log, 3, 5)])
    except RuntimeError as exc:
        assert "Overlapping" in str(exc)
    else:
        raise AssertionError("overlap must fail")

    try:
        MODULE.merge_segments([(log, 1, 2), (log, 4, 5)])
    except RuntimeError as exc:
        assert "not contiguous" in str(exc)
    else:
        raise AssertionError("gap must fail")


def test_segment_assignment_supports_windows_drive_paths():
    label, path, start, end = MODULE.parse_segment_assignment(
        r"PSC=C:\runs\train.log@1:234"
    )
    assert label == "PSC"
    assert str(path) == r"C:\runs\train.log"
    assert (start, end) == (1, 234)


def test_markdown_handles_pure_em_without_psc_metrics():
    row = MODULE.parse_metric_line(metric_line())
    for key in list(row):
        if key.startswith(MODULE.PREFIX):
            del row[key]
    report = MODULE.analyze([row], expected_workflows=128)

    markdown = MODULE.render_markdown({"PureEM": report})

    assert "| PureEM | 1 | 1-1 | n/a | n/a | n/a | 0.2000 | True |" in markdown
