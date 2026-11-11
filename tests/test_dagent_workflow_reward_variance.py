from scripts.analyze_dagent_workflow_reward_variance import (
    analyze_rows,
    histogram_metrics,
    parse_step_metrics,
    reconstruct_histogram,
)


def test_reconstruct_histogram_exactly() -> None:
    expected = (5, 4, 3, 2, 10)
    em, best_of_n, mean_std = histogram_metrics(expected)
    assert reconstruct_histogram(24, 4, em, best_of_n, mean_std) == expected


def test_parse_and_analyze_training_line() -> None:
    histogram = (6, 5, 4, 3, 6)
    em, best_of_n, mean_std = histogram_metrics(histogram)
    text = (
        "(TaskRunner pid=1) step:7 - "
        f"train/rollout/em_aggregate:{em} - "
        f"train/rollout/best_of_n_em:{best_of_n} - "
        f"train/rollout/within_question_em_std:{mean_std} - "
        "train/workflow/completed:96 - training/global_step:7"
    )
    report = analyze_rows(parse_step_metrics(text), rollout_n=4)
    assert report["reward_histogram"] == {"0": 6, "1": 5, "2": 4, "3": 3, "4": 6}
    assert report["zero_variance_rate"] == 0.5
    assert report["effective_advantage_workflows"] == 48
