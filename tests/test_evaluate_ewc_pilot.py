from scripts.evaluate_ewc_pilot import paired_summary


def row(pair_id, variant, answer, gold, correct):
    return {
        "pair_id": pair_id,
        "variant": variant,
        "final_answer": answer,
        "gold_answer": [gold],
        "em": correct,
    }


def test_paired_summary_distinguishes_correct_change_from_any_change():
    results = [
        row("one", "base", "alpha", "alpha", 1),
        row("one", "dependency_changed", "beta", "beta", 1),
        row("one", "answer_preserving", "alpha", "alpha", 1),
        row("two", "base", "gamma", "gamma", 1),
        row("two", "dependency_changed", "delta", "epsilon", 0),
        row("two", "answer_preserving", "zeta", "gamma", 0),
    ]

    summary = paired_summary(results)

    assert summary["pair_count"] == 2
    assert summary["joint_accuracy"] == 0.5
    assert summary["dependency_changed_accuracy"] == 0.5
    assert summary["changed_answer_rate"] == 1.0
    assert summary["unnecessary_answer_change_rate"] == 0.5
    assert summary["answer_preserving_invariance_rate"] == 0.5
    assert summary["stale_source_answer_rate"] == 0.0


def test_paired_summary_tracks_stale_source_answer_and_incomplete_pairs():
    results = [
        row("one", "base", "alpha", "alpha", 1),
        row("one", "dependency_changed", "alpha", "beta", 0),
        row("one", "answer_preserving", "alpha", "alpha", 1),
        row("incomplete", "base", "x", "x", 1),
    ]

    summary = paired_summary(results)

    assert summary["pair_count"] == 1
    assert summary["stale_source_answer_rate"] == 1.0
    assert summary["changed_answer_rate"] == 0.0
    assert summary["incomplete_pair_ids"] == ["incomplete"]
