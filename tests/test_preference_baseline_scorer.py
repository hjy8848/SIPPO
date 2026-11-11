from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_dagent_preference_baseline.py"
SPEC = importlib.util.spec_from_file_location("preference_baseline_scorer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(pair_id, sum_margin, mean_margin, chosen_length=4, rejected_length=4, **kwargs):
    row = {
        "pair_id": pair_id,
        "sum_margin": sum_margin,
        "mean_margin": mean_margin,
        "chosen_token_count": chosen_length,
        "rejected_token_count": rejected_length,
        "agent_name": "ReasonerAgent",
        "variant": "factual_condition",
    }
    row.update(kwargs)
    return row


def test_length_bucket_boundaries():
    assert MODULE.length_bucket(-33) == "chosen_33+_tokens_shorter"
    assert MODULE.length_bucket(-32) == "chosen_9-32_tokens_shorter"
    assert MODULE.length_bucket(-1) == "chosen_1-8_tokens_shorter"
    assert MODULE.length_bucket(0) == "equal_length"
    assert MODULE.length_bucket(8) == "chosen_1-8_tokens_longer"
    assert MODULE.length_bucket(9) == "chosen_9-32_tokens_longer"
    assert MODULE.length_bucket(33) == "chosen_33+_tokens_longer"


def test_summary_keeps_dpo_and_length_normalized_metrics_separate():
    rows = [
        result("a", 2.0, -0.2, chosen_length=2, rejected_length=5),
        result("b", -1.0, 0.4, chosen_length=8, rejected_length=3),
    ]
    summary = MODULE.summarize(rows)
    assert summary["count"] == 2
    assert summary["sum_pair_accuracy"] == 0.5
    assert summary["mean_pair_accuracy"] == 0.5
    assert summary["sum_margin_mean"] == 0.5
    assert summary["mean_margin_mean"] == 0.1
    assert summary["chosen_token_count_mean"] == 5
    assert summary["rejected_token_count_mean"] == 4


def test_grouped_summary_reports_factual_and_flipped_separately():
    rows = [
        result("a", 1.0, 0.1, variant="factual_condition"),
        result("b", -1.0, -0.1, variant="flipped_condition"),
    ]
    grouped = MODULE.grouped_summary(rows, "variant")
    assert grouped["factual_condition"]["sum_pair_accuracy"] == 1.0
    assert grouped["flipped_condition"]["sum_pair_accuracy"] == 0.0


def test_combine_pair_calculates_both_margin_definitions():
    row = {
        "pair_id": "p",
        "group_id": "g",
        "split": "train",
        "agent_name": "QueryRewriteAgent",
        "variant": "factual_dependency",
        "downstream_relation": "occupation",
        "operation": "earlier",
        "reasoning_mode": "direct",
    }
    scores = {
        "chosen": {"token_count": 2, "logp_sum": -1.0, "logp_mean": -0.5},
        "rejected": {"token_count": 4, "logp_sum": -3.0, "logp_mean": -0.75},
    }
    combined = MODULE.combine_pair(row, scores)
    assert combined["sum_margin"] == 2.0
    assert combined["mean_margin"] == 0.25
    assert combined["token_count_delta"] == -2
    assert combined["length_bucket"] == "chosen_1-8_tokens_shorter"
