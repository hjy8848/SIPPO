import csv

import pytest

from scripts.audit_dagent_lexicographic_f1_reward import (
    audit_groups,
    load_pseudo_siblings,
    rloo_advantages,
)


def _group(dataset, em, f1):
    labels = ("SFT", "PureEM", "PSC")
    return {
        "dataset": dataset,
        "question": "q",
        "gold_answer": "[\"a\"]",
        "models": {
            label: {"prediction": label, "em": em[index], "f1": f1[index]}
            for index, label in enumerate(labels)
        },
    }


def test_em_f1_tiebreak_preserves_all_mixed_em_rloo_signs():
    groups = [
        _group("wiki", [1, 0, 0], [1.0, 0.99, 0.0]),
        _group("wiki", [1, 1, 0], [1.0, 1.0, 0.99]),
        _group("wiki", [0, 1, 0], [0.99, 1.0, 0.8]),
    ]
    report = audit_groups(groups, bootstrap_samples=100, seed=4)
    assert report["mixed_em_members_checked"] == 9
    assert report["mixed_em_sign_mismatches"] == 0
    assert report["maximum_mixed_em_advantage_delta"] == 0.0


def test_f1_adds_signal_only_when_em_is_tied():
    groups = [
        _group("wiki", [0, 0, 0], [0.0, 0.5, 0.9]),
        _group("wiki", [0, 0, 0], [0.0, 0.0, 0.0]),
        _group("wiki", [1, 1, 1], [1.0, 1.0, 1.0]),
    ]
    report = audit_groups(groups, bootstrap_samples=100, seed=4)
    assert report["new_f1_signal_groups"] == 1
    assert report["all_em_zero_groups"] == 2
    assert rloo_advantages([0.0, 0.05, 0.09]).tolist() == pytest.approx(
        [-0.07, 0.005, 0.065]
    )


def test_loader_reconstructs_and_strictly_checks_all_pairs(tmp_path):
    path = tmp_path / "paired.csv"
    fields = [
        "reference", "candidate", "dataset", "question", "gold_answer",
        "reference_prediction", "candidate_prediction", "reference_em",
        "candidate_em", "reference_f1", "candidate_f1",
    ]
    metrics = {
        "SFT": ("x", 0, 0.0),
        "PureEM": ("a", 1, 1.0),
        "PSC": ("almost", 0, 0.5),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for reference, candidate in (("SFT", "PureEM"), ("SFT", "PSC"), ("PureEM", "PSC")):
            ref = metrics[reference]
            cand = metrics[candidate]
            writer.writerow(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "dataset": "toy",
                    "question": "q",
                    "gold_answer": '["a"]',
                    "reference_prediction": ref[0],
                    "candidate_prediction": cand[0],
                    "reference_em": ref[1],
                    "candidate_em": cand[1],
                    "reference_f1": ref[2],
                    "candidate_f1": cand[2],
                }
            )
    groups = load_pseudo_siblings(path, datasets={"toy": 1})
    assert len(groups) == 1
    assert set(groups[0]["models"]) == {"SFT", "PureEM", "PSC"}


def test_report_scope_never_authorizes_training():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_dagent_lexicographic_f1_reward.py"
    ).read_text(encoding="utf-8")
    assert '"advance_to_current_policy_rollout_pilot"' in source
    assert '"one_batch_gradient_check": False' in source
    assert '"short_training": False' in source
