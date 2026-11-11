import json

from scripts.audit_dagent_current_policy_lexicographic_rollouts import (
    load_rollout_groups,
)


def test_current_policy_loader_requires_complete_four_sibling_contract(tmp_path):
    root = tmp_path / "trajectories"
    root.mkdir()
    rows = []
    for rollout_index, (em, f1) in enumerate(((0, 0.0), (0, 0.5), (1, 1.0), (0, 0.25))):
        rows.append(
            {
                "sample_index": 7,
                "rollout_index": rollout_index,
                "question": "q",
                "gold_answer": ["a"],
                "pred_answer": "a" if em else "x",
                "em": em,
                "f1": f1,
                "pipeline_error": None,
                "error": None,
                "raw_trace": {
                    "contract_protocol": {"version": "cp-dagent-v2-no-declared-out"},
                    "sft_events": [],
                },
            }
        )
    path = root / "full_logs_run_toy.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    groups, evidence = load_rollout_groups(root, "run", datasets={"toy": 1})
    assert len(groups) == 1
    assert set(groups[0]["models"]) == {
        "rollout_0", "rollout_1", "rollout_2", "rollout_3"
    }
    assert evidence["contract"]["passed"] is True


def test_current_policy_report_scope_stops_before_training():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_dagent_current_policy_lexicographic_rollouts.py"
    ).read_text(encoding="utf-8")
    assert '"optimizer_updates": 0' in source
    assert '"advance_to_one_batch_gradient_check"' in source
    assert '"short_training": False' in source
