import json

import pytest

from scripts.compare_shoppingbench_classifier_only_online import compare


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_classifier_online_gate_requires_identity_route_and_outcome(tmp_path):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    for task in ("product", "shop", "voucher"):
        old_rows, new_rows = [], []
        for index in range(16):
            sample_id = f"{task}-{index:04d}"
            plan = f"<plan><search>{sample_id}</search></plan>"
            score = {"exact_id_fraction": 1.0, "exact_sequence_success": True,
                     "first_id_exact": True}
            old_rows.append({"sample_id": sample_id, "score": score,
                             "retrieval_calls": 1, "raw_trace": {"raw_plan_output": plan}})
            new_rows.append({"sample_id": sample_id, "score": score,
                             "retrieval_calls": 1, "raw_trace": {
                                 "raw_plan_output": plan,
                                 "planner_output_override_used": True,
                                 "plan_override_used": False,
                                 "sft_events": [],
                                 "contract_protocol": {"plan_compiler": {"request_kind": task}},
                             }})
        write_jsonl(old_root / task / "results.jsonl", old_rows)
        write_jsonl(new_root / task / "results.jsonl", new_rows)

    report = compare(old_root, new_root, tmp_path / "report")
    assert report["passed"] is True
    assert report["training_authorized"] is False

    broken = json.loads((new_root / "shop" / "results.jsonl").read_text().splitlines()[0])
    broken["raw_trace"]["raw_plan_output"] = "changed"
    rows = [broken] + [json.loads(line) for line in
                       (new_root / "shop" / "results.jsonl").read_text().splitlines()[1:]]
    write_jsonl(new_root / "shop" / "results.jsonl", rows)
    assert compare(old_root, new_root, tmp_path / "report2")["identity_pass"] is False


def test_cli_returns_nonzero_when_the_gate_fails(monkeypatch, tmp_path):
    from scripts import compare_shoppingbench_classifier_only_online as module

    monkeypatch.setattr(module, "compare", lambda *args: {"baseline": {}, "candidate": {},
        "identity_pass": False, "routing_pass": True, "query_rewrite_zero": True,
        "outcome_nonregression_pass": True, "passed": False})
    monkeypatch.setattr("sys.argv", ["compare", "--baseline-root", str(tmp_path),
                                     "--candidate-root", str(tmp_path),
                                     "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
