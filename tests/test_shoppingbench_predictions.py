import json

import pytest

from scripts.run_shoppingbench_predictions import (
    append_journal,
    atomic_jsonl_dump,
    contract_flag,
    load_items,
    load_plan_overrides,
    request_prediction,
    score_ids,
)


def test_jsonl_writers_escape_unicode_line_separators(tmp_path):
    payload = {"sample_id": "x", "text": "before\u0085after\u2028tail"}
    results = tmp_path / "results.jsonl"
    journal = tmp_path / "journal.jsonl"

    atomic_jsonl_dump([payload], results)
    append_journal(journal, payload)

    for path in (results, journal):
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) == 1
        assert json.loads(text) == payload


def test_load_product_items_normalizes_single_reward(tmp_path):
    dataset = tmp_path / "product.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query": "Find a bronze cymbal",
                "reward": {"product_id": "591486855", "title": ["A cymbal"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    items = load_items(dataset, "product", 0, 0)
    assert items[0].sample_id == "product-0000"
    assert items[0].expected_product_ids == ["591486855"]


def test_score_ids_is_position_aligned_and_strict():
    score = score_ids(["22", "11"], ["11", "22"])
    assert score["matched_product_ids"] == 0
    assert score["exact_id_fraction"] == 0
    assert score["first_id_exact"] is False
    assert score["exact_sequence_success"] is False


def test_score_ids_accepts_exact_sequence():
    score = score_ids(["11", "22"], ["11", "22"])
    assert score["exact_id_fraction"] == 1
    assert score["first_id_exact"] is True
    assert score["exact_sequence_success"] is True


def test_product_first_id_metric_matches_official_evaluator():
    score = score_ids(["11", "99"], ["11"])
    assert score["first_id_exact"] is True
    assert score["exact_sequence_success"] is False


def test_contract_flag_freezes_paired_evaluation_arm():
    assert contract_flag("original") is False
    assert contract_flag("contract") is True
    assert contract_flag("default") is None


def test_plan_override_manifest_must_cover_selected_items(tmp_path):
    dataset = tmp_path / "product.jsonl"
    dataset.write_text(
        json.dumps({"query": "Find A", "reward": {"product_id": "1"}}) + "\n",
        encoding="utf-8",
    )
    items = load_items(dataset, "product", 0, 1)
    manifest = tmp_path / "plans.json"
    manifest.write_text(json.dumps({"product-0000": "<plan></plan>"}), encoding="utf-8")
    assert load_plan_overrides(manifest, items) == {"product-0000": "<plan></plan>"}


def test_request_includes_frozen_plan_override(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "protocol_profile": "shoppingbench",
                "ablation_config": {"contract_preserving_dag_v1": True},
                "plan_override_used": True,
                "planner_output_override_used": False,
                "final_answer": "<product_ids>1</product_ids>",
                "nodes_execution": [],
            }

    def fake_post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("scripts.run_shoppingbench_predictions.requests.post", fake_post)
    from scripts.run_shoppingbench_predictions import ShoppingBenchItem

    request_prediction(
        ShoppingBenchItem("product-0000", 0, "Find A", [{"product_id": "1"}], None),
        api_url="http://local",
        timeout=1,
        retries=0,
        arm="contract",
        plan_override="<plan sink=\"#1#\"></plan>",
    )
    assert captured["contract_preserving_dag_v1"] is True
    assert captured["plan_override"] == '<plan sink="#1#"></plan>'


def test_request_routes_frozen_planner_output_through_compiler(monkeypatch):
    captured = {}
    frozen = '  <plan sink="#1#"><search id="1">Find A</search></plan>\n'

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "protocol_profile": "shoppingbench",
                "ablation_config": {"contract_preserving_dag_v1": True},
                "plan_override_used": False,
                "planner_output_override_used": True,
                "raw_plan_output": frozen,
                "contract_protocol": {
                    "plan_compiler": {"enabled": True, "applied": True},
                    "replan_attempted": False,
                },
                "final_answer": "<product_ids>1</product_ids>",
                "nodes_execution": [],
            }

    def fake_post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("scripts.run_shoppingbench_predictions.requests.post", fake_post)
    from scripts.run_shoppingbench_predictions import ShoppingBenchItem

    result = request_prediction(
        ShoppingBenchItem("voucher-0000", 0, "Find A", [{"product_id": "1"}], None),
        api_url="http://local",
        timeout=1,
        retries=0,
        arm="contract",
        planner_output_override=frozen,
    )

    assert captured["planner_output_override"] == frozen
    assert "plan_override" not in captured
    assert result["contract_error"] is None


def test_request_rejects_both_override_modes():
    from scripts.run_shoppingbench_predictions import ShoppingBenchItem

    with pytest.raises(ValueError, match="mutually exclusive"):
        request_prediction(
            ShoppingBenchItem("voucher-0000", 0, "Find A", [{"product_id": "1"}], None),
            api_url="http://local",
            timeout=1,
            retries=0,
            arm="contract",
            plan_override="<plan></plan>",
            planner_output_override="<plan></plan>",
        )


def test_planner_output_manifest_preserves_raw_bytes(tmp_path):
    dataset = tmp_path / "voucher.jsonl"
    dataset.write_text(
        json.dumps({"query": "Find A", "reward": [{"product_id": "1"}]}) + "\n",
        encoding="utf-8",
    )
    items = load_items(dataset, "voucher", 0, 1)
    frozen = "  <plan></plan>\n"
    manifest = tmp_path / "raw_plans.json"
    manifest.write_text(json.dumps({"voucher-0000": frozen}), encoding="utf-8")

    assert load_plan_overrides(manifest, items, preserve_text=True) == {
        "voucher-0000": frozen
    }
