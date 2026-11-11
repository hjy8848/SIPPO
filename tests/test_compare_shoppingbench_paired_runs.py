from scripts.compare_shoppingbench_paired_runs import (
    compare_runs,
    row_metrics,
    validate_runtime_contracts,
)


def make_row(sample_id, score, *, valid=True, retrieval_calls=1):
    return {
        "sample_id": sample_id,
        "query": f"query {sample_id}",
        "expected_product_ids": ["1"],
        "score": {
            "exact_id_fraction": score,
            "exact_sequence_success": score == 1,
            "first_id_exact": score == 1,
        },
        "format_error": None if valid else "bad format",
        "contract_error": None,
        "pipeline_error": None,
        "request_error": None,
        "retrieval_calls": retrieval_calls,
        "search_nodes": retrieval_calls,
    }


def make_runtime_row(*, contract, compiler_applied=False, pipeline_error=None):
    return {
        "pipeline_error": pipeline_error,
        "request_error": None,
        "raw_trace": {
            "protocol_profile": "shoppingbench",
            "plan_override_used": False,
            "ablation_config": {"contract_preserving_dag_v1": contract},
            "contract_protocol": {
                "plan_compiler": {
                    "enabled": contract,
                    "applied": compiler_applied,
                    "gold_used": False,
                }
            },
        }
    }


def test_runtime_gate_requires_live_compiler_without_gold_or_override():
    original = {
        "product-0000": make_runtime_row(contract=False),
        "product-0001": {"raw_trace": {"error": "invalid baseline plan"}},
    }
    contract = {
        "product-0000": make_runtime_row(contract=True, compiler_applied=True),
        "product-0001": make_runtime_row(contract=True, compiler_applied=True),
    }
    assert validate_runtime_contracts(original, contract)["passed"] is True

    contract["product-0000"]["raw_trace"]["contract_protocol"]["plan_compiler"][
        "gold_used"
    ] = True
    report = validate_runtime_contracts(original, contract)
    assert report["passed"] is False
    assert "compiler_used_no_gold" in report["mismatches"][0]


def test_runtime_gate_accepts_auditable_early_return_after_compilation():
    original = {"shop-0000": make_runtime_row(contract=False)}
    failed = make_runtime_row(
        contract=True,
        compiler_applied=True,
        pipeline_error="comparison question requires an explicit Reason node",
    )
    trace = failed["raw_trace"]
    trace.pop("protocol_profile")
    trace.pop("plan_override_used")
    trace.pop("ablation_config")
    report = validate_runtime_contracts(original, {"shop-0000": failed})
    assert report["passed"] is True
    assert report["path_counts"]["failed_after_compilation"] == 1


def test_runtime_gate_accepts_recorded_compiler_failure_without_execution():
    original = {"product-0000": make_runtime_row(contract=False)}
    failed = make_runtime_row(
        contract=True,
        pipeline_error="system plan compiler failed: no atomic Search query",
    )
    trace = failed["raw_trace"]
    trace["execution_error"] = failed["pipeline_error"]
    trace["nodes_execution"] = []
    trace.pop("protocol_profile")
    trace.pop("plan_override_used")
    trace.pop("ablation_config")
    report = validate_runtime_contracts(original, {"product-0000": failed})
    assert report["passed"] is True
    assert report["path_counts"]["failed_during_compilation"] == 1


def test_runtime_gate_rejects_failed_trace_with_contradictory_metadata():
    original = {"product-0000": make_runtime_row(contract=False)}
    failed = make_runtime_row(
        contract=True,
        compiler_applied=True,
        pipeline_error="comparison question requires an explicit Reason node",
    )
    failed["raw_trace"]["plan_override_used"] = True
    report = validate_runtime_contracts(original, {"product-0000": failed})
    assert report["passed"] is False
    assert any("plan_override_unused_not_contradicted" in item for item in report["mismatches"])


def test_strict_comparison_is_paired_and_reports_direction():
    original = {
        "product-0000": make_row("product-0000", 0, valid=False),
        "shop-0000": make_row("shop-0000", 0),
        "voucher-0000": make_row("voucher-0000", 1),
    }
    contract = {
        "product-0000": make_row("product-0000", 1),
        "shop-0000": make_row("shop-0000", 0),
        "voucher-0000": make_row("voucher-0000", 1),
    }
    report, rows = compare_runs(original, contract, bootstrap_samples=1000, seed=3)
    primary = report["overall"]["metrics"]["exact_id_fraction"]
    assert primary["paired_delta"] == 1 / 3
    assert primary["improved_pairs"] == 1
    assert primary["worsened_pairs"] == 0
    assert len(rows) == 3


def test_validity_counts_all_contract_and_execution_errors():
    row = make_row("product-0000", 0)
    assert row_metrics(row)["valid"] == 1
    row["pipeline_error"] = "failed"
    metrics = row_metrics(row)
    assert metrics["valid"] == 0
    assert metrics["execution_failure"] == 1
    assert metrics["pipeline_or_request_failure"] == 1


def test_execution_failure_includes_format_and_contract_failures():
    row = make_row("product-0000", 0)
    row["format_error"] = "expected one product_ids element"
    metrics = row_metrics(row)
    assert metrics["valid"] == 0
    assert metrics["execution_failure"] == 1
    assert metrics["pipeline_or_request_failure"] == 0
