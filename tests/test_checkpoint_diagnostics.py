from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "exp" / "analyze_all_checkpoint_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_diagnostics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def trace(plan: str, **extra):
    return {
        "plan_output": plan,
        "nodes_execution": [],
        "plan_warnings": [],
        **extra,
    }


def test_plan_without_legacy_type_is_valid_and_gets_exact_topology():
    value = trace(
        '<plan><search id="1">A?</search><search id="2">B?</search>'
        '<reason id="3">Compare #1# and #2#.</reason>'
        '<search id="4">Find details about #3#.</search></plan>'
    )
    assert MODULE.strict_valid_plan(value) == (True, "valid")
    assert MODULE.get_plan_type(value) == "S-S-R-S"


def test_execution_error_is_not_silently_counted_as_valid():
    value = trace(
        '<plan><search id="1">A?</search>'
        '<reason id="2">Use #2#.</reason></plan>',
        execution_error="Node 2 references future/current node #2.",
    )
    assert MODULE.strict_valid_plan(value) == (
        False,
        "invalid_or_future_reference",
    )


def test_parser_rejects_extra_text_and_non_contiguous_ids():
    extra = trace('prefix <plan><search id="1">A?</search></plan>')
    assert MODULE.strict_valid_plan(extra) == (False, "no_parseable_nodes")
    skipped = trace(
        '<plan><search id="1">A?</search><search id="3">B?</search></plan>'
    )
    assert MODULE.strict_valid_plan(skipped) == (
        False,
        "non_contiguous_node_ids",
    )
