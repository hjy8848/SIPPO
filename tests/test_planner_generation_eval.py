import importlib.util
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_dagent_planner_generation.py"
SPEC = importlib.util.spec_from_file_location("planner_generation_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cli_can_run_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--prompt-variant" in completed.stdout


def test_validator_aligned_prompt_discloses_runtime_constraints():
    prompt = MODULE.planner_prompt("validator_aligned")
    assert "at most 5 nodes" in prompt
    assert "Every <reason>" in prompt
    assert "must not consist only of a #ID#" in prompt
    assert "root <search>" in prompt


def test_single_constraint_variants_change_only_the_requested_rule():
    max_nodes = MODULE.planner_prompt("max_nodes")
    assert "at most 5 nodes" in max_nodes
    assert "Every <reason>" not in max_nodes
    assert "must not consist only of a #ID#" not in max_nodes

    reason_references = MODULE.planner_prompt("reason_references")
    assert "at most 5 nodes" not in reason_references
    assert "Every <reason>" in reason_references
    assert "must not consist only of a #ID#" not in reason_references

    concrete_search = MODULE.planner_prompt("concrete_search")
    assert "at most 5 nodes" not in concrete_search
    assert "Every <reason>" not in concrete_search
    assert "must not consist only of a #ID#" in concrete_search


def test_runtime_planner_prompt_matches_validated_max_node_variant():
    from qa_manager.ROSA import protocol

    assert protocol.PLANNER_SYSTEM_MSG == MODULE.planner_prompt("max_nodes") + "\n"


def test_classify_plan_matches_runtime_legality_contract():
    question = "Who is the father of Ada Lovelace's husband?"
    valid = MODULE.classify_plan(
        question,
        '<plan><search id="1">Who was Ada Lovelace\'s husband?</search>'
        '<search id="2">Who was the father of #1#?</search></plan>',
    )
    assert valid["valid"]
    assert valid["topology"] == "S-S"

    naked = MODULE.classify_plan(
        question,
        '<plan><search id="1">Ada Lovelace husband</search>'
        '<search id="2">#1#</search></plan>',
    )
    assert not naked["valid"]
    assert "naked_placeholder_search" in naked["errors"]

    reason_without_reference = MODULE.classify_plan(
        question,
        '<plan><search id="1">Ada Lovelace husband</search>'
        '<reason id="2">Select the answer.</reason></plan>',
    )
    assert not reason_without_reference["valid"]
    assert any("does not reference previous evidence" in error for error in reason_without_reference["errors"])


def test_classify_plan_rejects_hidden_node_limit_and_future_reference():
    question = "Compare two people."
    too_long = "<plan>" + "".join(
        f'<search id="{index}">Compare person fact {index}</search>'
        for index in range(1, 7)
    ) + "</plan>"
    result = MODULE.classify_plan(question, too_long)
    assert not result["valid"]
    assert any("maximum is 5" in error for error in result["errors"])

    future = MODULE.classify_plan(
        question,
        '<plan><search id="1">Compare #2#</search>'
        '<search id="2">Compare person two</search></plan>',
    )
    assert not future["valid"]
    assert "future_or_self_reference" in future["errors"]
    assert "missing_previous_reference" in future["errors"]
