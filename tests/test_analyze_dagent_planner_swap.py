from pathlib import Path

import pytest

from scripts.analyze_dagent_planner_swap import analyze_runs, render_markdown
from scripts.compare_dagent_formal_runs import FormalRun


def make_run(label: str, em_values: list[float], specialist: bool) -> FormalRun:
    rows = {}
    for index, em in enumerate(em_values):
        rows[index] = {
            "sample_index": index,
            "question": f"Question {index}",
            "gold_answer": [f"answer {index}"],
            "pred_answer": f"answer {index}" if em else "wrong",
            "em": em,
            "f1": em,
            "error": None,
            "pipeline_error": None,
            "search_steps": 2,
            "retrieval_calls": 2,
            "raw_trace": {
                "planner_routing": {"specialist_generated": specialist},
                "execution_error": "",
            },
        }
    return FormalRun(
        label=label,
        root=Path(f"{label}.json"),
        rows={"2wikimultihopqa": rows},
    )


def test_planner_swap_analysis_reports_type_effects_and_routing():
    reference = make_run("Pure", [0.0] + [1.0] * 499, False)
    candidate = make_run("Swap", [1.0] * 500, True)
    question_types = {
        f"question{index}": "bridge_comparison" if index == 0 else "comparison"
        for index in range(500)
    }

    report = analyze_runs(
        [reference, candidate],
        question_types,
        bootstrap_samples=100,
        seed=7,
    )

    assert report["runs"]["Swap"]["specialist_generated_rate"] == pytest.approx(1.0)
    bridge = report["comparisons"][0]["types"]["bridge_comparison"]
    assert bridge["em"]["delta"] == pytest.approx(1.0)
    assert "Official Question Types" in render_markdown(report)
