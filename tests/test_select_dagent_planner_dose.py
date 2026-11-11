from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_dagent_planner_dose.py"
SPEC = importlib.util.spec_from_file_location("select_dagent_planner_dose", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def comparison(metric_values):
    return {"metrics": metric_values}


def candidate(label, semantic_ci, valid, reason_search):
    reasoner = comparison(
        {
            "semantic_chosen": {
                "candidate": 0.76,
                "delta": -0.001,
                "bootstrap_95ci": semantic_ci,
            }
        }
    )
    planner = comparison(
        {
            "valid": {"candidate": valid, "delta": valid - 0.951},
            "reason_search": {
                "candidate": reason_search,
                "delta": reason_search - 0.059,
            },
            "over_five_nodes": {"candidate": 0.01, "delta": -0.003},
        }
    )
    return MODULE.evaluate_candidate(
        label,
        reasoner,
        planner,
        planner_valid_baseline=0.951,
        matched_reason_search=1 / 1625,
    )


def test_candidate_must_pass_all_preregistered_gates():
    passing = candidate("dose28", [-0.01, 0.005], 0.96, 0.04)
    degraded = candidate("dose57", [-0.03, -0.01], 0.97, 0.06)
    assert passing["passes_all_gates"]
    assert not degraded["passes_all_gates"]
    assert not degraded["gates"]["reasoner_not_significantly_worse"]


def test_pareto_frontier_removes_dominated_candidates():
    first = candidate("dose0", [-0.01, 0.01], 0.96, 0.03)
    dominated = candidate("dose28", [-0.01, 0.01], 0.955, 0.02)
    tradeoff = candidate("dose57", [-0.01, 0.01], 0.958, 0.05)
    assert MODULE.pareto_frontier([first, dominated, tradeoff]) == [
        "dose0",
        "dose57",
    ]
