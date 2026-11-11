from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_first_driver_supports_causal_exact_planner_routing():
    script = (
        ROOT / "exp/evaluate_dagent_evidence_first_train1000_model.sh"
    ).read_text(encoding="utf-8")

    assert 'PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-}"' in script
    assert 'PLANNER_ROUTE_MODE="${PLANNER_ROUTE_MODE:-causal_exact}"' in script
    assert 'SPECIALIST_ROUTE_SCOPE="${SPECIALIST_ROUTE_SCOPE:-planner}"' in script
    assert 'export PLANNER_API_URL="http://127.0.0.1:${PLANNER_PORT}/v1/chat/completions"' in script
    assert "start_planner_server\nbash" in script
    assert '--served-model-name "${PLANNER_MODEL_NAME}"' in script
    assert "evaluation_metadata_subset.jsonl" in script
    assert "filter_dagent_run_to_questions.py" in script
    assert "export PLANNER_MODEL_NAME PLANNER_ROUTE_MODE SPECIALIST_ROUTE_SCOPE" in script


def test_evidence_first_driver_separates_main_and_planner_gpus():
    script = (
        ROOT / "exp/evaluate_dagent_evidence_first_train1000_model.sh"
    ).read_text(encoding="utf-8")

    assert '[[ "${PLANNER_CUDA_VISIBLE_DEVICES}" != "${EVAL_CUDA_VISIBLE_DEVICES}" ]]' in script
    assert 'CUDA_VISIBLE_DEVICES="${PLANNER_CUDA_VISIBLE_DEVICES}" setsid' in script
