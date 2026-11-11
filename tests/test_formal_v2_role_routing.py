from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_driver_keeps_role_routing_optional():
    script = (ROOT / "exp/evaluate_dagent_formal_v2_model.sh").read_text(
        encoding="utf-8"
    )

    assert 'PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-}"' in script
    assert 'if [[ -n "${PLANNER_MODEL_PATH}" ]]' in script
    assert 'export PLANNER_API_URL="http://127.0.0.1:${PLANNER_PORT}/v1/chat/completions"' in script
    assert "start_planner_server\nstart_server_with_retry" in script
    assert '--served-model-name "${PLANNER_MODEL_NAME}"' in script
    assert 'CUDA_VISIBLE_DEVICES="${PLANNER_CUDA_VISIBLE_DEVICES}" setsid' in script
    assert 'PLANNER_ROUTE_MODE="${PLANNER_ROUTE_MODE:-all}"' in script
    assert "export PLANNER_ROUTE_MODE" in script
    assert 'SPECIALIST_ROUTE_SCOPE="${SPECIALIST_ROUTE_SCOPE:-planner}"' in script
    assert "export PLANNER_ROUTE_MODE SPECIALIST_ROUTE_SCOPE" in script
    assert 'export PLANNER_FALLBACK_TO_MAIN="${PLANNER_FALLBACK_TO_MAIN:-1}"' in script


def test_formal_driver_derives_counts_from_frozen_manifest():
    script = (ROOT / "exp/evaluate_dagent_formal_v2_model.sh").read_text(
        encoding="utf-8"
    )

    assert "print(len(pd.read_parquet(sys.argv[1])))" in script
    assert "expected=500" not in script
    assert '[[ "${dataset}" == "bamboogle" ]]' not in script


def test_formal_driver_supports_an_explicit_frozen_dataset_subset():
    script = (ROOT / "exp/evaluate_dagent_formal_v2_model.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'FORMAL_DATASETS="${FORMAL_DATASETS:-2wikimultihopqa hotpotqa musique bamboogle}"'
        in script
    )
    assert script.count('for dataset in "${FORMAL_DATASET_ARRAY[@]}"') == 3
    assert '"datasets": ${FORMAL_DATASETS_JSON}' in script
    assert "dataset_names = sys.argv[5:]" in script


def test_formal_driver_records_prompt_and_runtime_node_limits_separately():
    script = (ROOT / "exp/evaluate_dagent_formal_v2_model.sh").read_text(
        encoding="utf-8"
    )

    assert 'export MAX_NODES_PER_TURN="${MAX_NODES_PER_TURN:-5}"' in script
    assert (
        'export PLANNER_PROMPT_MAX_NODES="${PLANNER_PROMPT_MAX_NODES:-5}"'
        in script
    )
    assert '"planner_prompt_max_nodes": ${PLANNER_PROMPT_MAX_NODES}' in script
    assert '"runtime_max_nodes_per_turn": ${MAX_NODES_PER_TURN}' in script
    assert '"max_nodes_per_turn": 5' not in script
