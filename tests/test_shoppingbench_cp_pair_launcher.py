from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "exp"
    / "evaluate_shoppingbench_cp_pair_qwen25_3b_zju47_gpu7_smoke_r3.sh"
)
R4_LAUNCHER = (
    ROOT
    / "exp"
    / "evaluate_shoppingbench_cp_pair_qwen25_3b_zju47_gpu3_smoke_r4.sh"
)
R5_LAUNCHER = (
    ROOT
    / "exp"
    / "evaluate_shoppingbench_cp_pair_qwen25_3b_zju47_gpu2_smoke_r5.sh"
)
R6_LAUNCHER = (
    ROOT
    / "exp"
    / "evaluate_shoppingbench_cp_pair_qwen25_3b_zju54_gpu0_smoke_r6.sh"
)
R7_LAUNCHER = (
    ROOT
    / "exp"
    / "evaluate_shoppingbench_cp_pair_qwen25_3b_zju47_gpu1_smoke_r7.sh"
)
R8_LAUNCHER = (
    ROOT
    / "exp"
    / "evaluate_shoppingbench_cp_pair_qwen25_3b_zju47_gpu4_full750_r8.sh"
)
R8_GPU5_TAKEOVER = (
    ROOT
    / "exp"
    / "evaluate_shoppingbench_cp_pair_qwen25_3b_zju47_gpu5_full750_r8_takeover.sh"
)
ARTIFACT_LIVE_LAUNCHER = (
    ROOT
    / "exp"
    / "evaluate_shoppingbench_artifact_runtime_qwen25_3b_zju53_gpu0_live_r4.sh"
)


def test_paired_launcher_freezes_one_model_and_two_explicit_arms():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Qwen2.5-3B-Instruct" in source
    assert 'ARMS="${ARMS:-original contract}"' in source
    assert 'for arm in ${ARMS}' in source
    assert '--arm "${arm}"' in source
    assert "GEN_TEMPERATURE=0" in source
    assert "external_shopping_api_calls" in source
    assert "SHOPPINGBENCH_LOCAL_ONLY=1" in source
    assert "PLANNER_API_URL=" in source
    assert "export JAVA_HOME" in source
    assert 'export PYTHONPATH="${JADE_ROOT}' in source
    assert "import fastapi, httpx, qa_manager" in source


def test_paired_launcher_requires_exclusive_gpu_and_unique_artifacts():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "GPU-6cc64992-8458-687c-1588-bb9016aa3f10" in source
    assert "--query-compute-apps=pid,process_name,used_memory" in source
    assert "Refusing to overwrite existing artifact" in source
    assert "Refusing existing cache" in source
    assert "kill -TERM" not in source
    assert "kill -KILL" not in source


def test_r4_wrapper_freezes_new_gpu_ports_tag_and_cache():
    source = R4_LAUNCHER.read_text(encoding="utf-8")
    assert "GPU_INDEX=3" in source
    assert "GPU-92e299e4-69dc-adf1-85e7-8bfc54898c2d" in source
    assert "RETRIEVER_PORT=5635" in source
    assert "ROSA_PORT=8181" in source
    assert "smoke-r4-artifact" in source
    assert "CACHE_TAG=sbpair-q25-r4g3" in source
    assert "kill -TERM" not in source


def test_r5_wrapper_freezes_gpu2_and_distinct_resources():
    source = R5_LAUNCHER.read_text(encoding="utf-8")
    assert "GPU_INDEX=2" in source
    assert "GPU-beefff4e-6ee0-c6d3-e551-d03f97adf855" in source
    assert "RETRIEVER_PORT=5637" in source
    assert "ROSA_PORT=8183" in source
    assert "gpu2-smoke-r5-artifact-repair" in source
    assert "CACHE_TAG=sbp-r5g2" in source
    assert "kill -TERM" not in source


def test_r6_wrapper_enables_only_the_system_plan_compiler():
    source = R6_LAUNCHER.read_text(encoding="utf-8")
    assert "TARGET_HOST=zju-54" in source
    assert "GPU_INDEX=0" in source
    assert "GPU-5cb72bea-f94f-3e32-ac79-b65e37014ad7" in source
    assert "RETRIEVER_PORT=5638" in source
    assert "ROSA_PORT=8184" in source
    assert "gpu0-smoke-r6-systemcompiler" in source
    assert "SHOPPING_SYSTEM_PLAN_COMPILER_V1=1" in source
    assert "kill -TERM" not in source


def test_r7_wrapper_moves_the_same_compiler_smoke_to_zju47_gpu1():
    source = R7_LAUNCHER.read_text(encoding="utf-8")
    assert "TARGET_HOST=zju-47" in source
    assert "GPU_INDEX=1" in source
    assert "GPU-34b2b2c9-8873-1ddd-6093-a0faeed6cba3" in source
    assert "RETRIEVER_PORT=5640" in source
    assert "ROSA_PORT=8186" in source
    assert "gpu1-smoke-r7-systemcompiler" in source
    assert "SHOPPING_SYSTEM_PLAN_COMPILER_V1=1" in source
    assert "kill -TERM" not in source


def test_base_launcher_uses_injected_unique_resources():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'GPU_INDEX="${GPU_INDEX:-7}"' in source
    assert 'RUN_TAG="${RUN_TAG:-ShoppingBench-' in source
    assert 'cache_path="/dev/shm/${CACHE_TAG}-${cache}"' in source
    assert 'TARGET_HOST="${TARGET_HOST:-zju-47}"' in source
    assert 'ARMS="${ARMS:-original contract}"' in source
    assert "plan_override_args" in source
    assert 'PLANNER_OUTPUT_OVERRIDE_MANIFEST="${PLANNER_OUTPUT_OVERRIDE_MANIFEST:-}"' in source
    assert '--planner-output-override-manifest "${PLANNER_OUTPUT_OVERRIDE_MANIFEST}"' in source
    assert "Plan and Planner-output overrides are mutually exclusive" in source
    assert 'EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-2}"' in source
    assert '--concurrency "${EVAL_CONCURRENCY}"' in source
    assert '"${SAMPLE_COUNT}" "${EVAL_CONCURRENCY}"' in source
    assert 'OUTPUT_SCOPE="${OUTPUT_SCOPE:-smoke}"' in source
    assert '"source_sha256"' in source
    assert '"${RUN_PURPOSE}" "${JADE_ROOT}"' in source


def test_r8_wrapper_freezes_full_750_pair_on_zju47_gpu4():
    source = R8_LAUNCHER.read_text(encoding="utf-8")
    assert 'TARGET_HOST="${TARGET_HOST:-zju-47}"' in source
    assert 'GPU_INDEX="${GPU_INDEX:-4}"' in source
    assert "GPU-f68525c2-8e8a-a678-e083-f973d9c93468" in source
    assert "SAMPLE_COUNT=250" in source
    assert "EVAL_CONCURRENCY=16" in source
    assert "OUTPUT_SCOPE=paired" in source
    assert "full750-r8-systemcompiler" in source
    assert "SHOPPING_SYSTEM_PLAN_COMPILER_V1=1" in source
    assert "compare_shoppingbench_paired_runs.py" in source
    assert "audit_shoppingbench_compiled_replay.py" in source
    assert '--expected-per-task 250' in source
    assert '--output-dir "${RUN_DIR}/comparison"' in source
    assert "comparison_status=$?" in source
    assert '--output "${RUN_DIR}/comparison/contract_failure_attribution.json"' in source
    assert 'exit "${comparison_status}"' in source
    assert "base_status=$?" in source
    assert 'summary_path = run_dir / "paired" / arm / task / "summary.json"' in source
    assert 'summary.get("completed") != expected' in source
    assert 'summary.get("pending") != 0' in source
    assert "kill -TERM" not in source


def test_r8_gpu5_takeover_changes_only_exclusive_resources():
    source = R8_GPU5_TAKEOVER.read_text(encoding="utf-8")
    assert "TARGET_HOST=zju-47" in source
    assert "GPU_INDEX=5" in source
    assert "GPU-00b6332c-0297-f3ae-1be9-3cc3de6fa522" in source
    assert "RETRIEVER_PORT=5642" in source
    assert "ROSA_PORT=8188" in source
    assert "CACHE_TAG=sbp-r8g5-takeover" in source
    assert "gpu4_full750_r8.sh" in source
    assert "RUN_TAG" not in source
    assert "SAMPLE_COUNT" not in source
    assert "EVAL_CONCURRENCY" not in source
    assert "kill -TERM" not in source


def test_live_artifact_wrapper_is_distinct_from_paired_smoke():
    source = ARTIFACT_LIVE_LAUNCHER.read_text(encoding="utf-8")
    assert "TARGET_HOST=zju-53" in source
    assert "GPU_INDEX=0" in source
    assert "GPU-fae7a2e0-ea2a-fd9b-b776-63399a0d7401" in source
    assert "ARMS=contract" in source
    assert "shoppingbench_plan_override_smoke_manifest_r1.json" in source
    assert "rosa-blackwell-cu128" in source
    assert "zju53-gpu0-live-r4" in source
