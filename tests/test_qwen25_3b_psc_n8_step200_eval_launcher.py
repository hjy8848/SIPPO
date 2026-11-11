from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "exp"
    / "evaluate_qwen25_3b_psc_n8_step200_cpv2_formal_train1000_a800_repl_gpu4_r1.sh"
)


def test_step200_eval_launcher_freezes_checkpoint_and_protocol():
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "STEP=200",
        "latest_checkpointed_iteration.txt",
        "EXPECTED_DATA_SIZE=1947",
        "EXPECTED_MODEL_SHARD_SIZE=6794448763",
        "EXPECTED_OPTIM_SHARD_SIZE=12343913955",
        "EXPECTED_EXTRA_STATE_SIZE=15077",
        "CONTRACT_PRESERVING_DAG_V1_REQUEST=1",
        "MAX_NODES_PER_TURN=0",
        "PLANNER_PROMPT_MAX_NODES=5",
        "CHAT_TEMPLATE_ENABLE_THINKING=false",
        "GPU_INDEX=4",
        "GPU-2a478145-88e8-4713-aaf9-1db5f593f4fb",
    ):
        assert required in text


def test_step200_eval_launcher_has_smoke_and_full_runtime_gates():
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "NUM_SAMPLES=16",
        '"${FORMAL_ROOT}/trajectories" "${RUN_NAME}" 1625',
        '"${TRAIN_ROOT}/trajectories" "${RUN_NAME}" 1000',
        "from qa_manager.ROSA.contract_protocol import CONTRACT_PROTOCOL_VERSION",
        "query_rewrite_events",
        "nonfinite_or_missing_metric_values",
        'payload.get("metadata_joined") != 1000',
        "FORMAL_SUMMARY_SHA=808830cdde1a6a3d8377d31a250170edc36369f6999a086416947a5a126b267d",
        "TRAIN_PARQUET_SHA=82f9f34699117a17fc8d53fc3c188f8c1e70e535ddda13c3b89d746532e25269",
        "TRAIN_METADATA_SHA=e3e6d982bdd3a0902c4926225146ef8fc425f904ffdcd3e32adadd6682bb2fa1",
    ):
        assert required in text


def test_step200_eval_launcher_writes_only_fresh_runtime_evidence():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "RUNTIME_ROOT=/tmp/h/q25psc-n8-step200-eval-r1" in text
    assert "Refusing to overwrite merged-model evidence" in text
    for path_name in ("MODEL_DIR", "SMOKE_ROOT", "FORMAL_ROOT", "TRAIN_ROOT"):
        assert f'--path-absent "${{{path_name}}}"' in text
