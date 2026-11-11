from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cp_psc_smoke_launcher_freezes_protocol_and_two_step_budget():
    text = (
        ROOT / "exp" / "run_dagent_qwen25_3b_cp_psc_mixed15k_smoke2_zju57_r1.sh"
    ).read_text(encoding="utf-8")
    assert "CP_DAGENT_ENABLED=true" in text
    assert "PLANNER_PROMPT_MAX_NODES=5" in text
    assert "MAX_NODES_PER_TURN=0" in text
    assert "FINAL_ANSWER_RETRY_LIMIT=0" in text
    assert "TOTAL_TRAINING_STEPS=2" in text
    assert "PLANNER_POC_MODE=real" in text


def test_common_launcher_forwards_cp_contract_without_changing_legacy_default():
    text = (ROOT / "run_rosa_train_2x_rtx_pro_6000.sh").read_text(encoding="utf-8")
    assert 'PYTHON_BIN="${PYTHON_BIN:-}"' in text
    assert 'PYTHON_BIN="$(command -v python3 || command -v python || true)"' in text
    assert 'CP_DAGENT_ENABLED="${CP_DAGENT_ENABLED:-false}"' in text
    assert 'TYPED_SELECTION_CREDIT_ENABLED="${TYPED_SELECTION_CREDIT_ENABLED:-false}"' in text
    assert '+qa_manager.contract_preserving_dag_v2="${CP_DAGENT_ENABLED}"' in text
    assert '+qa_manager.planner_prompt_max_nodes="${PLANNER_PROMPT_MAX_NODES}"' in text
    assert 'algorithm.typed_selection_credit.enabled="${TYPED_SELECTION_CREDIT_ENABLED}"' in text
    assert 'algorithm.typed_selection_credit.alpha="${TYPED_SELECTION_CREDIT_ALPHA}"' in text
    assert 'PLANNER_POC_COVERAGE_AUDIT_ONLY="${PLANNER_POC_COVERAGE_AUDIT_ONLY:-false}"' in text
    assert 'algorithm.planner_outcome_contrast.coverage_audit_only="${PLANNER_POC_COVERAGE_AUDIT_ONLY}"' in text


def test_smoke_layer_preserves_explicit_cp_planner_limit_with_legacy_zero_default():
    text = (
        ROOT / "exp" / "run_dagent_qwen25_3b_psc_mixed15k_smoke2_zju57.sh"
    ).read_text(encoding="utf-8")
    assert 'PLANNER_PROMPT_MAX_NODES="${PLANNER_PROMPT_MAX_NODES:-0}"' in text
    assert "export PLANNER_PROMPT_MAX_NODES=0" not in text
    assert "planner_prompt_max_nodes=${PLANNER_PROMPT_MAX_NODES}" in text


def test_smoke_layer_preserves_explicit_psc_signal_gate_with_strict_default():
    text = (
        ROOT / "exp" / "run_dagent_qwen25_3b_psc_mixed15k_smoke2_zju57.sh"
    ).read_text(encoding="utf-8")
    assert 'PLANNER_POC_REQUIRE_SIGNAL="${PLANNER_POC_REQUIRE_SIGNAL:-true}"' in text
    assert "export PLANNER_POC_REQUIRE_SIGNAL=true" not in text
    assert '("qwen2", 2048, 36): "Qwen2.5-3B"' in text
    assert '("qwen3", 2560, 36): "Qwen3-4B"' in text


def test_smoke_layer_preserves_explicit_thinking_mode():
    text = (
        ROOT / "exp" / "run_dagent_qwen25_3b_psc_mixed15k_smoke2_zju57.sh"
    ).read_text(encoding="utf-8")
    assert 'MODEL_ENABLE_THINKING="${MODEL_ENABLE_THINKING:-null}"' in text
    assert "export MODEL_ENABLE_THINKING=null" not in text


def test_smoke_layer_accepts_multi_gpu_and_preserves_nccl_override():
    text = (
        ROOT / "exp" / "run_dagent_qwen25_3b_psc_mixed15k_smoke2_zju57.sh"
    ).read_text(encoding="utf-8")
    assert "requires exactly one physical GPU" not in text
    assert 'DAGENT_NCCL_P2P_DISABLE="${DAGENT_NCCL_P2P_DISABLE:-1}"' in text


def test_a800_typed_selection_smoke_freezes_uuid_contract_and_two_steps():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_cp_psc_typed_selection_smoke2_a800_gpu5_r1.sh"
    ).read_text(encoding="utf-8")
    assert "GPU-776efd61-282b-f136-a4db-7269d1b5f87a" in text
    assert "DAGENT_ALLOW_SHARED_GPU=false" in text
    assert "CP_DAGENT_ENABLED=true" in text
    assert "PLANNER_POC_MODE=real" in text
    assert "PLANNER_POC_LOSS_WEIGHT=0.02" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=true" in text
    assert "TYPED_SELECTION_CREDIT_ALPHA=0.2" in text
    assert "TYPED_SELECTION_CREDIT_REQUIRE_SIGNAL=true" in text
    assert "TOTAL_TRAINING_STEPS=2" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text
    assert "/tmp/hejingyu/" in text
    assert "/dev/shm" not in text


def test_a800_gpu7_r2_uses_new_uuid_tag_and_artifact_paths():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_cp_psc_typed_selection_smoke2_a800_gpu7_r2.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=7" in text
    assert "GPU-54f0dbd9-9a67-618d-044c-066fa4be80f6" in text
    assert "smoke2-a800-gpu7-r2" in text
    assert "q25-cppsc-ts-s2-g7-r2" in text
    assert "gpu5-r1" not in text
    assert "python311-dev-a800-r1/rootfs/usr/include" in text
    assert 'export CPATH="${python_header_root}/python3.11:${python_header_root}"' in text
    assert "TYPED_SELECTION_CREDIT_REQUIRE_SIGNAL=true" in text
    assert "TOTAL_TRAINING_STEPS=2" in text


def test_a800_gpu7_r3_preserves_failed_r2_and_uses_fresh_paths():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_cp_psc_typed_selection_smoke2_a800_gpu7_r3.sh"
    ).read_text(encoding="utf-8")
    assert "GPU-54f0dbd9-9a67-618d-044c-066fa4be80f6" in text
    assert "smoke2-a800-gpu7-r3" in text
    assert "q25-cppsc-ts-s2-g7-r3" in text
    assert "gpu7-r2" not in text
    assert "PLANNER_PROMPT_MAX_NODES=5" in text
    assert "PLANNER_POC_LOSS_WEIGHT=0.02" in text
    assert "TYPED_SELECTION_CREDIT_ALPHA=0.2" in text
    assert "TOTAL_TRAINING_STEPS=2" in text


def test_a800_gpu7_r4_uses_vanilla_instruct_and_disables_typed_selection():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_instruct_cp_psc_smoke2_a800_gpu7_r4.sh"
    ).read_text(encoding="utf-8")
    assert "MODEL_PATH=/ces124/models/Qwen2.5-3B-Instruct" in text
    assert "DeepSeek800" not in text
    assert "CP_DAGENT_ENABLED=true" in text
    assert "PLANNER_POC_MODE=real" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=true" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=2" in text
    assert "smoke2-a800-gpu7-r4" in text


def test_a800_gpu7_full468_is_independent_and_checkpointed():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_instruct_cp_psc_full468_a800_gpu7_r1.sh"
    ).read_text(encoding="utf-8")
    assert "MODEL_PATH=/ces124/models/Qwen2.5-3B-Instruct" in text
    assert "DeepSeek800" not in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=468" in text
    assert "SAVE_FREQ=50" in text
    assert "TEST_FREQ=-1" in text
    assert "validation_mixed_2wiki_hotpot_musique_seed42.parquet" in text
    assert "q25i-cppsc-f468-g7-r1" in text
    assert "smoke2-a800-gpu7-r4" not in text


def test_a800_gpu7_sparse_psc_full468_keeps_vanilla_rl_contract():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_instruct_cp_sparse_psc_full468_a800_gpu7_r2.sh"
    ).read_text(encoding="utf-8")
    assert "MODEL_PATH=/ces124/models/Qwen2.5-3B-Instruct" in text
    assert "DeepSeek800" not in text
    assert "CP_DAGENT_ENABLED=true" in text
    assert "PLANNER_POC_MODE=real" in text
    assert "PLANNER_POC_LOSS_WEIGHT=0.02" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=false" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=468" in text
    assert "SAVE_FREQ=50" in text
    assert "SparsePSC" in text
    assert "q25i-cpspsc-f468-g7-r2" in text


def test_a800_gpu7_psc_coverage_audit_never_updates_or_checkpoints():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_instruct_cp_psc_coverage_audit_a800_gpu7_r5.sh"
    ).read_text(encoding="utf-8")
    assert "MODEL_PATH=/ces124/models/Qwen2.5-3B-Instruct" in text
    assert "PLANNER_POC_MODE=real" in text
    assert "PLANNER_POC_MINIBATCH_MODE=question_preserving" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=false" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=true" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=1" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text
    assert "coverage-audit1-a800-gpu7-r5" in text

    trainer_text = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
        encoding="utf-8"
    )
    assert "if self.use_critic and not skip_planner_poc_coverage_audit_update:" in trainer_text
    assert 'metrics["critic/optimizer_steps_expected"] = 0.0' in trainer_text
    assert "and not skip_planner_poc_coverage_audit_update" in trainer_text
    assert 'metrics["actor/optimizer_steps_expected"] = 0.0' in trainer_text


def test_a800_gpu0_r6_changes_only_the_audit_gpu_and_artifact_identity():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_instruct_cp_psc_coverage_audit_a800_gpu0_r6.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=0" in text
    assert "GPU-1d28f249-ff08-9bde-fc97-dc8ae87b0ea1" in text
    assert "coverage-audit1-a800-gpu0-r6" in text
    assert "q25i-cppsc-ca1-g0-r6" in text
    assert "MODEL_PATH=/ces124/models/Qwen2.5-3B-Instruct" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=true" in text
    assert "TOTAL_TRAINING_STEPS=1" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text


def test_a800_gpu2_r7_keeps_the_zero_update_audit_contract():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen25_3b_instruct_cp_psc_coverage_audit_a800_gpu2_r7.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2" in text
    assert "GPU-672499bd-8bc3-a7ef-f804-90387489bd1c" in text
    assert "coverage-audit1-a800-gpu2-r7" in text
    assert "q25i-cppsc-ca1-g2-r7" in text
    assert "MODEL_PATH=/ces124/models/Qwen2.5-3B-Instruct" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=false" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=true" in text
    assert "TOTAL_TRAINING_STEPS=1" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text


def test_a800_qwen3_4b_r8_changes_only_model_specific_contract_fields():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_coverage_audit_a800_gpu2_r8.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2" in text
    assert "GPU-672499bd-8bc3-a7ef-f804-90387489bd1c" in text
    assert "MODEL_PATH=/ces124/models/Qwen3-4B" in text
    assert "validation_mixed_2wiki_hotpot_musique_seed42.parquet" in text
    assert "Qwen3-4B-CPv2-NoOut-PSC-Mixed15k-coverage-audit1-a800-gpu2-r8" in text
    assert "q34b-cppsc-ca1-g2-r8" in text
    assert "MODEL_ENABLE_THINKING=false" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=false" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=true" in text
    assert "TOTAL_TRAINING_STEPS=1" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text


def test_a800_qwen3_4b_r9_is_thinking_only_zero_update_audit():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_thinkingoff_coverage_audit_a800_gpu2_r9.sh"
    ).read_text(encoding="utf-8")
    assert "MODEL_PATH=/ces124/models/Qwen3-4B" in text
    assert "MODEL_ENABLE_THINKING=false" in text
    assert "PRINT_INVALID_ROLLOUTS=true" in text
    assert "q34b-cppsc-thinkingoff-ca1-g2-r9" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=true" in text
    assert "TOTAL_TRAINING_STEPS=1" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text


def test_a800_qwen3_4b_r10_only_repairs_ray_path_length():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_thinkingoff_coverage_audit_a800_gpu2_r10.sh"
    ).read_text(encoding="utf-8")
    assert "MODEL_PATH=/ces124/models/Qwen3-4B" in text
    assert "MODEL_ENABLE_THINKING=false" in text
    assert "PRINT_INVALID_ROLLOUTS=true" in text
    assert "cache_root=/tmp/h/q34r10" in text
    assert "gpu2-r10" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=true" in text
    assert "TOTAL_TRAINING_STEPS=1" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text


def test_a800_qwen3_4b_r11_is_typed_prompt_zero_update_audit():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_typedprompt_coverage_audit_a800_gpu2_r11.sh"
    ).read_text(encoding="utf-8")
    assert "MODEL_PATH=/ces124/models/Qwen3-4B" in text
    assert "MODEL_ENABLE_THINKING=false" in text
    assert "PRINT_INVALID_ROLLOUTS=true" in text
    assert "cache_root=/tmp/h/q34r11" in text
    assert "TypedPrompt" in text
    assert "gpu2-r11" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=true" in text
    assert "TOTAL_TRAINING_STEPS=1" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text


def test_a800_qwen3_4b_r12_changes_only_typed_selection_coverage():
    r11 = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_typedprompt_coverage_audit_a800_gpu2_r11.sh"
    ).read_text(encoding="utf-8")
    r12 = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_typedselection_coverage_audit_a800_gpu2_r12.sh"
    ).read_text(encoding="utf-8")

    assert "MODEL_PATH=/ces124/models/Qwen3-4B" in r12
    assert "MODEL_ENABLE_THINKING=false" in r12
    assert "TYPED_SELECTION_CREDIT_ENABLED=true" in r12
    assert "TYPED_SELECTION_CREDIT_ALPHA=0.2" in r12
    assert "TYPED_SELECTION_CREDIT_REQUIRE_SIGNAL=false" in r12
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=true" in r12
    assert "TOTAL_TRAINING_STEPS=1" in r12
    assert "SAVE_FREQ=-1" in r12
    assert "TEST_FREQ=-1" in r12
    assert "cache_root=/tmp/h/q34r12" in r12
    assert "gpu2-r12" in r12

    normalized_r11 = r11.replace(
        "TYPED_SELECTION_CREDIT_ENABLED=false",
        "TYPED_SELECTION_CREDIT_ENABLED=true",
    ).replace("TypedPrompt", "TypedSelection").replace("q34r11", "q34r12").replace(
        "gpu2-r11", "gpu2-r12"
    )
    normalized_r11 = normalized_r11.replace(
        "export TYPED_SELECTION_CREDIT_ENABLED=true\n",
        "export TYPED_SELECTION_CREDIT_ENABLED=true\n"
        "export TYPED_SELECTION_CREDIT_ALPHA=0.2\n",
    )
    assert r12 == normalized_r11


def test_a800_qwen3_4b_gpu7_r13_is_strict_two_update_poc_gate():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_smoke2_a800_gpu7_r13.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=7" in text
    assert "GPU-54f0dbd9-9a67-618d-044c-066fa4be80f6" in text
    assert "MODEL_PATH=/ces124/models/Qwen3-4B" in text
    assert "MODEL_ENABLE_THINKING=false" in text
    assert "PLANNER_POC_MODE=real" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=true" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=false" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=2" in text
    assert "SAVE_FREQ=-1" in text
    assert "TEST_FREQ=-1" in text
    assert "cache_root=/tmp/h/q34r13" in text
    assert "gpu7-r13" in text


def test_a800_qwen3_4b_gpu7_r14_is_independent_sparse_poc_full468():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_sparse_psc_full468_a800_gpu7_r14.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=7" in text
    assert "GPU-54f0dbd9-9a67-618d-044c-066fa4be80f6" in text
    assert "MODEL_PATH=/ces124/models/Qwen3-4B" in text
    assert "validation_mixed_2wiki_hotpot_musique_seed42.parquet" in text
    assert "MODEL_ENABLE_THINKING=false" in text
    assert "PLANNER_POC_MODE=real" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=false" in text
    assert "PLANNER_POC_COVERAGE_AUDIT_ONLY=false" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=468" in text
    assert "SAVE_FREQ=50" in text
    assert "TEST_FREQ=-1" in text
    assert "cache_root=/tmp/h/q34r14" in text
    assert "gpu7-r14" in text
    assert "gpu7-r13" not in text


def test_a800_replacement_gpu23_r15_is_strict_two_update_poc_gate():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_smoke2_a800_repl_gpu23_r15.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2,3" in text
    assert "GPU-71d2e625-f0d6-19eb-54c2-f5ec6192315b" in text
    assert "GPU-07bab419-405b-4f0b-2522-4561b10e4826" in text
    assert "rosa-vllm010-a800-repl-r1" in text
    assert "DAGENT_NCCL_P2P_DISABLE=0" in text
    assert "MODEL_PATH=/ces124/models/Qwen3-4B" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=true" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=2" in text
    assert "SAVE_FREQ=-1" in text
    assert "q34r15" in text
    assert "gpu23-r15" in text


def test_a800_replacement_gpu23_r16_is_independent_sparse_poc_full468():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_sparse_psc_full468_a800_repl_gpu23_r16.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2,3" in text
    assert "GPU-71d2e625-f0d6-19eb-54c2-f5ec6192315b" in text
    assert "GPU-07bab419-405b-4f0b-2522-4561b10e4826" in text
    assert "rosa-vllm010-a800-repl-r1" in text
    assert "DAGENT_NCCL_P2P_DISABLE=0" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=false" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=468" in text
    assert "SAVE_FREQ=50" in text
    assert "q34r16" in text
    assert "gpu23-r16" in text
    assert "gpu23-r15" not in text


def test_a800_replacement_gpu23_r17_adds_only_verified_compiler_contract():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_psc_smoke2_a800_repl_gpu23_r17.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2,3" in text
    assert "gcc12-a800-repl-r1" in text
    assert 'export CC="${toolchain_root}/bin/gcc"' in text
    assert 'export LD_LIBRARY_PATH="${toolchain_root}/lib/binutils' in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=true" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=2" in text
    assert "SAVE_FREQ=-1" in text
    assert "q34r17" in text
    assert "gpu23-r17" in text
    assert "gpu23-r15" not in text


def test_a800_replacement_gpu23_r18_is_compiler_enabled_full468():
    text = (
        ROOT
        / "exp"
        / "run_dagent_qwen3_4b_cp_sparse_psc_full468_a800_repl_gpu23_r18.sh"
    ).read_text(encoding="utf-8")
    assert "DAGENT_CUDA_VISIBLE_DEVICES=2,3" in text
    assert "gcc12-a800-repl-r1" in text
    assert 'export CC="${toolchain_root}/bin/gcc"' in text
    assert "PLANNER_POC_MODE=real" in text
    assert "PLANNER_POC_REQUIRE_SIGNAL=false" in text
    assert "TYPED_SELECTION_CREDIT_ENABLED=false" in text
    assert "TOTAL_TRAINING_STEPS=468" in text
    assert "SAVE_FREQ=50" in text
    assert "TEST_FREQ=-1" in text
    assert "q34r18" in text
    assert "gpu23-r18" in text
    assert "gpu23-r17" not in text
