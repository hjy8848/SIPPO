from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def read(name: str) -> str:
    return (EXP / name).read_text(encoding="utf-8")


def test_core_freezes_paper_matched_original4081_contract():
    text = read("run_dagent_qwen3_4b_original4081_sft_a800_repl_gpu5.sh")
    required = (
        "Qwen3-4B",
        "rosa_sft_deepseek_2wiki_full_v1",
        "db492678f48de62de95854bab613f4c79928f44b8d17e3dad4c9eca36261784f",
        "a5c05c1789fb95e1668317d1c0db1c396098fe5fad6351e7a6c8523300d8bd9d",
        '[[ "${train_rows}" -eq 4081 && "${val_rows}" -eq 224 ]]',
        "TRAIN_BATCH_SIZE=16",
        "MICRO_BATCH_SIZE=1",
        "TOTAL_EPOCHS=2",
        "formal_steps=\"$(( (train_rows / TRAIN_BATCH_SIZE) * TOTAL_EPOCHS ))\"",
        "LEARNING_RATE=1e-5",
        "MAX_LENGTH=2048",
        "data.multiturn.enable_thinking_key=enable_thinking",
        "model.enable_gradient_checkpointing=true",
        "use_remove_padding=true",
        "trainer.resume_mode=disable",
    )
    for value in required:
        assert value in text


def test_core_is_fail_closed_on_gpu_inputs_and_artifacts():
    text = read("run_dagent_qwen3_4b_original4081_sft_a800_repl_gpu5.sh")
    required = (
        "38lmdgu9bdc9m-0",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES=5",
        "GPU-05322139-8f57-32ce-692d-1aee157ba9b5",
        "GPU5 already has an NVML compute PID",
        "Refusing to overwrite Original4081 SFT output",
        "Refusing to reuse SFT cache",
        "TOTAL_TRAINING_STEPS must be 2 or 510",
        "DRY_RUN_OK",
    )
    for value in required:
        assert value in text
    for forbidden in ("pkill", "ray stop", "rm -", "kill "):
        assert forbidden not in text


def test_smoke_and_formal_wrappers_use_disjoint_fresh_artifacts():
    smoke = read("run_dagent_qwen3_4b_original4081_sft_smoke2_a800_repl_gpu5_r1.sh")
    formal = read("run_dagent_qwen3_4b_original4081_sft_full510_a800_repl_gpu5_r1.sh")

    assert "TOTAL_TRAINING_STEPS=2" in smoke
    assert "Original4081SFT-NoThink-smoke2-A800Repl-gpu5-r1" in smoke
    assert "q34o-sft-smk-g5-r1" in smoke
    assert "TOTAL_TRAINING_STEPS=510" in formal
    assert "Original4081SFT-NoThink-step510-A800Repl-gpu5-r1" in formal
    assert "q34o-sft-510-g5-r1" in formal
    assert smoke != formal
