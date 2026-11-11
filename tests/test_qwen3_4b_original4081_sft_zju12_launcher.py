from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"


def read(name: str) -> str:
    return (EXP / name).read_text(encoding="utf-8")


def test_zju12_core_freezes_original4081_and_dual_a100_contract():
    text = read("run_dagent_qwen3_4b_original4081_sft_zju12_gpu23.sh")
    required = (
        '[[ "$(hostname)" == "zju-12" ]]',
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        'export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${EXPECTED_GPU_UUIDS[*]}")"',
        "GPU-fee40a59-dd8f-50a2-37a5-717c5bed5bd7",
        "GPU-d4443f3c-9642-7ffb-8270-fad3c7586612",
        "Qwen3-4B",
        "rosa_sft_deepseek_2wiki_full_v1",
        "db492678f48de62de95854bab613f4c79928f44b8d17e3dad4c9eca36261784f",
        "a5c05c1789fb95e1668317d1c0db1c396098fe5fad6351e7a6c8523300d8bd9d",
        "TRAIN_BATCH_SIZE=16",
        "MICRO_BATCH_SIZE=1",
        "TOTAL_EPOCHS=2",
        "LEARNING_RATE=1e-5",
        "MAX_LENGTH=2048",
        "ATTN_IMPLEMENTATION=flash_attention_2",
        "CHECKPOINT_SAVE_CONTENTS='[\"hf_model\"]'",
        "--nproc_per_node=2",
        "trainer.n_gpus_per_node=2",
        "trainer.resume_mode=disable",
        "TOTAL_TRAINING_STEPS must be 2 or 510",
        "DRY_RUN_OK",
    )
    for value in required:
        assert value in text
    for forbidden in ("pkill", "ray stop", "rm -", "kill "):
        assert forbidden not in text


def test_zju12_smoke_and_formal_use_disjoint_fresh_artifacts():
    smoke = read("run_dagent_qwen3_4b_original4081_sft_smoke2_zju12_gpu23_r1.sh")
    formal = read("run_dagent_qwen3_4b_original4081_sft_full510_zju12_gpu23_r1.sh")

    assert "TOTAL_TRAINING_STEPS=2" in smoke
    assert "smoke2-A100-zju12-gpu23-r1" in smoke
    assert "q34o-sft-sm-g23-r1" in smoke
    assert "TOTAL_TRAINING_STEPS=510" in formal
    assert "step510-A100-zju12-gpu23-r1" in formal
    assert "q34o-sft-510-g23-r1" in formal
    assert smoke != formal


def test_zju12_smoke_r2_preserves_the_failed_r1_claim_identity():
    smoke = read("run_dagent_qwen3_4b_original4081_sft_smoke2_zju12_gpu23_r2.sh")

    assert "TOTAL_TRAINING_STEPS=2" in smoke
    assert "smoke2-A100-zju12-gpu23-r2" in smoke
    assert "q34o-sft-sm-g23-r2" in smoke
    assert "gpu23-r1" not in smoke
