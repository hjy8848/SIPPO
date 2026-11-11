from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "exp" / "run_dagent_qwen3_17b_unified_sft_zju53_gpu5.sh"


def test_qwen3_sft_contract_is_explicit():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "Qwen/Qwen3-1.7B" in script
    assert "dagent_unified_sft_from_base_matched_exposure_v1" in script
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"' in script
    assert 'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-560}"' in script
    assert 'ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"' in script
    assert 'data.multiturn.enable_thinking_key=enable_thinking' in script
    assert 'model.attn_implementation="${ATTN_IMPLEMENTATION}"' in script
    assert "use_remove_padding=false" in script
    assert 'trainer.resume_mode=disable' in script
    assert 'trainer.checkpoint.save_contents=\'["hf_model"]\'' in script


def test_sft_trainer_attention_backend_is_configurable():
    trainer = (SCRIPT.parents[1] / "verl" / "trainer" / "fsdp_sft_trainer.py").read_text(encoding="utf-8")

    assert 'self.config.model.get("attn_implementation", "flash_attention_2")' in trainer
