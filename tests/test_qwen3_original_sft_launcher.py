from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "exp" / "run_dagent_qwen3_17b_original_sft_zju12_gpu3.sh"


def test_qwen3_original_sft_launcher_freezes_historical_data_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Qwen/Qwen3-1.7B" in text
    assert "data/rosa_sft_deepseek_2wiki_full_v1" in text
    assert "db492678f48de62de95854bab613f4c79928f44b8d17e3dad4c9eca36261784f" in text
    assert "a5c05c1789fb95e1668317d1c0db1c396098fe5fad6351e7a6c8523300d8bd9d" in text
    assert '[[ "${train_rows}" -eq 4081 && "${val_rows}" -eq 224 ]]' in text
    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"' in text
    assert 'TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"' in text
    assert 'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-510}"' in text


def test_qwen3_original_sft_launcher_is_pinned_to_safe_gpu3_nothink_fa2():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"' in text
    assert '[[ "${CUDA_VISIBLE_DEVICES}" == "3" ]]' in text
    assert 'ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"' in text
    assert '[[ "${thinking_enabled}" -eq 0 ]]' in text
    assert "data.multiturn.enable_thinking_key=enable_thinking" in text
    assert "use_remove_padding=true" in text
    assert "trainer.resume_mode=disable" in text
    assert "Refusing to overwrite original-SFT output" in text
    assert '[[ "${DRY_RUN:-0}" == "1" ]]' in text
    assert "[DRY_RUN_OK] Original SFT contract passed all input gates." in text
