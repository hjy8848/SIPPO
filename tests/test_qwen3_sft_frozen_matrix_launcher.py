from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "exp" / "evaluate_qwen3_17b_sft560_frozen_matrix.sh"


def test_frozen_matrix_allows_a_distinct_sft_label_and_summary_root():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'EVAL_MODEL_LABEL="${EVAL_MODEL_LABEL:-Qwen3-1.7B SFT560}"' in text
    assert 'SUMMARY_ROOT="${SUMMARY_ROOT:-${JADE_ROOT}/exp/qwen3_sft560_frozen_matrix}"' in text
    assert 'SUMMARY_DIR="${SUMMARY_ROOT}/${RUN_NAME}"' in text
    assert 'EVAL_LABEL="${EVAL_MODEL_LABEL} frozen baseline"' in text
    assert 'EVAL_LABEL="${EVAL_MODEL_LABEL} Train1000 diagnostic"' in text
    assert 'EVAL_LABEL="${EVAL_MODEL_LABEL} independent extension 138"' in text


def test_frozen_matrix_retains_the_frozen_three_suite_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "evaluate_dagent_formal_v2_model.sh" in text
    assert "evaluate_dagent_evidence_first_train1000_model.sh" in text
    assert "NUM_SAMPLES=138" in text
    assert '"2wikimultihopqa": 500' in text
    assert '"hotpotqa": 500' in text
    assert '"musique": 500' in text
    assert '"bamboogle": 125' in text
    assert 'float(payload.get("pipeline_error_rate", -1.0)) != 0.0' in text
