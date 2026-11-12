import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "exp" / "eval_rosa_with_trace.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_rosa_with_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expand_rollouts_preserves_question_identity_and_adds_repeat_id():
    module = load_module()
    source = [{"sample_index": "a", "question": "q", "answer": ["x"]}]
    expanded = module.expand_rollouts(source, 3)
    assert [row["rollout_index"] for row in expanded] == [0, 1, 2]
    assert {row["sample_index"] for row in expanded} == {"a"}
    assert source == [{"sample_index": "a", "question": "q", "answer": ["x"]}]


def test_resume_keeps_completed_rollouts_and_retries_only_api_failures(tmp_path):
    module = load_module()
    expected = module.expand_rollouts(
        [{"sample_index": "a", "question": "q", "answer": ["x"]}],
        3,
    )
    output = tmp_path / "results.json"
    output.write_text(
        json.dumps(
            [
                {
                    "sample_index": "a",
                    "rollout_index": 0,
                    "question": "q",
                    "error": None,
                },
                {
                    "sample_index": "a",
                    "rollout_index": 1,
                    "question": "q",
                    "error": "temporary network failure",
                },
            ]
        ),
        encoding="utf-8",
    )

    kept, pending = module.load_resumable_results(
        output,
        expected,
        retry_failed=True,
    )
    assert [module.rollout_key(row) for row in kept] == [("a", 0, "q")]
    assert {module.rollout_key(row) for row in pending} == {
        ("a", 1, "q"),
        ("a", 2, "q"),
    }


def test_resume_rejects_results_from_a_different_shard(tmp_path):
    module = load_module()
    expected = [{"sample_index": "a", "rollout_index": 0, "question": "q"}]
    output = tmp_path / "results.json"
    output.write_text(
        json.dumps(
            [{"sample_index": "b", "rollout_index": 0, "question": "other"}]
        ),
        encoding="utf-8",
    )

    try:
        module.load_resumable_results(output, expected, retry_failed=True)
    except ValueError as error:
        assert "Unexpected resumed rollout key" in str(error)
    else:
        raise AssertionError("Expected shard identity mismatch to be rejected")
