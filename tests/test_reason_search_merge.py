import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GEN_TEST = load("reason_search_test_helpers", ROOT / "tests/test_reason_search_dataset.py")
MERGE = load("reason_search_merge", ROOT / "scripts/merge_dagent_reason_search_datasets.py")


def test_independent_audit_accepts_grounded_record():
    record = GEN_TEST.MODULE.build_one(
        GEN_TEST.seed(), GEN_TEST.FakeLLM(), GEN_TEST.FakeRetriever()
    )
    assert MERGE.audit_record(record) is None


def test_independent_audit_rejects_compound_question():
    record = GEN_TEST.MODULE.build_one(
        GEN_TEST.seed(), GEN_TEST.FakeLLM(), GEN_TEST.FakeRetriever()
    )
    record["question"] = (
        "Which person was born earlier, Ada Alpha or Bea Beta, and where were they born?"
    )
    assert MERGE.audit_record(record) == "question requests winner and downstream answer"
