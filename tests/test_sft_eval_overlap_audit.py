import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_dagent_sft_eval_overlap.py"
SPEC = importlib.util.spec_from_file_location("sft_eval_overlap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_formal(root, questions_by_dataset):
    root.mkdir()
    for name in MODULE.FORMAL_DATASETS:
        pd.DataFrame({"question": questions_by_dataset.get(name, [f"held out {name}"]) }).to_parquet(
            root / f"{name}.parquet", index=False
        )


def test_audit_accepts_disjoint_questions(tmp_path):
    train_path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "question": ["Train A", "Train B", "Train A"],
            "unified_source": ["new", "replay", "new"],
        }
    ).to_parquet(train_path, index=False)
    formal = tmp_path / "formal"
    write_formal(formal, {})

    report = MODULE.audit(train_path, formal, "unified_source", "new")

    assert report["train_unique_questions"] == 2
    assert report["causal_unique_questions"] == 1
    assert all(report["hard_gates"].values())


def test_audit_detects_normalized_overlap(tmp_path):
    train_path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "question": ["  SAME\u3000Question  "],
            "unified_source": ["new"],
        }
    ).to_parquet(train_path, index=False)
    formal = tmp_path / "formal"
    write_formal(formal, {"hotpotqa": ["same question"]})

    report = MODULE.audit(train_path, formal, "unified_source", "new")

    assert report["datasets"]["hotpotqa"]["all_train_exact_overlap"] == 1
    assert report["datasets"]["hotpotqa"]["causal_exact_overlap"] == 1
    assert not all(report["hard_gates"].values())
