from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.build_dagent_unified_em_rl_mix import build


def rl_row(source: str, question: str, row_id: str) -> dict:
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "fact-reasoning",
        "reward_model": {"ground_truth": {"target": ["answer"], "question": question}},
        "extra_info": {"id": row_id, "question": question},
    }


def write_parquet(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_builds_disjoint_exact_mix(tmp_path: Path) -> None:
    base_train = tmp_path / "base_train.parquet"
    base_validation = tmp_path / "base_validation.parquet"
    reason_train = tmp_path / "reason_train.parquet"
    reason_validation = tmp_path / "reason_validation.parquet"
    sft_train = tmp_path / "sft_train.parquet"
    sft_validation = tmp_path / "sft_validation.parquet"
    report_path = tmp_path / "sft_report.json"
    output = tmp_path / "output"

    write_parquet(
        base_train,
        [
            rl_row(source, f"base {source} {index}", f"base-{source}-{index}")
            for source, count in (("2wiki", 4), ("hotpotqa", 3), ("musique", 3))
            for index in range(count)
        ],
    )
    write_parquet(base_validation, [rl_row("2wiki", "base validation", "base-val")])
    write_parquet(
        reason_train,
        [rl_row("reason", f"reason {index}", f"reason-{index}") for index in range(4)],
    )
    write_parquet(reason_validation, [rl_row("reason", "reason validation", "reason-val")])
    pd.DataFrame(
        [
            {"question": "reason 0", "trajectory_id": "reason-0"},
            {"question": "unrelated replay", "trajectory_id": "replay-0"},
        ]
    ).to_parquet(sft_train, index=False)
    pd.DataFrame([{"question": "sft validation", "trajectory_id": "sft-val"}]).to_parquet(
        sft_validation, index=False
    )
    report_path.write_text(
        json.dumps({"selected_new_trajectory_ids": ["reason-0"]}), encoding="utf-8"
    )

    report = build(
        argparse.Namespace(
            base_train=base_train,
            base_validation=base_validation,
            reason_train=reason_train,
            reason_validation=reason_validation,
            sft_train=sft_train,
            sft_validation=sft_validation,
            sft_report=report_path,
            output_dir=output,
            base_rows=5,
            expected_reason_rows=3,
            seed=42,
        )
    )

    assert report["base_selected_counts"] == {"2wiki": 2, "hotpotqa": 1, "musique": 2}
    assert report["train_rows"] == 8
    assert report["validation_rows"] == 2
    assert all(report["hard_gates"].values())
    train = pd.read_parquet(output / "train.parquet")
    assert "reason 0" not in {value[0]["content"] for value in train["prompt"]}


def test_rejects_unexpected_remaining_reason_count(tmp_path: Path) -> None:
    base = tmp_path / "base.parquet"
    validation = tmp_path / "validation.parquet"
    reason = tmp_path / "reason.parquet"
    reason_validation = tmp_path / "reason_validation.parquet"
    sft = tmp_path / "sft.parquet"
    report_path = tmp_path / "report.json"
    write_parquet(base, [rl_row("2wiki", "base", "base")])
    write_parquet(validation, [rl_row("2wiki", "validation", "validation")])
    write_parquet(reason, [rl_row("reason", "reason 0", "reason-0"), rl_row("reason", "reason 1", "reason-1")])
    write_parquet(reason_validation, [rl_row("reason", "reason validation", "reason-val")])
    pd.DataFrame([{"question": "reason 0"}]).to_parquet(sft, index=False)
    report_path.write_text(json.dumps({"selected_new_trajectory_ids": ["reason-0"]}), encoding="utf-8")

    args = argparse.Namespace(
        base_train=base,
        base_validation=validation,
        reason_train=reason,
        reason_validation=reason_validation,
        sft_train=sft,
        sft_validation=sft,
        sft_report=report_path,
        output_dir=tmp_path / "output",
        base_rows=1,
        expected_reason_rows=2,
        seed=42,
    )
    try:
        build(args)
    except ValueError as error:
        assert "expected 2 unused reason rows, found 1" in str(error)
    else:
        raise AssertionError("expected a remaining-count failure")
