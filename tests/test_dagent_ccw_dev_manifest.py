from pathlib import Path

import pandas as pd
import pytest

from scripts.build_dagent_ccw_dev_manifest import SOURCE_TO_DATASET, build


def make_source(path: Path) -> None:
    rows = []
    for source in [*SOURCE_TO_DATASET, "dagent_reason_search_wiki18"]:
        for index in range(12):
            question = f"{source} question {index}"
            rows.append(
                {
                    "data_source": source,
                    "prompt": [{"role": "user", "content": question}],
                    "reward_model": {
                        "ground_truth": {"question": question, "target": [f"a{index}"]}
                    },
                    "extra_info": {"answer": [f"a{index}"]},
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_build_is_deterministic_and_excludes_reason_search(tmp_path):
    source = tmp_path / "validation.parquet"
    make_source(source)

    first = build(source, tmp_path / "first", count=5, seed=7)
    second = build(source, tmp_path / "second", count=5, seed=7)

    assert first["datasets"] == second["datasets"]
    assert first["excluded_sources"] == ["dagent_reason_search_wiki18"]
    for dataset in SOURCE_TO_DATASET.values():
        selected = pd.read_parquet(tmp_path / "first" / f"{dataset}.parquet")
        assert len(selected) == 5
        assert set(selected["data_source"]) == {dataset}


def test_build_refuses_overwrite_and_invalid_count(tmp_path):
    source = tmp_path / "validation.parquet"
    make_source(source)
    output = tmp_path / "manifest"
    build(source, output, count=3, seed=1)

    with pytest.raises(FileExistsError):
        build(source, output, count=3, seed=1)
    with pytest.raises(ValueError, match="positive"):
        build(source, tmp_path / "invalid", count=0, seed=1)


def test_build_can_exclude_a_previous_manifest(tmp_path):
    source = tmp_path / "validation.parquet"
    make_source(source)
    pilot_dir = tmp_path / "pilot"
    formal_dir = tmp_path / "formal"
    build(source, pilot_dir, count=4, seed=3)
    report = build(source, formal_dir, count=8, seed=4, exclude_dir=pilot_dir)

    for dataset in SOURCE_TO_DATASET.values():
        pilot = pd.read_parquet(pilot_dir / f"{dataset}.parquet")
        formal = pd.read_parquet(formal_dir / f"{dataset}.parquet")
        pilot_questions = {
            row[0]["content"] for row in pilot["prompt"].map(list)
        }
        formal_questions = {
            row[0]["content"] for row in formal["prompt"].map(list)
        }
        assert pilot_questions.isdisjoint(formal_questions)
        assert report["datasets"][dataset]["excluded_question_rows"] == 4
