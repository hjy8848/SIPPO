import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/filter_dagent_run_to_questions.py"
SPEC = importlib.util.spec_from_file_location("filter_dagent_run_to_questions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_filter_aligns_extra_info_questions_and_preserves_source_index(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {"sample_index": 9, "question": "Question B?", "em": 0},
                {"sample_index": 4, "question": "Question A?", "em": 1},
            ]
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "subset.parquet"
    pd.DataFrame(
        {"extra_info": [{"question": "Question A?"}, {"question": "Question B?"}]}
    ).to_parquet(dataset, index=False)
    output = tmp_path / "nested/output.json"

    report = MODULE.filter_run(source, dataset, output)
    rows = json.loads(output.read_text(encoding="utf-8"))

    assert report["selected_rows"] == 2
    assert [row["question"] for row in rows] == ["Question A?", "Question B?"]
    assert [row["sample_index"] for row in rows] == [0, 1]
    assert [row["source_sample_index"] for row in rows] == [4, 9]


def test_filter_refuses_missing_duplicate_and_overwrite(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([{"sample_index": 0, "question": "Only question?", "em": 1}]),
        encoding="utf-8",
    )
    dataset = tmp_path / "subset.parquet"
    pd.DataFrame({"question": ["Missing question?"]}).to_parquet(dataset, index=False)
    output = tmp_path / "output.json"

    with pytest.raises(ValueError, match="missing question"):
        MODULE.filter_run(source, dataset, output)

    output.write_text("[]", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        MODULE.filter_run(source, dataset, output)


def test_filter_metadata_writes_only_subset_rows(tmp_path):
    source = tmp_path / "metadata.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"question": question, "value": value})
            for question, value in (("Question B?", 2), ("Question A?", 1))
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "subset.parquet"
    pd.DataFrame({"question": ["Question A?"]}).to_parquet(dataset, index=False)
    output = tmp_path / "metadata/subset.jsonl"

    report = MODULE.filter_metadata(source, dataset, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert report["selected_rows"] == 1
    assert rows == [{"question": "Question A?", "value": 1}]
