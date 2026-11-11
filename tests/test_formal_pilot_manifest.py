from pathlib import Path

import pandas as pd
import pytest

from scripts.build_dagent_formal_pilot_manifest import DEFAULT_COUNTS, build


def make_source(root: Path) -> None:
    root.mkdir()
    for dataset, requested in DEFAULT_COUNTS.items():
        rows = max(60, requested + 5)
        pd.DataFrame(
            {
                "question": [f"{dataset} question {index}" for index in range(rows)],
                "golden_answers": [[f"answer {index}"] for index in range(rows)],
            }
        ).to_parquet(root / f"{dataset}.parquet", index=False)


def test_build_is_deterministic_and_preserves_rows(tmp_path):
    source = tmp_path / "source"
    make_source(source)

    first = build(source, tmp_path / "first", seed=42)
    second = build(source, tmp_path / "second", seed=42)

    assert first["datasets"] == second["datasets"]
    for dataset, requested in DEFAULT_COUNTS.items():
        source_frame = pd.read_parquet(source / f"{dataset}.parquet")
        pilot = pd.read_parquet(tmp_path / "first" / f"{dataset}.parquet")
        assert len(pilot) == requested
        assert set(pilot["question"]).issubset(set(source_frame["question"]))


def test_build_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    output = tmp_path / "pilot"
    build(source, output, seed=42)

    with pytest.raises(FileExistsError):
        build(source, output, seed=42)
