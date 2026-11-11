import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.filter_dagent_formal_run_to_manifest import DATASETS, filter_run


def make_inputs(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    trajectories = source / "trajectories"
    trajectories.mkdir(parents=True)
    manifest = root / "manifest"
    manifest.mkdir()
    for dataset in DATASETS:
        rows = [
            {
                "sample_index": index,
                "question": f"{dataset} question {index}",
                "gold_answer": [f"answer {index}"],
                "em": float(index % 2),
                "f1": float(index % 2),
            }
            for index in range(6)
        ]
        (trajectories / f"full_logs_source_{dataset}.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )
        pd.DataFrame(
            [
                {"sample_index": index, "question": f"{dataset} question {index}"}
                for index in (1, 4)
            ]
        ).to_parquet(manifest / f"{dataset}.parquet", index=False)
    return source, manifest


def test_filter_run_preserves_exact_manifest_alignment(tmp_path):
    source, manifest = make_inputs(tmp_path)
    output = tmp_path / "filtered"
    report = filter_run(source, manifest, output, "pilot")

    for dataset in DATASETS:
        path = output / "trajectories" / f"full_logs_pilot_{dataset}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert [row["sample_index"] for row in rows] == [0, 1]
        assert [row["source_sample_index"] for row in rows] == [1, 4]
        assert report["datasets"][dataset]["selected"] == 2


def test_filter_run_rejects_existing_output(tmp_path):
    source, manifest = make_inputs(tmp_path)
    output = tmp_path / "filtered"
    filter_run(source, manifest, output, "pilot")

    with pytest.raises(FileExistsError):
        filter_run(source, manifest, output, "pilot")
