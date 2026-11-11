#!/usr/bin/env python3
"""Freeze a balanced Reason-to-Search pilot plus direct-QA controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason-source", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reason-per-ref", type=int, default=24)
    parser.add_argument("--controls", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _reason_record(row: pd.Series) -> dict | None:
    extra = dict(row["extra_info"])
    searches = re.findall(
        r'<search id="([12])">(.*?)</search>',
        str(extra["gold_plan"]),
        flags=re.DOTALL,
    )
    if len(searches) != 2:
        return None
    candidates = [
        re.sub(r"\s+(?:date of birth|birth date|birth year)\s*$", "", task).strip()
        for _, task in searches
    ]
    supporting = _as_list(extra.get("supporting_titles"))
    if not supporting:
        return None
    target_entity = supporting[-1]
    matches = [
        index
        for index, candidate in enumerate(candidates, start=1)
        if candidate.casefold() in target_entity.casefold()
        or target_entity.casefold() in candidate.casefold()
    ]
    if len(matches) != 1:
        return None
    ground_truth = dict(row["reward_model"])["ground_truth"]
    return {
        "sample_index": f"typed_{extra['id']}",
        "question": str(extra["question"]),
        "golden_answers": _as_list(ground_truth["target"]),
        "source_dataset": "reason_search_anchor_v2_independent_test_v1",
        "source_split": "test",
        "protocol_slice": "reason_to_search",
        "source_id": str(extra["id"]),
        "oracle_selected_ref": str(matches[0]),
        "candidate_1": candidates[0],
        "candidate_2": candidates[1],
        "gold_plan": str(extra["gold_plan"]),
    }


def _is_direct_control(question: str) -> bool:
    blocked = re.compile(
        r"\b(?:closer|earlier|later|older|younger|both|same|between|compared|more|less)\b",
        re.IGNORECASE,
    )
    return blocked.search(question) is None


def _question_sha(frame: pd.DataFrame) -> str:
    payload = "\n".join(frame["question"].astype(str))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(
    reason_source: Path,
    control_source: Path,
    output_dir: Path,
    reason_per_ref: int,
    controls: int,
    seed: int,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot manifest: {output_dir}")
    output_dir.mkdir(parents=True)

    reason_frame = pd.read_parquet(reason_source)
    records = [record for _, row in reason_frame.iterrows() if (record := _reason_record(row))]
    rng = random.Random(seed)
    selected_reason: list[dict] = []
    for ref in ("1", "2"):
        candidates = sorted(
            (record for record in records if record["oracle_selected_ref"] == ref),
            key=lambda record: record["source_id"],
        )
        if len(candidates) < reason_per_ref:
            raise ValueError(f"Only {len(candidates)} usable reason rows for selected ref {ref}")
        selected_reason.extend(rng.sample(candidates, reason_per_ref))

    control_frame = pd.read_parquet(control_source)
    control_candidates = control_frame[
        control_frame["question"].astype(str).map(_is_direct_control)
    ].copy()
    if len(control_candidates) < controls:
        raise ValueError(f"Only {len(control_candidates)} direct controls are available")
    control_positions = rng.sample(list(control_candidates.index), controls)
    selected_controls = []
    for position in control_positions:
        row = control_candidates.loc[position]
        selected_controls.append(
            {
                "sample_index": f"control_{row['sample_index']}",
                "question": str(row["question"]),
                "golden_answers": _as_list(row["golden_answers"]),
                "source_dataset": str(row["source_dataset"]),
                "source_split": str(row["source_split"]),
                "protocol_slice": "direct_control",
                "source_id": str(row["sample_index"]),
                "oracle_selected_ref": "",
                "candidate_1": "",
                "candidate_2": "",
                "gold_plan": "",
            }
        )

    selected_reason.sort(key=lambda record: record["sample_index"])
    selected_controls.sort(key=lambda record: record["sample_index"])
    pilot = pd.DataFrame(selected_reason + selected_controls)
    pilot_path = output_dir / "typed_handoff_pilot.parquet"
    pilot.to_parquet(pilot_path, index=False)
    report = {
        "schema_version": 1,
        "hypothesis": (
            "Immutable source-reference handoff reduces Reason-to-Search identity loss "
            "without changing direct-QA behavior."
        ),
        "seed": seed,
        "reason_source": str(reason_source.resolve()),
        "control_source": str(control_source.resolve()),
        "counts": {
            "reason_to_search": len(selected_reason),
            "oracle_ref_1": sum(row["oracle_selected_ref"] == "1" for row in selected_reason),
            "oracle_ref_2": sum(row["oracle_selected_ref"] == "2" for row in selected_reason),
            "direct_control": len(selected_controls),
            "total": len(pilot),
        },
        "question_sha256": _question_sha(pilot),
        "decision_rule": {
            "primary": "typed reason_to_search EM must be strictly greater than original",
            "safety": [
                "typed direct_control EM and F1 must not be lower than original",
                "both variants must have zero pipeline/API errors",
                "typed handoff must reduce oracle-entity transfer errors on eligible workflows",
            ],
        },
        "selected_sample_indices": pilot["sample_index"].tolist(),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = build(
        args.reason_source,
        args.control_source,
        args.output_dir,
        args.reason_per_ref,
        args.controls,
        args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
