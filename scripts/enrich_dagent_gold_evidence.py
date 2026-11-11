#!/usr/bin/env python3
"""Restore gold evidence titles in DAGENT mixed train/validation parquet files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Iterator
import unicodedata

import pandas as pd


def to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return to_python(value.as_py())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        return to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    return value


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def unique_text(values: Iterable[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "").strip().strip('"')
        key = unicodedata.normalize("NFKC", text).casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def evidence_key(kind: str, value: Any) -> str:
    normalized = normalize_question(value)
    return f"{kind}:{normalized}" if normalized else ""


def add_evidence(
    index: dict[str, list[str]],
    key: str,
    titles: list[str],
) -> int:
    if not key or not titles:
        return 0
    if key in index and index[key] != titles:
        return 1
    index[key] = titles
    return 0


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> bool:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        refill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if not refill():
                raise ValueError(f"empty JSON file: {path}")
        if buffer[position] != "[":
            raise ValueError(f"expected a top-level JSON array: {path}")
        position += 1

        while True:
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError(f"unterminated JSON array: {path}")
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                refill()
                continue
            if not isinstance(value, dict):
                raise ValueError(f"dataset item must be an object, got {type(value).__name__}")
            yield value
            position = end
            if position > chunk_size:
                refill()


def build_2wiki_index(
    arrow_paths: Iterable[Path],
    json_paths: Iterable[Path],
) -> tuple[dict[str, list[str]], int]:
    from datasets import Dataset

    index: dict[str, list[str]] = {}
    conflicts = 0
    for path in arrow_paths:
        dataset = Dataset.from_file(str(path))
        for row in dataset:
            metadata = to_python(row.get("metadata", {}))
            facts = metadata.get("supporting_facts", {}) if isinstance(metadata, dict) else {}
            titles = unique_text(facts.get("title", []) if isinstance(facts, dict) else [])
            conflicts += add_evidence(index, evidence_key("q", row.get("question")), titles)
    for path in json_paths:
        for row in iter_json_array(path):
            facts = to_python(row.get("supporting_facts", []))
            titles = unique_text(item[0] for item in facts if isinstance(item, (list, tuple)) and item)
            conflicts += add_evidence(index, evidence_key("id", row.get("_id", row.get("id"))), titles)
            conflicts += add_evidence(index, evidence_key("q", row.get("question")), titles)
    return index, conflicts


def build_musique_index(paths: Iterable[Path]) -> tuple[dict[str, list[str]], int]:
    from datasets import Dataset

    index: dict[str, list[str]] = {}
    conflicts = 0
    for path in paths:
        dataset = Dataset.from_file(str(path))
        for row in dataset:
            paragraphs = to_python(row.get("paragraphs", []))
            titles = unique_text(
                paragraph.get("title")
                for paragraph in paragraphs
                if isinstance(paragraph, dict) and paragraph.get("is_supporting")
            )
            conflicts += add_evidence(index, evidence_key("id", row.get("id")), titles)
            conflicts += add_evidence(index, evidence_key("q", row.get("question")), titles)
    return index, conflicts


def row_question(row: pd.Series) -> str:
    info = to_python(row.get("extra_info", {}))
    reward = to_python(row.get("reward_model", {}))
    if isinstance(info, dict) and info.get("question"):
        return str(info["question"])
    if isinstance(reward, dict) and reward.get("question"):
        return str(reward["question"])
    return ""


def enrich_frame(
    frame: pd.DataFrame,
    title_indices: dict[str, dict[str, list[str]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    updated_info = []
    updated_rewards = []
    counts: dict[str, Counter] = {}

    for _, row in output.iterrows():
        source = str(row.get("data_source", "")).strip().casefold()
        source = "2wiki" if source in {"2wiki", "2wikimultihopqa"} else source
        counts.setdefault(source, Counter())
        counts[source]["rows"] += 1

        info = to_python(row.get("extra_info", {}))
        reward = to_python(row.get("reward_model", {}))
        info = info if isinstance(info, dict) else {}
        reward = reward if isinstance(reward, dict) else {}
        ground_truth = to_python(reward.get("ground_truth", {}))
        ground_truth = ground_truth if isinstance(ground_truth, dict) else {}

        existing = unique_text(info.get("golden_titles", []))
        if existing and not all(title.isdigit() for title in existing):
            titles = existing
            counts[source]["preserved"] += 1
        else:
            raw_id = info.get("raw_id", info.get("id", ""))
            source_index = title_indices.get(source, {})
            titles = source_index.get(evidence_key("id", raw_id), [])
            if not titles:
                titles = source_index.get(evidence_key("q", row_question(row)), [])
            if titles:
                counts[source]["restored"] += 1
            else:
                counts[source]["missing"] += 1

        info["golden_titles"] = titles
        if ground_truth.get("type"):
            info["question_type"] = str(ground_truth["type"])
        if titles:
            info["gold_evidence_source"] = "official_supporting_facts"
        ground_truth["golden_titles"] = titles
        supporting = to_python(ground_truth.get("supporting_facts", {}))
        supporting = supporting if isinstance(supporting, dict) else {}
        supporting["title"] = titles
        ground_truth["supporting_facts"] = supporting
        reward["ground_truth"] = ground_truth
        updated_info.append(info)
        updated_rewards.append(reward)

    output["extra_info"] = updated_info
    output["reward_model"] = updated_rewards
    report = {source: dict(counter) for source, counter in sorted(counts.items())}
    return output, report


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--two-wiki-arrow", type=Path, action="append", default=[])
    parser.add_argument("--two-wiki-json", type=Path, action="append", default=[])
    parser.add_argument("--musique-arrow", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.input) != len(args.output):
        raise ValueError("--input and --output must be supplied the same number of times")
    for path in args.output:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {path}; pass --overwrite intentionally")

    two_wiki, two_wiki_conflicts = build_2wiki_index(args.two_wiki_arrow, args.two_wiki_json)
    musique, musique_conflicts = build_musique_index(args.musique_arrow)
    title_indices = {"2wiki": two_wiki, "musique": musique}
    report: dict[str, Any] = {
        "source_indices": {
            "2wiki_questions": len(two_wiki),
            "2wiki_conflicts": two_wiki_conflicts,
            "musique_questions": len(musique),
            "musique_conflicts": musique_conflicts,
        },
        "outputs": [],
    }
    for source_path, output_path in zip(args.input, args.output):
        enriched, counts = enrich_frame(pd.read_parquet(source_path), title_indices)
        atomic_parquet(enriched, output_path)
        report["outputs"].append(
            {"input": str(source_path), "output": str(output_path), "rows": len(enriched), "counts": counts}
        )

    report_path = args.output[0].with_suffix(".gold_evidence_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
