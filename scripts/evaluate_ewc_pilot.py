#!/usr/bin/env python3
"""Evaluate EWC instances through an overlay retriever and a JADE server."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests


def normalize(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def em(answer: object, gold: list[str]) -> int:
    value = normalize(answer)
    return int(bool(value) and any(value == normalize(item) for item in gold))


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def paired_summary(results: list[dict[str, object]]) -> dict[str, object]:
    """Summarize the three EWC worlds at the pair level.

    A state-changing world is successful only when its own answer is correct;
    merely emitting something different from the base-world answer is reported
    separately as sensitivity.  ``stale_source_answer_rate`` is deliberately
    an answer-level proxy: it measures changed worlds answered with the base
    world's gold answer, not an inferred claim about hidden model state.
    """
    expected_variants = {"base", "dependency_changed", "answer_preserving"}
    pairs: dict[str, dict[str, dict[str, object]]] = {}
    for row in results:
        pairs.setdefault(str(row["pair_id"]), {})[str(row["variant"])] = row

    complete = {
        pair_id: worlds
        for pair_id, worlds in pairs.items()
        if set(worlds) == expected_variants
    }
    incomplete = sorted(set(pairs) - set(complete))
    total = len(complete)
    base_correct = changed_correct = preserve_correct = joint_correct = 0
    changed_answer = unnecessary_change = preserve_invariant = stale_source = 0

    for worlds in complete.values():
        base = worlds["base"]
        changed = worlds["dependency_changed"]
        preserve = worlds["answer_preserving"]
        base_answer = normalize(base["final_answer"])
        changed_answer_text = normalize(changed["final_answer"])
        preserve_answer = normalize(preserve["final_answer"])
        base_correct += int(bool(base["em"]))
        changed_correct += int(bool(changed["em"]))
        preserve_correct += int(bool(preserve["em"]))
        joint_correct += int(bool(base["em"]) and bool(changed["em"]) and bool(preserve["em"]))
        changed_answer += int(bool(base_answer) and bool(changed_answer_text) and base_answer != changed_answer_text)
        unnecessary_change += int(bool(base_answer) and bool(preserve_answer) and base_answer != preserve_answer)
        preserve_invariant += int(bool(base_answer) and base_answer == preserve_answer)
        stale_source += int(
            bool(changed_answer_text)
            and any(changed_answer_text == normalize(answer) for answer in base["gold_answer"])
        )

    return {
        "pair_count": total,
        "incomplete_pair_ids": incomplete,
        "base_accuracy": ratio(base_correct, total),
        "dependency_changed_accuracy": ratio(changed_correct, total),
        "answer_preserving_accuracy": ratio(preserve_correct, total),
        "joint_accuracy": ratio(joint_correct, total),
        "changed_answer_rate": ratio(changed_answer, total),
        "unnecessary_answer_change_rate": ratio(unnecessary_change, total),
        "answer_preserving_invariance_rate": ratio(preserve_invariant, total),
        "stale_source_answer_rate": ratio(stale_source, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-instances", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.instances.read_text(encoding="utf-8").split("\n") if line.strip()]
    if args.max_instances < 0:
        raise SystemExit("--max-instances must be non-negative")
    if args.max_instances:
        rows = rows[: args.max_instances]
    if not rows:
        raise SystemExit("no instances selected")
    results = []
    for index, row in enumerate(rows, 1):
        selected = requests.post(
            args.retriever.rstrip("/") + "/select",
            json={"instance_id": row["instance_id"]},
            timeout=args.timeout,
        )
        selected.raise_for_status()
        response = requests.post(
            args.model.rstrip("/") + "/rosa/chat",
            json={"query": row["question"]},
            timeout=args.timeout,
        )
        response.raise_for_status()
        trace = response.json()
        final_answer = trace.get("final_answer", "")
        results.append(
            {
                "instance_id": row["instance_id"],
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "final_answer": final_answer,
                "em": em(final_answer, row["gold_answer"]),
                "pipeline_error": bool(trace.get("error")),
                "trace": trace,
            }
        )
        print(json.dumps({"completed": index, "total": len(rows), "variant": row["variant"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results) + "\n",
        encoding="utf-8",
    )
    summary = {
        "instances": len(results),
        "em": sum(row["em"] for row in results),
        "em_rate": sum(row["em"] for row in results) / len(results),
        "pipeline_errors": sum(row["pipeline_error"] for row in results),
        "by_variant": {
            variant: {
                "count": sum(row["variant"] == variant for row in results),
                "em": sum(row["variant"] == variant and row["em"] for row in results),
            }
            for variant in sorted({row["variant"] for row in results})
        },
        "paired": paired_summary(results),
    }
    args.output.with_name("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
