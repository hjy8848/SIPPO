#!/usr/bin/env python3
"""Render selected paired traces into compact batches for human review."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any, Mapping


TITLE_RE = re.compile(r"\(Title:\s*([^\)]+)\)")


def normalize_answer(value: Any) -> str:
    text = str(value).casefold()
    text = "".join(character if character not in string.punctuation else " " for character in text)
    return " ".join(token for token in text.split() if token not in {"a", "an", "the"})


def evidence_flags(case: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, bool]:
    gold = [normalize_answer(value) for value in case.get("gold_answer") or []]
    contexts = normalize_answer(
        " ".join(str(node.get("search_context") or "") for node in arm.get("nodes_execution") or [])
    )
    conclusions = normalize_answer(
        " ".join(str(node.get("conclusion") or "") for node in arm.get("nodes_execution") or [])
    )
    prediction = normalize_answer(arm.get("prediction") or "")
    return {
        "gold_in_context": any(value and value in contexts for value in gold),
        "gold_in_conclusion": any(value and value in conclusions for value in gold),
        "prediction_is_gold_substring": any(
            prediction and (prediction in value or value in prediction) for value in gold
        ),
    }


def render_arm(name: str, arm: Mapping[str, Any], context_chars: int) -> list[str]:
    lines = [
        f"{name}: pred={arm.get('prediction')!r} EM={arm.get('em')} F1={arm.get('f1')} retrievals={arm.get('retrieval_calls')}",
        f"plan: {str(arm.get('plan') or '').replace(chr(10), ' ')}",
    ]
    for node in arm.get("nodes_execution") or []:
        context = str(node.get("search_context") or "")
        titles = ", ".join(TITLE_RE.findall(context))
        snippet = " ".join(context.split())[:context_chars]
        lines.append(
            f"node {node.get('id')} {str(node.get('type', '')).upper()}: "
            f"query={node.get('executed_query') or node.get('resolved_query') or node.get('query')!r}; "
            f"conclusion={node.get('conclusion')!r}; titles=[{titles}]; context={snippet}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--context-chars", type=int, default=900)
    parser.add_argument("--flags-only", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    for case in cases:
        index = int(str(case["case_id"])[1:])
        if not args.start <= index <= args.end:
            continue
        if args.flags_only:
            parts = [case["case_id"], case["dataset"], case["stratum"]]
            for name in ("pure", "psc"):
                arm = case[name]
                flags = evidence_flags(case, arm)
                parts.append(
                    f"{name}:em={arm['em']},f1={arm['f1']:.3f},ctx={int(flags['gold_in_context'])},"
                    f"conc={int(flags['gold_in_conclusion'])},alias={int(flags['prediction_is_gold_substring'])},"
                    f"pred={arm['prediction']!r}"
                )
            print(" | ".join(parts))
            continue
        print("=" * 100)
        print(f"{case['case_id']} {case['dataset']} {case['stratum']} weight={case['selection_weight']:.4f}")
        print(f"Q: {case['question']}")
        print(f"GOLD: {case['gold_answer']}")
        print("\n".join(render_arm("PURE", case["pure"], args.context_chars)))
        print("\n".join(render_arm("PSC", case["psc"], args.context_chars)))


if __name__ == "__main__":
    main()
