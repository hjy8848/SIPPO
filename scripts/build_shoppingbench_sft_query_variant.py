#!/usr/bin/env python3
"""Build a query-only ShoppingBench SFT-spec variant without full-title copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scripts.audit_shoppingbench_sft_query_alignment import normalize, summarize
from scripts.audit_shoppingbench_sft_spec_retrieval import atomic_queries_from_visible_query


TASKS = ("product", "shop", "voucher")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "an", "and", "for", "from", "in", "new", "of", "on", "or", "original",
    "the", "to", "with", "high", "quality", "best", "sale", "free", "pcs", "piece",
}
TEMPLATES = {
    "product": (
        "I'm looking for a catalog match.",
        "Could you help me locate an item?",
        "Please find a suitable product for me.",
        "I need an item with specific requirements.",
        "Can you search for a matching product?",
        "Show me an item fitting this description.",
        "I want to buy something like this.",
        "Find a catalog option that fits.",
    ),
    "shop": (
        "I need one seller for several items.",
        "Could you locate a shop carrying this list?",
        "Please find one store for all requested items.",
        "I want these products from a single seller.",
        "Can you match this list to one shop?",
        "Show me one store that carries everything.",
        "Help me source this group from one seller.",
        "Find a common shop for the following items.",
    ),
    "voucher": (
        "I want to use a voucher on a purchase.",
        "Could you find an item for this voucher order?",
        "Please match a product to these purchase terms.",
        "I need a product that works with my voucher.",
        "Can you search for this discounted purchase?",
        "Show me an item for the following checkout terms.",
        "Help me make a voucher-eligible purchase.",
        "Find a catalog option for this promotion.",
    ),
}


def reward_products(row: dict[str, Any]) -> list[dict[str, Any]]:
    reward = row.get("reward")
    return [reward] if isinstance(reward, dict) else list(reward or [])


def title_text(product: dict[str, Any]) -> str:
    title = product.get("title")
    if isinstance(title, list):
        return str(title[0]) if title else ""
    return str(title or "")


def title_tokens(title: str) -> list[str]:
    return TOKEN_RE.findall(title)


def document_frequency(rows: Iterable[dict[str, Any]]) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    total = 0
    for row in rows:
        for product in reward_products(row):
            tokens = {token.lower() for token in title_tokens(title_text(product))}
            if tokens:
                counts.update(tokens)
                total += 1
    return counts, total


def partial_descriptor(title: str, frequencies: Counter[str], total: int) -> str:
    tokens = title_tokens(title)
    if not tokens:
        return "requested merchandise"
    candidates = [
        (index, token)
        for index, token in enumerate(tokens)
        if token.lower() not in STOPWORDS
    ] or list(enumerate(tokens))
    target = max(1, min(5, (len(tokens) * 2 + 4) // 5))
    ranked = sorted(
        candidates,
        key=lambda item: (
            -math.log((total + 1) / (frequencies[item[1].lower()] + 1)),
            -len(item[1]),
            item[0],
        ),
    )
    selected = sorted(ranked[:target])
    descriptor = " ".join(token for _, token in selected)
    if normalize(descriptor) == normalize(title):
        return "requested merchandise"
    return descriptor


def visible_constraints(row: dict[str, Any]) -> list[str]:
    atoms = atomic_queries_from_visible_query(row)
    constraints = []
    for atom in atoms:
        marker = "; it should "
        constraint = atom.split(marker, 1)[1] if marker in atom else "match the requested specifications"
        constraints.append(constraint.strip(" ."))
    return constraints


def variant_query(
    row: dict[str, Any], frequencies: Counter[str], total: int
) -> tuple[str, list[str]]:
    task = str(row["task"])
    products = reward_products(row)
    constraints = visible_constraints(row)
    if len(products) != len(constraints):
        raise ValueError(f"{row.get('spec_id')}: product/constraint count mismatch")
    atoms = [
        f"{partial_descriptor(title_text(product), frequencies, total)}; it should {constraint}"
        for product, constraint in zip(products, constraints, strict=True)
    ]
    index = int(str(row["spec_id"]).rsplit("-", 1)[1]) % len(TEMPLATES[task])
    prefix = TEMPLATES[task][index]
    if task == "product":
        query = f"{prefix} Product details: {atoms[0]}."
    elif task == "shop":
        numbered = "; ".join(f"({i}) {atom}" for i, atom in enumerate(atoms, 1))
        query = f"{prefix} Product details: {numbered}."
    elif task == "voucher":
        voucher = row.get("voucher") or {}
        details = (
            atoms[0]
            if len(atoms) == 1
            else "; ".join(f"({i}) {atom}" for i, atom in enumerate(atoms, 1))
        )
        query = (
            f"{prefix} Product details: {details}. Purchase terms: budget "
            f"{voucher.get('budget')} Philippine pesos; platform voucher valid above "
            f"{voucher.get('threshold')} pesos with a fixed discount of "
            f"{voucher.get('face_value')} pesos."
        )
    else:
        raise ValueError(f"unsupported task: {task!r}")
    return query, atoms


def without_query(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"query", "query_variant"}}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(source_dir: Path, formal_test_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    source = {
        split: [json.loads(line) for line in (source_dir / f"{split}.jsonl").open(encoding="utf-8") if line.strip()]
        for split in ("train", "val")
    }
    frequencies, total = document_frequency(row for rows in source.values() for row in rows)
    transformed: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    atom_counts: Counter[str] = Counter()
    for split, rows in source.items():
        for row in rows:
            query, atoms = variant_query(row, frequencies, total)
            updated = dict(row)
            updated["query"] = query
            updated["query_variant"] = "partial_title_idf_query_only_r3a"
            transformed[split].append(updated)
            atom_counts[str(row["task"])] += len(atoms)

    formal_queries = set()
    for task in TASKS:
        path = formal_test_dir / f"synthesize_{task}_test.jsonl"
        formal_queries.update(
            normalize(json.loads(line).get("query")) for line in path.open(encoding="utf-8") if line.strip()
        )
    generated_queries = {
        normalize(row["query"]) for rows in transformed.values() for row in rows
    }
    metrics = {
        task: {
            split: summarize([row for row in transformed[split] if row["task"] == task])
            for split in ("train", "val")
        }
        for task in TASKS
    }
    gates = {
        "only_query_and_variant_metadata_changed": all(
            without_query(before) == without_query(after)
            for split in ("train", "val")
            for before, after in zip(source[split], transformed[split], strict=True)
        ),
        "full_title_rate_at_most_0_10": all(
            metrics[task][split]["any_gold_title_exact_substring_rate"] <= 0.10
            for task in TASKS for split in ("train", "val")
        ),
        "title_token_recall_at_most_0_70": all(
            metrics[task][split]["mean_gold_title_token_recall_in_query"] <= 0.70
            for task in TASKS for split in ("train", "val")
        ),
        "top_prefix_share_at_most_0_25": all(
            metrics[task][split]["top_five_token_prefix_share"] <= 0.25
            for task in TASKS for split in ("train", "val")
        ),
        "product_ids_absent_from_queries": all(
            str(product_id) not in row["query"]
            for rows in transformed.values() for row in rows
            for product_id in row["source_product_ids"]
        ),
        "formal_queries_disjoint": not bool(generated_queries & formal_queries),
    }
    if not all(gates.values()):
        raise ValueError(f"query-only variant gate failed: {gates}")
    output_dir.mkdir(parents=True)
    for split, rows in transformed.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    manifest = {
        "schema": "shoppingbench-cp-sft-query-variant-v1",
        "variant": "partial_title_idf_query_only_r3a",
        "source": str(source_dir),
        "rows": {split: len(rows) for split, rows in transformed.items()},
        "atomic_query_counts": dict(atom_counts),
        "gates": gates,
        "metrics": metrics,
        "formal_test_usage": "exact-query exclusion only; no formal title, answer, or query text generates the variant",
        "status": "query_style_ablation_only_not_training_data",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--formal-test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.source_dir, args.formal_test_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
