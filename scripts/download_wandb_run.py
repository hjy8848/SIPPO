#!/usr/bin/env python3
"""Download and summarize a W&B run for quick ROSA training checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUN = "genshinimpact114514ee-zhejiang-university/ROSA/2gqw1sgx"

IMPORTANT_KEYS = [
    "_step",
    "_runtime",
    "training/global_step",
    "train_metrics/final_reward",
    "train_metrics/final_f1",
    "train_metrics/final_em",
    "train_metrics/best_of_n_reward",
    "train_metrics/within_question_reward_std",
    "train_metrics/completed_workflows",
    "train_metrics/failed_question_groups",
    "train_metrics/turn_number",
    "train_metrics/retrieval_times",
    "critic/rewards/mean",
    "critic/rewards/max",
    "critic/rewards/min",
    "critic/score/mean",
    "critic/advantages/mean",
    "critic/advantages/max",
    "critic/advantages/min",
    "actor/entropy",
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/kl_loss",
    "actor/clip_ratio",
    "actor/lr",
    "response/aborted_ratio",
    "response_length/mean",
    "response_length/clip_ratio",
    "response_length_non_aborted/mean",
    "response_length_non_aborted/clip_ratio",
    "prompt_length/mean",
    "prompt_length/clip_ratio",
]


def normalize_run_path(run_path: str) -> str:
    """Accept W&B UI/API path variants and return entity/project/run_id."""
    run_path = run_path.strip()
    run_path = re.sub(r"^https?://wandb\.ai/", "", run_path)
    run_path = run_path.split("?", 1)[0].strip("/")
    parts = [part for part in run_path.split("/") if part]
    if len(parts) == 4 and parts[2] == "runs":
        return f"{parts[0]}/{parts[1]}/{parts[3]}"
    if len(parts) == 3:
        return "/".join(parts)
    raise ValueError(
        "Run path must look like entity/project/run_id, "
        "entity/project/runs/run_id, or a W&B run URL."
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def to_builtin(value: Any) -> Any:
    """Convert W&B/numpy-ish objects into JSON-safe Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    try:
        item = getattr(value, "item", None)
    except Exception:
        item = None
    if callable(item):
        try:
            return to_builtin(item())
        except Exception:
            pass
    return str(value)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value))


def last_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        value = row.get(key)
        if value is not None:
            return value
    return None


def numeric_tail(rows: list[dict[str, Any]], key: str, n: int = 20) -> list[float]:
    values: list[float] = []
    for row in reversed(rows):
        value = row.get(key)
        if is_number(value):
            values.append(float(value))
            if len(values) >= n:
                break
    return list(reversed(values))


def trend(values: list[float]) -> str:
    if len(values) < 2:
        return "unknown"
    delta = values[-1] - values[0]
    scale = max(1e-8, abs(values[0]), abs(values[-1]))
    if abs(delta) / scale < 0.03:
        return "flat"
    return "up" if delta > 0 else "down"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_builtin(v) for k, v in row.items()})


def print_key_table(rows: list[dict[str, Any]]) -> None:
    present = [key for key in IMPORTANT_KEYS if any(row.get(key) is not None for row in rows)]
    val_like = sorted(
        key
        for row in rows
        for key in row.keys()
        if key.startswith(("test_metrics/", "val/", "validation/")) and row.get(key) is not None
    )
    for key in val_like:
        if key not in present:
            present.append(key)

    print("\n重要指标最后值:")
    for key in present:
        values = numeric_tail(rows, key, n=20)
        value = last_value(rows, key)
        suffix = f"  trend={trend(values)}" if values else ""
        print(f"  {key}: {value}{suffix}")


def print_health_notes(rows: list[dict[str, Any]]) -> None:
    print("\n自动体检:")
    if not rows:
        print("  [BAD] 没有下载到 history 行。")
        return

    step = last_value(rows, "training/global_step")
    if step is None:
        step = last_value(rows, "_step")
    print(f"  [INFO] history rows={len(rows)}, latest step={step}")

    completed = last_value(rows, "train_metrics/completed_workflows")
    if is_number(completed):
        if float(completed) <= 0:
            print("  [BAD] completed_workflows <= 0，rollout 没有产出有效样本。")
        elif float(completed) < 16:
            print("  [WARN] completed_workflows 偏少，检查 rollout 失败或 batch 配置。")
        else:
            print("  [OK] completed_workflows 看起来有有效 rollout。")

    reward_values = numeric_tail(rows, "train_metrics/final_reward", n=20)
    if reward_values:
        if max(reward_values) == 0 and min(reward_values) == 0:
            print("  [BAD] 最近 final_reward 全是 0，训练信号可能无效。")
        elif max(reward_values) - min(reward_values) < 1e-6:
            print("  [WARN] 最近 final_reward 几乎没有变化，RLOO 方差可能不足。")
        else:
            print(f"  [OK] final_reward 有变化，趋势={trend(reward_values)}。")

    std_values = numeric_tail(rows, "train_metrics/within_question_reward_std", n=20)
    if std_values:
        if max(std_values) < 1e-6:
            print("  [BAD] within_question_reward_std 接近 0，K 条 workflow 没有形成相对优势。")
        elif sum(std_values) / len(std_values) < 0.02:
            print("  [WARN] within_question_reward_std 很低，RLOO 信号可能弱。")
        else:
            print("  [OK] within-question reward std 不为 0，RLOO 有比较信号。")

    entropy_values = numeric_tail(rows, "actor/entropy", n=20)
    if entropy_values:
        if entropy_values[-1] < 0.05:
            print("  [WARN] actor entropy 很低，策略可能过早塌缩。")
        else:
            print("  [OK] actor entropy 未见明显塌缩。")

    clip_values = numeric_tail(rows, "response_length/clip_ratio", n=20)
    if clip_values and max(clip_values) > 0.3:
        print("  [WARN] response_length/clip_ratio 偏高，回答可能经常被截断。")

    aborted = numeric_tail(rows, "response/aborted_ratio", n=20)
    if aborted and max(aborted) > 0.05:
        print("  [WARN] response/aborted_ratio > 5%，有生成失败或空回答风险。")


def download_history(run: Any, max_rows: int | None, page_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(run.scan_history(page_size=page_size)):
        rows.append(dict(row))
        if max_rows is not None and idx + 1 >= max_rows:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="W&B run path or URL.")
    parser.add_argument("--out-dir", default="wandb_runs", help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit history rows for quick checks.")
    parser.add_argument("--page-size", type=int, default=1000, help="W&B scan_history page size.")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Run: pip install wandb") from exc

    run_path = normalize_run_path(args.run)
    out_dir = Path(args.out_dir) / safe_name(run_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(run_path)

    metadata = {
        "path": run_path,
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "url": run.url,
        "created_at": str(getattr(run, "created_at", "")),
        "updated_at": str(getattr(run, "updated_at", "")),
    }
    (out_dir / "metadata.json").write_text(json.dumps(to_builtin(metadata), ensure_ascii=False, indent=2), "utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(to_builtin(dict(run.summary)), ensure_ascii=False, indent=2), "utf-8"
    )
    (out_dir / "config.json").write_text(json.dumps(to_builtin(dict(run.config)), ensure_ascii=False, indent=2), "utf-8")

    rows = download_history(run, max_rows=args.max_rows, page_size=args.page_size)
    jsonl_path = out_dir / "history.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")
    write_csv(out_dir / "history.csv", rows)

    print(f"Downloaded W&B run: {run_path}")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")
    print(f"Output: {out_dir.resolve()}")
    print_key_table(rows)
    print_health_notes(rows)


if __name__ == "__main__":
    main()
