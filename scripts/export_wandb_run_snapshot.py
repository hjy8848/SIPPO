#!/usr/bin/env python3
"""Export a read-only W&B run snapshot for offline metric analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import wandb


def normalize_run_path(value: str) -> str:
    value = value.strip().strip("/")
    if "/runs/" in value:
        entity_project, run_id = value.split("/runs/", 1)
        value = f"{entity_project}/{run_id.split('/', 1)[0]}"
    parts = value.split("/")
    if len(parts) != 3:
        raise ValueError("run must be ENTITY/PROJECT/RUN_ID or a W&B run URL path")
    return value


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    private_dict = getattr(value, "_dict", None)
    if isinstance(private_dict, dict):
        return json_safe(private_dict)
    if callable(getattr(type(value), "tolist", None)):
        return json_safe(value.tolist())
    try:
        return json_safe(dict(value))
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="ENTITY/PROJECT/RUN_ID or URL-style run path")
    parser.add_argument("--base-url", default="https://api.bandw.top")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--page-size", type=int, default=500)
    args = parser.parse_args()

    run_path = normalize_run_path(args.run)
    run_id = run_path.rsplit("/", 1)[-1]
    output = args.output or Path("exp/run_logs") / f"wandb_{run_id}_snapshot.json"

    api = wandb.Api(overrides={"base_url": args.base_url})
    run = api.run(run_path)
    rows = [json_safe(dict(row)) for row in run.scan_history(page_size=args.page_size)]
    metric_keys = sorted({key for row in rows for key in row if not key.startswith("_")})

    snapshot = {
        "run_path": run_path,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "config": json_safe(dict(run.config)),
        "summary": json_safe(dict(run.summary)),
        "metric_keys": metric_keys,
        "history_rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    steps = [row.get("_step") for row in rows if isinstance(row.get("_step"), int)]
    print(f"[DONE] run={run_path} name={run.name} state={run.state}")
    print(f"[DONE] history_rows={len(rows)} max_step={max(steps) if steps else 'n/a'} metrics={len(metric_keys)}")
    print(f"[DONE] output={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
