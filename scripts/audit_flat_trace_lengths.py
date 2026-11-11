#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    model_path = Path(args.model).expanduser()
    print({"model_path": str(model_path), "is_dir": model_path.is_dir(), "has_tokenizer": (model_path / "tokenizer_config.json").is_file()})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    frame = pd.read_parquet(args.data)
    lengths = [
        len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        for messages in frame["messages"]
    ]
    ordered = sorted(lengths)
    print({
        "rows": len(lengths),
        "min": min(lengths),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[int(len(ordered) * 0.90)],
        "p95": ordered[int(len(ordered) * 0.95)],
        "max": max(lengths),
        "over_2048": sum(length > 2048 for length in lengths),
        "over_4096": sum(length > 4096 for length in lengths),
        "over_8192": sum(length > 8192 for length in lengths),
    })


if __name__ == "__main__":
    main()
