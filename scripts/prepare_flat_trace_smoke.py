#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=32)
    parser.add_argument("--val-rows", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pd.read_parquet(args.source_dir / "flat_sft_train.parquet").head(args.train_rows).to_parquet(
        args.output_dir / "flat_sft_train.parquet", index=False
    )
    pd.read_parquet(args.source_dir / "flat_sft_val.parquet").head(args.val_rows).to_parquet(
        args.output_dir / "flat_sft_val.parquet", index=False
    )
    print({"output_dir": str(args.output_dir), "train_rows": args.train_rows, "val_rows": args.val_rows})


if __name__ == "__main__":
    main()
