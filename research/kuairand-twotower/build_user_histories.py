"""Build leakage-safe user histories from prepared KuaiRand events."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kuairand_twotower.data import build_histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bounded user histories")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    histories = build_histories(pd.read_parquet(args.events), limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    histories.to_parquet(args.output, index=False)
    print(f"{len(histories)} request histories written to {args.output}")


if __name__ == "__main__":
    main()
