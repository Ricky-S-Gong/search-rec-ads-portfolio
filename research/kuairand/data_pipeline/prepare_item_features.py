"""Prepare static video metadata for the Item Tower."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def prepare_item_features(basic_path: Path, events_path: Path, output_path: Path) -> Path:
    features = pd.read_csv(basic_path).rename(columns={"video_id": "item_id"})
    if "item_id" not in features:
        raise ValueError("video metadata is missing video_id")
    if features["item_id"].duplicated().any():
        raise ValueError("video metadata has duplicate video_id values")

    event_items = pd.read_parquet(events_path, columns=["item_id"])["item_id"].unique()
    missing = set(event_items).difference(features["item_id"])
    if missing:
        raise ValueError(f"video metadata is missing {len(missing)} event item IDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KuaiRand Item Tower metadata")
    parser.add_argument("--basic", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_item_features(args.basic, args.events, args.output))


if __name__ == "__main__":
    main()
