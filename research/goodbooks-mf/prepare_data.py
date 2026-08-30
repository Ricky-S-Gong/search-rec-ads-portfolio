"""Build the canonical Goodreads Poetry dataset consumed by every MF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from goodbooks_mf.artifacts import write_bundle
from goodbooks_mf.data import (
    encode_ids,
    iterative_k_core,
    load_staged,
    sample_users,
    stage_to_sqlite,
)
from goodbooks_mf.split import per_user_temporal_split


ROOT = Path(__file__).parent
DEFAULT_RAW = ROOT / "data" / "raw" / "goodreads_interactions_poetry.json.gz"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "goodreads-poetry-v1"


def build(raw_path: Path, output_dir: Path, config: dict) -> dict:
    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw interactions: {raw_path}")
    staging_path = output_dir.parent / "poetry_staging.sqlite3"
    stage_to_sqlite(raw_path, staging_path)
    frame = load_staged(staging_path)
    filter_kwargs = {
        "min_user_interactions": int(config["min_user_interactions"]),
        "min_user_ratings": int(config["min_user_ratings"]),
        "min_item_interactions": int(config["min_item_interactions"]),
        "min_item_ratings": int(config["min_item_ratings"]),
    }
    frame = iterative_k_core(frame, **filter_kwargs)
    frame = sample_users(frame, config.get("max_users"), int(config["seed"]))
    frame = iterative_k_core(frame, **filter_kwargs)
    encoded, _, _ = encode_ids(frame)
    train, validation, test = per_user_temporal_split(
        encoded,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    combined = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    # Re-encode after cold-item filtering so the frozen matrices have no index gaps.
    combined = combined.drop(columns=["user_idx", "item_idx"])
    combined, users, items = encode_ids(combined)
    columns = [
        "user_idx", "item_idx", "rating", "is_read", "is_reviewed", "event_time", "split"
    ]
    manifest = write_bundle(
        output_dir,
        combined[columns],
        users,
        items,
        seed=int(config["seed"]),
        config=config,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build(args.raw, args.output, config)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
