"""Train Ziqi's Basic MF and FunkSVD models on the frozen data split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from goodbooks_mf.artifacts import verify_bundle
from goodbooks_mf.models import BasicMF, FunkSVD, rmse


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "goodreads-poetry-v1"


def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def evaluate(model, test: pd.DataFrame) -> dict:
    explicit = test[test["rating"] > 0]
    predictions = np.clip(
        model.predict(explicit["user_idx"], explicit["item_idx"]), 1, 5
    )
    return {
        "rmse": round(rmse(explicit["rating"], predictions), 6),
        "mae": round(mae(explicit["rating"], predictions), 6),
        "explicit_test_ratings": int(len(explicit)),
    }


def run(data_dir: Path, output_path: Path, smoke: bool = False) -> dict:
    manifest = verify_bundle(data_dir)
    train = pd.read_parquet(data_dir / "train.parquet")
    validation = pd.read_parquet(data_dir / "validation.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    if smoke:
        users = set(sorted(train["user_idx"].unique())[:100])
        train = train[train["user_idx"].isin(users)]
        validation = validation[validation["user_idx"].isin(users)]
        test = test[test["user_idx"].isin(users)]
    n_users = int(max(train["user_idx"].max(), validation["user_idx"].max(), test["user_idx"].max()) + 1)
    n_items = int(max(train["item_idx"].max(), validation["item_idx"].max(), test["item_idx"].max()) + 1)
    common = {
        "n_users": n_users,
        "n_items": n_items,
        "n_factors": 40 if not smoke else 8,
        "learning_rate": 0.01,
        "reg_lambda": 0.02,
        "n_epochs": 100 if not smoke else 5,
        "patience": 10,
        "seed": int(manifest["seed"]),
    }
    models = {
        "basic_mf": BasicMF(**common),
        "funksvd": FunkSVD(**common),
    }
    results = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(train, validation)
        results[name] = {
            **evaluate(model, test),
            "best_epoch": model.best_epoch,
            "epochs_trained": model.n_epochs_trained,
            "training_seconds": round(time.perf_counter() - started, 6),
            "hyperparameters": common,
        }
    payload = {
        "dataset_version": manifest["version"],
        "seed": manifest["seed"],
        "data_counts": manifest["counts"],
        "models": results,
        "ranking_metrics_status": "pending shared evaluation functions and candidate set",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ziqi_metrics.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output, args.smoke), indent=2))


if __name__ == "__main__":
    main()
