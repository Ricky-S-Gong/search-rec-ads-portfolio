"""Train Ziqi's Basic MF and FunkSVD models on the frozen data split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from goodbooks_mf.artifacts import verify_bundle
from goodbooks_mf.evaluation import evaluate_ranking, evaluate_ratings, prepare_ranking_data
from goodbooks_mf.models import BasicMF, FunkSVD


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "processed" / "goodreads-poetry-v1"


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
    ranking_data = prepare_ranking_data(train, test)
    results = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(train, validation)
        training_seconds = time.perf_counter() - started

        started = time.perf_counter()
        rating_metrics = evaluate_ratings(model, test)
        rating_inference_seconds = time.perf_counter() - started
        started = time.perf_counter()
        ranking_metrics = evaluate_ranking(model, ranking_data)
        ranking_inference_seconds = time.perf_counter() - started
        results[name] = {
            **rating_metrics,
            **ranking_metrics,
            "best_epoch": model.best_epoch,
            "epochs_trained": model.n_epochs_trained,
            "training_seconds": round(training_seconds, 6),
            "rating_inference_seconds": round(rating_inference_seconds, 6),
            "ranking_inference_seconds": round(ranking_inference_seconds, 6),
            "inference_seconds": round(
                rating_inference_seconds + ranking_inference_seconds,
                6,
            ),
            "hyperparameters": common,
        }
    payload = {
        "dataset_version": manifest["version"],
        "seed": manifest["seed"],
        "data_counts": manifest["counts"],
        "models": results,
        "ranking_protocol": {
            "candidate_policy": "full_train_catalog_excluding_seen",
            "relevance": "rating >= 4 OR (rating == 0 AND is_read)",
            "k_values": [5, 10, 20],
        },
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
