from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rating_metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape or actual_values.size == 0:
        raise ValueError("actual and predicted ratings must have the same non-empty shape")
    errors = predicted_values - actual_values
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }


def ranking_metrics(
    truth: dict[int, set[int]],
    recommendations: dict[int, list[int]],
    k: int = 10,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for user_id in sorted(truth):
        relevant = truth[user_id]
        if not relevant:
            continue
        ranked = recommendations.get(user_id, [])[:k]
        hits = [1 if item in relevant else 0 for item in ranked]
        recall = sum(hits) / len(relevant)
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_hits = min(k, len(relevant))
        idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
        rows.append({"user_id": user_id, "recall_at_10": recall, "ndcg_at_10": dcg / idcg})
    per_user = pd.DataFrame(rows, columns=["user_id", "recall_at_10", "ndcg_at_10"])
    if per_user.empty:
        return {"users": 0, "recall_at_10": 0.0, "ndcg_at_10": 0.0}, per_user
    return {
        "users": int(len(per_user)),
        "recall_at_10": float(per_user["recall_at_10"].mean()),
        "ndcg_at_10": float(per_user["ndcg_at_10"].mean()),
    }, per_user


def user_bootstrap_interval(
    per_user: pd.DataFrame,
    column: str,
    seed: int = 42,
    samples: int = 1_000,
) -> list[float]:
    values = per_user[column].to_numpy(dtype=float)
    if len(values) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def recommendation_stats(
    recommendations: dict[int, list[int]], item_counts: pd.Series, catalog_size: int
) -> dict:
    items = [item for values in recommendations.values() for item in values]
    if not items or catalog_size <= 0:
        return {"catalog_coverage": 0.0, "long_tail_share": 0.0, "novelty": 0.0}
    counts = item_counts.astype(float)
    head_size = max(1, int(math.ceil(len(counts) * 0.2)))
    head = set(counts.sort_values(ascending=False).head(head_size).index.astype(int))
    total = float(counts.sum())
    novelty = [
        -math.log2(max(float(counts.get(item, 0.0)), 1.0) / total)
        for item in items
    ]
    return {
        "catalog_coverage": len(set(items)) / catalog_size,
        "long_tail_share": sum(item not in head for item in items) / len(items),
        "novelty": float(np.mean(novelty)),
    }
