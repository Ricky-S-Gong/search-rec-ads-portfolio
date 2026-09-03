from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _paired_values(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    if actual_values.shape != predicted_values.shape or actual_values.size == 0:
        raise ValueError("actual and predicted values must have the same non-empty shape")
    return actual_values, predicted_values


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_values, predicted_values = _paired_values(actual, predicted)
    return float(np.sqrt(np.mean(np.square(actual_values - predicted_values))))


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_values, predicted_values = _paired_values(actual, predicted)
    return float(np.mean(np.abs(actual_values - predicted_values)))


def _top_k(recommended: Iterable[int], k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(recommended)[:k]


def precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    ranked = _top_k(recommended, k)
    return sum(item in relevant for item in ranked) / k


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    ranked = _top_k(recommended, k)
    return sum(item in relevant for item in ranked) / len(relevant)


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    ranked = _top_k(recommended, k)
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, item in enumerate(ranked)
        if item in relevant
    )
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / idcg


@dataclass(frozen=True)
class RankingEvaluationData:
    catalog: np.ndarray
    users: np.ndarray
    seen_by_user: dict[int, set[int]]
    relevant_by_user: dict[int, set[int]]


def evaluate_ratings(model, test: pd.DataFrame) -> dict[str, float | int]:
    required = {"user_idx", "item_idx", "rating"}
    missing = required.difference(test.columns)
    if missing:
        raise ValueError(f"missing rating columns: {', '.join(sorted(missing))}")
    explicit = test.loc[test["rating"] > 0]
    if explicit.empty:
        raise ValueError("at least one explicit test rating is required")
    predictions = np.clip(
        model.predict(explicit["user_idx"], explicit["item_idx"]), 1, 5
    )
    return {
        "rmse": rmse(explicit["rating"], predictions),
        "mae": mae(explicit["rating"], predictions),
        "evaluated_rating_count": int(len(explicit)),
        "evaluated_rating_users": int(explicit["user_idx"].nunique()),
    }


def prepare_ranking_data(train: pd.DataFrame, test: pd.DataFrame) -> RankingEvaluationData:
    train_required = {"user_idx", "item_idx"}
    test_required = {"user_idx", "item_idx", "rating", "is_read"}
    missing_train = train_required.difference(train.columns)
    missing_test = test_required.difference(test.columns)
    if missing_train or missing_test:
        missing = sorted(missing_train | missing_test)
        raise ValueError(f"missing ranking columns: {', '.join(missing)}")

    catalog = np.sort(train["item_idx"].unique().astype(np.int64))
    catalog_set = set(catalog.tolist())
    seen_by_user = {
        int(user): set(group["item_idx"].astype(int))
        for user, group in train.groupby("user_idx", observed=True)
    }
    relevant_rows = test.loc[
        (test["rating"] >= 4) | ((test["rating"] == 0) & test["is_read"].astype(bool))
    ]
    relevant_by_user: dict[int, set[int]] = {}
    for user, group in relevant_rows.groupby("user_idx", observed=True):
        user_id = int(user)
        relevant = (
            set(group["item_idx"].astype(int)) & catalog_set
        ) - seen_by_user.get(user_id, set())
        if relevant:
            relevant_by_user[user_id] = relevant
    users = np.array(sorted(relevant_by_user), dtype=np.int64)
    return RankingEvaluationData(catalog, users, seen_by_user, relevant_by_user)


def evaluate_ranking(
    model,
    ranking_data: RankingEvaluationData,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float | int | str]:
    if not k_values or any(k <= 0 for k in k_values) or len(set(k_values)) != len(k_values):
        raise ValueError("k_values must contain unique positive integers")

    totals = {metric: {k: 0.0 for k in k_values} for metric in ("precision", "recall", "ndcg")}
    max_k = max(k_values)
    for user in ranking_data.users:
        user_id = int(user)
        seen = ranking_data.seen_by_user.get(user_id, set())
        candidates = np.array(
            [item for item in ranking_data.catalog if int(item) not in seen],
            dtype=np.int64,
        )
        scores = np.asarray(
            model.predict(np.full(len(candidates), user_id, dtype=np.int64), candidates),
            dtype=np.float64,
        ).reshape(-1)
        if scores.shape != candidates.shape or not np.isfinite(scores).all():
            raise ValueError("model must return one finite score per candidate")
        order = np.lexsort((candidates, -scores))[:max_k]
        recommended = candidates[order].tolist()
        relevant = ranking_data.relevant_by_user[user_id]
        for k in k_values:
            totals["precision"][k] += precision_at_k(recommended, relevant, k)
            totals["recall"][k] += recall_at_k(recommended, relevant, k)
            totals["ndcg"][k] += ndcg_at_k(recommended, relevant, k)

    user_count = len(ranking_data.users)
    result: dict[str, float | int | str] = {
        "candidate_policy": "full_train_catalog_excluding_seen",
        "catalog_size": int(len(ranking_data.catalog)),
        "evaluated_ranking_users": user_count,
    }
    for metric, values in totals.items():
        for k, total in values.items():
            result[f"{metric}_at_{k}"] = total / user_count if user_count else 0.0
    return result
