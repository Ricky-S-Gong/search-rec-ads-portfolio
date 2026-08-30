from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(actual_array - predicted_array))))


def _triplets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"user_idx", "item_idx", "rating"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing rating columns: {', '.join(sorted(missing))}")
    explicit = frame[frame["rating"] > 0]
    if explicit.empty:
        raise ValueError("at least one explicit rating is required")
    return (
        explicit["user_idx"].to_numpy(dtype=np.int64),
        explicit["item_idx"].to_numpy(dtype=np.int64),
        explicit["rating"].to_numpy(dtype=np.float64),
    )


@dataclass
class _Snapshot:
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_bias: np.ndarray | None = None
    item_bias: np.ndarray | None = None


class BasicMF:
    """Explicit-feedback matrix factorization trained over observed triplets."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        n_factors: int = 50,
        learning_rate: float = 0.01,
        reg_lambda: float = 0.01,
        n_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        seed: int = 42,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.seed = seed
        self.best_epoch: int | None = None
        self.n_epochs_trained = 0

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None) -> "BasicMF":
        users, items, ratings = _triplets(train)
        self._validate_indices(users, items)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (self.n_items, self.n_factors))
        best_loss = np.inf
        best_snapshot: _Snapshot | None = None
        stale_epochs = 0
        for epoch in range(self.n_epochs):
            for position in rng.permutation(len(ratings)):
                self._update(int(users[position]), int(items[position]), float(ratings[position]))
            self.n_epochs_trained = epoch + 1
            if validation is None:
                continue
            val_users, val_items, val_ratings = _triplets(validation)
            loss = rmse(val_ratings, self.predict(val_users, val_items))
            if loss < best_loss - self.min_delta:
                best_loss = loss
                self.best_epoch = epoch
                best_snapshot = self._snapshot()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_snapshot is not None:
            self._restore(best_snapshot)
        return self

    def _validate_indices(self, users: np.ndarray, items: np.ndarray) -> None:
        if users.min() < 0 or users.max() >= self.n_users:
            raise ValueError("user_idx is outside the configured range")
        if items.min() < 0 or items.max() >= self.n_items:
            raise ValueError("item_idx is outside the configured range")

    def _update(self, user: int, item: int, rating: float) -> None:
        user_vector = self.user_factors[user].copy()
        error = rating - float(user_vector @ self.item_factors[item])
        self.user_factors[user] += self.learning_rate * (
            error * self.item_factors[item] - self.reg_lambda * user_vector
        )
        self.item_factors[item] += self.learning_rate * (
            error * user_vector - self.reg_lambda * self.item_factors[item]
        )

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(self.user_factors.copy(), self.item_factors.copy())

    def _restore(self, snapshot: _Snapshot) -> None:
        self.user_factors = snapshot.user_factors
        self.item_factors = snapshot.item_factors

    def predict(self, user_idx: int | Iterable[int], item_idx: int | Iterable[int]):
        users = np.asarray(user_idx, dtype=np.int64)
        items = np.asarray(item_idx, dtype=np.int64)
        predictions = np.sum(self.user_factors[users] * self.item_factors[items], axis=-1)
        return float(predictions) if predictions.ndim == 0 else predictions

    def recommend(
        self,
        user_idx: int,
        candidate_item_idxs: Iterable[int],
        seen_item_idxs: Iterable[int] = (),
        k: int = 10,
    ) -> list[tuple[int, float]]:
        seen = set(seen_item_idxs)
        candidates = np.array(sorted(set(candidate_item_idxs).difference(seen)), dtype=np.int64)
        if not len(candidates):
            return []
        scores = np.asarray(self.predict(np.full(len(candidates), user_idx), candidates))
        order = np.lexsort((candidates, -scores))[:k]
        return [(int(candidates[index]), float(scores[index])) for index in order]


class FunkSVD(BasicMF):
    """Biased FunkSVD with global, user, and item intercepts."""

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None) -> "FunkSVD":
        _, _, ratings = _triplets(train)
        self.global_mean = float(ratings.mean())
        self.user_bias = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias = np.zeros(self.n_items, dtype=np.float64)
        return super().fit(train, validation)  # type: ignore[return-value]

    def _update(self, user: int, item: int, rating: float) -> None:
        user_vector = self.user_factors[user].copy()
        prediction = (
            self.global_mean
            + self.user_bias[user]
            + self.item_bias[item]
            + float(user_vector @ self.item_factors[item])
        )
        error = rating - prediction
        self.user_bias[user] += self.learning_rate * (
            error - self.reg_lambda * self.user_bias[user]
        )
        self.item_bias[item] += self.learning_rate * (
            error - self.reg_lambda * self.item_bias[item]
        )
        self.user_factors[user] += self.learning_rate * (
            error * self.item_factors[item] - self.reg_lambda * user_vector
        )
        self.item_factors[item] += self.learning_rate * (
            error * user_vector - self.reg_lambda * self.item_factors[item]
        )

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(
            self.user_factors.copy(),
            self.item_factors.copy(),
            self.user_bias.copy(),
            self.item_bias.copy(),
        )

    def _restore(self, snapshot: _Snapshot) -> None:
        super()._restore(snapshot)
        if snapshot.user_bias is not None and snapshot.item_bias is not None:
            self.user_bias = snapshot.user_bias
            self.item_bias = snapshot.item_bias

    def predict(self, user_idx: int | Iterable[int], item_idx: int | Iterable[int]):
        users = np.asarray(user_idx, dtype=np.int64)
        items = np.asarray(item_idx, dtype=np.int64)
        predictions = (
            self.global_mean
            + self.user_bias[users]
            + self.item_bias[items]
            + np.sum(self.user_factors[users] * self.item_factors[items], axis=-1)
        )
        return float(predictions) if predictions.ndim == 0 else predictions
