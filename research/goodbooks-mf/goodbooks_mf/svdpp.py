from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .evaluation import rmse


@dataclass
class _Snapshot:
    user_factors: np.ndarray
    item_factors: np.ndarray
    implicit_factors: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray


class SVDPP:
    """Explicit-feedback SVD++ with train-only implicit histories."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        n_factors: int = 40,
        learning_rate: float = 0.01,
        reg_lambda: float = 0.02,
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

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None) -> "SVDPP":
        required = {"user_idx", "item_idx", "rating", "is_read", "is_reviewed"}
        missing = required.difference(train.columns)
        if missing:
            raise ValueError(f"missing SVD++ columns: {', '.join(sorted(missing))}")
        explicit = train.loc[train["rating"] > 0]
        if explicit.empty:
            raise ValueError("at least one explicit rating is required")
        users = explicit["user_idx"].to_numpy(dtype=np.int64)
        items = explicit["item_idx"].to_numpy(dtype=np.int64)
        ratings = explicit["rating"].to_numpy(dtype=np.float64)
        self._validate_indices(users, items)
        self._build_histories(train)

        rng = np.random.default_rng(self.seed)
        self.global_mean = float(ratings.mean())
        self.user_bias = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias = np.zeros(self.n_items, dtype=np.float64)
        self.user_factors = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (self.n_items, self.n_factors))
        self.implicit_factors = rng.normal(0, 0.1, (self.n_items, self.n_factors))

        best_loss = np.inf
        best_snapshot: _Snapshot | None = None
        stale_epochs = 0
        for epoch in range(self.n_epochs):
            for position in rng.permutation(len(ratings)):
                self._update(int(users[position]), int(items[position]), float(ratings[position]))
            self.n_epochs_trained = epoch + 1
            self._refresh_user_representations()
            if validation is None:
                continue
            validation_ratings = validation.loc[validation["rating"] > 0]
            if validation_ratings.empty:
                raise ValueError("at least one explicit validation rating is required")
            predictions = np.clip(
                self.predict(validation_ratings["user_idx"], validation_ratings["item_idx"]),
                1,
                5,
            )
            loss = rmse(validation_ratings["rating"], predictions)
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
        self._refresh_user_representations()
        return self

    def _build_histories(self, train: pd.DataFrame) -> None:
        implicit = train.loc[
            train["is_read"].astype(bool)
            | train["is_reviewed"].astype(bool)
            | train["rating"].gt(0)
        ]
        self.implicit_history = [np.empty(0, dtype=np.int64) for _ in range(self.n_users)]
        for user, group in implicit.groupby("user_idx", observed=True):
            user_id = int(user)
            if user_id < 0 or user_id >= self.n_users:
                raise ValueError("user_idx is outside the configured range")
            history = np.sort(group["item_idx"].unique().astype(np.int64))
            if len(history) and (history[0] < 0 or history[-1] >= self.n_items):
                raise ValueError("item_idx is outside the configured range")
            self.implicit_history[user_id] = history

    def _validate_indices(self, users: np.ndarray, items: np.ndarray) -> None:
        if users.min() < 0 or users.max() >= self.n_users:
            raise ValueError("user_idx is outside the configured range")
        if items.min() < 0 or items.max() >= self.n_items:
            raise ValueError("item_idx is outside the configured range")

    def _implicit_vector(self, user: int) -> np.ndarray:
        history = self.implicit_history[user]
        if not len(history):
            return np.zeros(self.n_factors, dtype=np.float64)
        return self.implicit_factors[history].sum(axis=0) / np.sqrt(len(history))

    def _update(self, user: int, item: int, rating: float) -> None:
        history = self.implicit_history[user]
        implicit_vector = self._implicit_vector(user)
        user_vector = self.user_factors[user].copy()
        item_vector = self.item_factors[item].copy()
        prediction = (
            self.global_mean
            + self.user_bias[user]
            + self.item_bias[item]
            + float(item_vector @ (user_vector + implicit_vector))
        )
        error = rating - prediction
        self.user_bias[user] += self.learning_rate * (
            error - self.reg_lambda * self.user_bias[user]
        )
        self.item_bias[item] += self.learning_rate * (
            error - self.reg_lambda * self.item_bias[item]
        )
        self.user_factors[user] += self.learning_rate * (
            error * item_vector - self.reg_lambda * user_vector
        )
        self.item_factors[item] += self.learning_rate * (
            error * (user_vector + implicit_vector) - self.reg_lambda * item_vector
        )
        if len(history):
            history_factors = self.implicit_factors[history].copy()
            self.implicit_factors[history] += self.learning_rate * (
                error * item_vector / np.sqrt(len(history))
                - self.reg_lambda * history_factors
            )

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(
            self.user_factors.copy(),
            self.item_factors.copy(),
            self.implicit_factors.copy(),
            self.user_bias.copy(),
            self.item_bias.copy(),
        )

    def _restore(self, snapshot: _Snapshot) -> None:
        self.user_factors = snapshot.user_factors
        self.item_factors = snapshot.item_factors
        self.implicit_factors = snapshot.implicit_factors
        self.user_bias = snapshot.user_bias
        self.item_bias = snapshot.item_bias

    def _refresh_user_representations(self) -> None:
        self.user_representations = self.user_factors.copy()
        for user in range(self.n_users):
            self.user_representations[user] += self._implicit_vector(user)

    def predict(self, user_idx: int | Iterable[int], item_idx: int | Iterable[int]):
        users = np.asarray(user_idx, dtype=np.int64)
        items = np.asarray(item_idx, dtype=np.int64)
        predictions = (
            self.global_mean
            + self.user_bias[users]
            + self.item_bias[items]
            + np.sum(self.user_representations[users] * self.item_factors[items], axis=-1)
        )
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
