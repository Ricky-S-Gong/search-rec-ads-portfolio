from __future__ import annotations

import numpy as np
import pandas as pd


class BiasBaseline:
    """Regularized global + user + item rating baseline."""

    def __init__(self, regularization: float = 10.0, iterations: int = 10):
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, ratings: pd.DataFrame) -> "BiasBaseline":
        self.global_mean = float(ratings["rating"].mean())
        self.user_bias: dict[int, float] = {int(user): 0.0 for user in ratings["user_id"].unique()}
        self.item_bias: dict[int, float] = {int(item): 0.0 for item in ratings["movie_id"].unique()}
        for _ in range(self.iterations):
            item_bias = ratings["movie_id"].map(self.item_bias).to_numpy(dtype=float)
            user_residual = ratings["rating"].to_numpy(dtype=float) - self.global_mean - item_bias
            user_sum = pd.Series(user_residual).groupby(ratings["user_id"].to_numpy()).sum()
            user_count = ratings.groupby("user_id", observed=True).size()
            self.user_bias = (user_sum / (user_count + self.regularization)).to_dict()
            user_bias = ratings["user_id"].map(self.user_bias).to_numpy(dtype=float)
            item_residual = ratings["rating"].to_numpy(dtype=float) - self.global_mean - user_bias
            item_sum = pd.Series(item_residual).groupby(ratings["movie_id"].to_numpy()).sum()
            item_count = ratings.groupby("movie_id", observed=True).size()
            self.item_bias = (item_sum / (item_count + self.regularization)).to_dict()
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        return float(np.clip(self.predict_raw(user_id, movie_id), 1, 5))

    def predict_raw(self, user_id: int, movie_id: int) -> float:
        """Return the unclipped baseline b_ui = mu + b_u + b_i."""
        return float(
            self.global_mean
            + self.user_bias.get(user_id, 0.0)
            + self.item_bias.get(movie_id, 0.0)
        )

    def residuals(self, ratings: pd.DataFrame) -> np.ndarray:
        """Return training residuals r_ui - b_ui in input-row order."""
        user_bias = ratings["user_id"].map(self.user_bias).fillna(0.0).to_numpy(dtype=float)
        item_bias = ratings["movie_id"].map(self.item_bias).fillna(0.0).to_numpy(dtype=float)
        baseline = self.global_mean + user_bias + item_bias
        return ratings["rating"].to_numpy(dtype=float) - baseline


class BayesianPopularity:
    """Movie mean shrunk toward the catalog mean for fair cold-start fallback."""

    def __init__(self, prior_weight: float = 25.0):
        self.prior_weight = prior_weight

    def fit(self, ratings: pd.DataFrame) -> "BayesianPopularity":
        self.global_mean = float(ratings["rating"].mean())
        grouped = ratings.groupby("movie_id", observed=True)["rating"].agg(["count", "mean"])
        grouped["score"] = (
            grouped["count"] * grouped["mean"] + self.prior_weight * self.global_mean
        ) / (grouped["count"] + self.prior_weight)
        self.scores = {int(item): float(score) for item, score in grouped["score"].items()}
        self.ranking = sorted(self.scores, key=lambda item: (-self.scores[item], item))
        return self
