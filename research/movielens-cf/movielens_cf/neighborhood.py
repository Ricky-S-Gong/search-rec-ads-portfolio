from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .baselines import BayesianPopularity


Mode = Literal["user", "item"]


@dataclass(frozen=True)
class Prediction:
    rating: float
    neighbor_count: int
    used_fallback: bool


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    score: float
    neighbor_count: int
    used_fallback: bool


def shrink_similarity(similarity: float, support: int, shrinkage: float) -> float:
    if support < 0 or shrinkage < 0:
        raise ValueError("support and shrinkage must be non-negative")
    return float(similarity * support / (support + shrinkage)) if support + shrinkage else 0.0


class NeighborhoodCF:
    """Sparse, mean-centered User-CF or adjusted-cosine Item-CF."""

    def __init__(
        self,
        mode: Mode,
        k: int = 40,
        min_support: int = 5,
        shrinkage: float = 25.0,
        min_neighbors: int = 2,
        block_size: int = 512,
    ):
        if mode not in ("user", "item"):
            raise ValueError("mode must be 'user' or 'item'")
        self.mode = mode
        self.k = k
        self.min_support = min_support
        self.shrinkage = shrinkage
        self.min_neighbors = min_neighbors
        self.block_size = block_size

    def fit(self, ratings: pd.DataFrame) -> "NeighborhoodCF":
        self.user_ids = np.sort(ratings["user_id"].unique()).astype(int)
        self.movie_ids = np.sort(ratings["movie_id"].unique()).astype(int)
        self.user_index = {value: index for index, value in enumerate(self.user_ids)}
        self.movie_index = {value: index for index, value in enumerate(self.movie_ids)}
        rows = ratings["user_id"].map(self.user_index).to_numpy()
        cols = ratings["movie_id"].map(self.movie_index).to_numpy()
        values = ratings["rating"].to_numpy(dtype=np.float32)
        shape = (len(self.user_ids), len(self.movie_ids))
        self.ratings = csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float32)
        self.binary = csr_matrix((np.ones(len(values), dtype=np.float32), (rows, cols)), shape=shape)
        counts = np.asarray(self.binary.sum(axis=1)).ravel()
        self.user_means = np.asarray(self.ratings.sum(axis=1)).ravel() / counts
        centered_values = values - self.user_means[rows]
        self.centered = csr_matrix((centered_values, (rows, cols)), shape=shape, dtype=np.float32)
        entity_values = self.centered if self.mode == "user" else self.centered.T.tocsr()
        entity_binary = self.binary if self.mode == "user" else self.binary.T.tocsr()
        self.similarity = self._top_k_similarity(entity_values, entity_binary)
        self.popularity = BayesianPopularity().fit(ratings)
        self.global_mean = float(values.mean())
        self.seen = {
            user_id: set(group["movie_id"].astype(int))
            for user_id, group in ratings.groupby("user_id", observed=True)
        }
        return self

    def _top_k_similarity(self, values: csr_matrix, binary: csr_matrix) -> csr_matrix:
        normalized = normalize(values, norm="l2", axis=1, copy=True)
        row_parts: list[int] = []
        col_parts: list[int] = []
        value_parts: list[float] = []
        entity_count = values.shape[0]
        for start in range(0, entity_count, self.block_size):
            end = min(start + self.block_size, entity_count)
            similarities = (normalized[start:end] @ normalized.T).tocsr()
            supports = (binary[start:end] @ binary.T).tocsr()
            for local_row in range(end - start):
                entity = start + local_row
                row = similarities.getrow(local_row)
                indices = row.indices
                scores = row.data
                if len(indices) == 0:
                    continue
                common = supports.getrow(local_row)[:, indices].toarray().ravel()
                mask = (indices != entity) & (scores > 0) & (common >= self.min_support)
                indices = indices[mask]
                scores = scores[mask] * common[mask] / (common[mask] + self.shrinkage)
                if len(indices) > self.k:
                    keep = np.argpartition(-scores, self.k - 1)[: self.k]
                    indices, scores = indices[keep], scores[keep]
                order = np.lexsort((indices, -scores))
                row_parts.extend([entity] * len(order))
                col_parts.extend(indices[order].tolist())
                value_parts.extend(scores[order].astype(float).tolist())
        return csr_matrix(
            (np.asarray(value_parts, dtype=np.float32), (row_parts, col_parts)),
            shape=(entity_count, entity_count),
        )

    def predict(self, user_id: int, movie_id: int) -> Prediction:
        user = self.user_index.get(user_id)
        item = self.movie_index.get(movie_id)
        if user is None or item is None:
            return Prediction(self._fallback_rating(movie_id), 0, True)
        if self.mode == "user":
            neighbors = self.similarity.getrow(user)
            rated = self.binary[neighbors.indices, item].toarray().ravel() > 0
            indices, weights = neighbors.indices[rated], neighbors.data[rated]
            residuals = self.centered[indices, item].toarray().ravel()
        else:
            neighbors = self.similarity.getrow(item)
            rated = self.binary[user, neighbors.indices].toarray().ravel() > 0
            indices, weights = neighbors.indices[rated], neighbors.data[rated]
            residuals = self.centered[user, indices].toarray().ravel()
        count = int(len(weights))
        if count < self.min_neighbors or not np.abs(weights).sum():
            return Prediction(self._fallback_rating(movie_id), count, True)
        score = self.user_means[user] + float(weights @ residuals / np.abs(weights).sum())
        return Prediction(float(np.clip(score, 1, 5)), count, False)

    def recommend(self, user_id: int, n: int = 10) -> list[Recommendation]:
        return self.recommend_many([user_id], n=n)[user_id]

    def recommend_many(
        self, user_ids: list[int] | np.ndarray, n: int = 10, batch_size: int = 256
    ) -> dict[int, list[Recommendation]]:
        """Score the full fitted catalog in sparse batches and remove seen movies."""
        result: dict[int, list[Recommendation]] = {}
        if n <= 0:
            return {int(user_id): [] for user_id in user_ids}
        known = [int(user_id) for user_id in user_ids if int(user_id) in self.user_index]
        for user_id in user_ids:
            user_id = int(user_id)
            if user_id not in self.user_index:
                seen = self.seen.get(user_id, set())
                result[user_id] = [
                    Recommendation(movie_id, self.popularity.scores[movie_id], 0, True)
                    for movie_id in self.popularity.ranking
                    if movie_id not in seen
                ][:n]
        absolute_similarity = self.similarity.copy()
        absolute_similarity.data = np.abs(absolute_similarity.data)
        neighbor_indicator = self.similarity.copy()
        neighbor_indicator.data = np.ones_like(neighbor_indicator.data)
        fallback_scores = np.asarray(
            [self.popularity.scores[int(movie_id)] for movie_id in self.movie_ids], dtype=float
        )
        for start in range(0, len(known), batch_size):
            batch_ids = known[start : start + batch_size]
            user_rows = np.asarray([self.user_index[user_id] for user_id in batch_ids])
            if self.mode == "user":
                weights = self.similarity[user_rows]
                numerator = (weights @ self.centered).toarray()
                denominator = (absolute_similarity[user_rows] @ self.binary).toarray()
                counts = (neighbor_indicator[user_rows] @ self.binary).toarray()
            else:
                numerator = (self.centered[user_rows] @ self.similarity.T).toarray()
                denominator = (self.binary[user_rows] @ absolute_similarity.T).toarray()
                counts = (self.binary[user_rows] @ neighbor_indicator.T).toarray()
            supported = (counts >= self.min_neighbors) & (denominator > 0)
            predictions = np.broadcast_to(self.user_means[user_rows, None], numerator.shape).copy()
            np.divide(numerator, denominator, out=numerator, where=denominator > 0)
            predictions += numerator
            predictions = np.clip(predictions, 1, 5)
            scores = np.where(supported, predictions, fallback_scores[None, :])
            for local_row, user_id in enumerate(batch_ids):
                seen_indices = [
                    self.movie_index[movie_id]
                    for movie_id in self.seen[user_id]
                    if movie_id in self.movie_index
                ]
                scores[local_row, seen_indices] = -np.inf
                take = min(n, int(np.isfinite(scores[local_row]).sum()))
                if take == 0:
                    result[user_id] = []
                    continue
                candidate_indices = np.argpartition(-scores[local_row], take - 1)[:take]
                order = np.lexsort((self.movie_ids[candidate_indices], -scores[local_row, candidate_indices]))
                result[user_id] = [
                    Recommendation(
                        int(self.movie_ids[index]),
                        float(scores[local_row, index]),
                        int(counts[local_row, index]),
                        not bool(supported[local_row, index]),
                    )
                    for index in candidate_indices[order]
                ]
        return result

    def _fallback_rating(self, movie_id: int) -> float:
        return float(np.clip(self.popularity.scores.get(movie_id, self.global_mean), 1, 5))
