"""Sparse explicit-feedback ALS following the project's reference chapter.

For a user ``u`` with observed items ``I_u``, the update is the closed-form
ridge-regression solution used by the reference implementation::

    p_u = solve(Q_I.T @ Q_I + lambda * I, Q_I.T @ r_u)

The item update is symmetric. Only stored, positive ratings participate in
either normal equation; sparse-matrix zeros remain unobserved values.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import sparse


def _as_explicit_csr(rating_matrix) -> sparse.csr_matrix:
    """Return a finite, non-negative CSR matrix containing observed ratings."""
    if sparse.issparse(rating_matrix):
        matrix = rating_matrix.tocsr(copy=True).astype(np.float64)
    else:
        values = np.asarray(rating_matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("rating_matrix must be two-dimensional")
        matrix = sparse.csr_matrix(values)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.nnz == 0:
        raise ValueError("at least one explicit rating is required")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("ratings must be finite")
    if np.any(matrix.data < 0):
        raise ValueError("explicit ratings must be non-negative")
    matrix.sort_indices()
    return matrix


class ALS:
    """Alternating least squares over observed explicit ratings only."""

    def __init__(
        self,
        n_factors: int = 50,
        reg_lambda: float = 0.01,
        n_iterations: int = 10,
        seed: int = 42,
    ):
        if n_factors <= 0:
            raise ValueError("n_factors must be positive")
        if reg_lambda <= 0:
            raise ValueError("reg_lambda must be positive")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        self.n_factors = n_factors
        self.reg_lambda = reg_lambda
        self.n_iterations = n_iterations
        self.seed = seed

    def fit(self, rating_matrix) -> "ALS":
        """Fit factors by alternating the reference user and item solves."""
        self.train_matrix = _as_explicit_csr(rating_matrix)
        self.train_matrix_t = self.train_matrix.T.tocsr()
        self.n_users, self.n_items = self.train_matrix.shape

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (self.n_items, self.n_factors))
        self.reg_eye = self.reg_lambda * np.eye(self.n_factors)
        self.training_rmse_: list[float] = []

        for _ in range(self.n_iterations):
            self._update_user_factors()
            self._update_item_factors()
            self.training_rmse_.append(self.observed_rmse())
        return self

    def _update_user_factors(self) -> None:
        """Fix item factors and solve one ridge regression per user."""
        for user in range(self.n_users):
            start, end = self.train_matrix.indptr[user : user + 2]
            items = self.train_matrix.indices[start:end]
            ratings = self.train_matrix.data[start:end]
            if not len(items):
                continue
            item_vectors = self.item_factors[items]
            normal_matrix = item_vectors.T @ item_vectors + self.reg_eye
            target = item_vectors.T @ ratings
            self.user_factors[user] = np.linalg.solve(normal_matrix, target)

    def _update_item_factors(self) -> None:
        """Fix user factors and solve one ridge regression per item."""
        for item in range(self.n_items):
            start, end = self.train_matrix_t.indptr[item : item + 2]
            users = self.train_matrix_t.indices[start:end]
            ratings = self.train_matrix_t.data[start:end]
            if not len(users):
                continue
            user_vectors = self.user_factors[users]
            normal_matrix = user_vectors.T @ user_vectors + self.reg_eye
            target = user_vectors.T @ ratings
            self.item_factors[item] = np.linalg.solve(normal_matrix, target)

    def predict(self, user_idx: int | Iterable[int], item_idx: int | Iterable[int]):
        users = np.asarray(user_idx, dtype=np.int64)
        items = np.asarray(item_idx, dtype=np.int64)
        predictions = np.sum(self.user_factors[users] * self.item_factors[items], axis=-1)
        return float(predictions) if predictions.ndim == 0 else predictions

    def observed_rmse(self, rating_matrix=None) -> float:
        """Compute RMSE strictly over stored ratings, never sparse zeros."""
        matrix = self.train_matrix if rating_matrix is None else _as_explicit_csr(rating_matrix)
        if matrix.shape != (self.n_users, self.n_items):
            raise ValueError("rating_matrix shape does not match the fitted model")
        rows, cols = matrix.nonzero()
        errors = matrix.data - self.predict(rows, cols)
        return float(np.sqrt(np.mean(np.square(errors))))

    def recommend(
        self,
        user_idx: int,
        candidate_item_idxs: Iterable[int],
        seen_item_idxs: Iterable[int] = (),
        k: int = 10,
    ) -> list[tuple[int, float]]:
        seen = set(seen_item_idxs)
        candidates = np.array(sorted(set(candidate_item_idxs).difference(seen)), dtype=np.int64)
        if not len(candidates) or k <= 0:
            return []
        scores = np.asarray(self.predict(np.full(len(candidates), user_idx), candidates))
        order = np.lexsort((candidates, -scores))[:k]
        return [(int(candidates[index]), float(scores[index])) for index in order]
