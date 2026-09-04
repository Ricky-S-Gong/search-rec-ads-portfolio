"""Bias-aware explicit ALS over observed rating residuals.

The model first estimates a regularized baseline from observed ratings only::

    baseline_ui = global_mean + user_bias_u + item_bias_i

It then applies the project's sparse explicit-feedback ALS updates to the
residual ``rating_ui - baseline_ui``. Predictions restore the baseline before
adding the latent interaction term::

    prediction_ui = baseline_ui + user_factors_u @ item_factors_i

Sparse-matrix zeros remain missing throughout; negative residuals are valid.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import sparse

from .als import ALS, _as_explicit_csr


class BiasAwareALS(ALS):
    """Two-stage regularized bias baseline plus residual explicit ALS."""

    def __init__(
        self,
        n_factors: int = 50,
        reg_lambda: float = 0.01,
        n_iterations: int = 10,
        bias_reg_lambda: float = 10.0,
        bias_iterations: int = 10,
        seed: int = 42,
    ):
        super().__init__(n_factors, reg_lambda, n_iterations, seed)
        if bias_reg_lambda <= 0:
            raise ValueError("bias_reg_lambda must be positive")
        if bias_iterations <= 0:
            raise ValueError("bias_iterations must be positive")
        self.bias_reg_lambda = bias_reg_lambda
        self.bias_iterations = bias_iterations

    def fit(self, rating_matrix) -> "BiasAwareALS":
        """Estimate biases, then factorize observed baseline residuals."""
        self.rating_matrix = _as_explicit_csr(rating_matrix)
        self.n_users, self.n_items = self.rating_matrix.shape
        self._fit_biases()

        rows = np.repeat(
            np.arange(self.n_users, dtype=np.int64),
            np.diff(self.rating_matrix.indptr),
        )
        residuals = self.rating_matrix.data - (
            self.global_mean + self.user_bias[rows] + self.item_bias[self.rating_matrix.indices]
        )
        # Preserve explicitly observed zero residuals: they still contribute to
        # the normal-equation design matrices even though their target is zero.
        self.residual_matrix = sparse.csr_matrix(
            (
                residuals,
                self.rating_matrix.indices.copy(),
                self.rating_matrix.indptr.copy(),
            ),
            shape=self.rating_matrix.shape,
        )
        self.train_matrix = self.residual_matrix
        self.train_matrix_t = self.train_matrix.T.tocsr()

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (self.n_items, self.n_factors))
        self.reg_eye = self.reg_lambda * np.eye(self.n_factors)
        self.training_rmse_: list[float] = []

        for _ in range(self.n_iterations):
            self._update_user_factors()
            self._update_item_factors()
            self.training_rmse_.append(self.observed_rmse())

        cold_users = np.diff(self.rating_matrix.indptr) == 0
        cold_items = np.diff(self.rating_matrix.tocsc().indptr) == 0
        self.user_factors[cold_users] = 0.0
        self.item_factors[cold_items] = 0.0
        return self

    def _fit_biases(self) -> None:
        """Fit regularized user and item intercepts by alternating updates."""
        coo = self.rating_matrix.tocoo()
        users = coo.row
        items = coo.col
        ratings = coo.data
        user_counts = np.bincount(users, minlength=self.n_users)
        item_counts = np.bincount(items, minlength=self.n_items)

        self.global_mean = float(np.mean(ratings))
        self.user_bias = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias = np.zeros(self.n_items, dtype=np.float64)

        for _ in range(self.bias_iterations):
            user_sums = np.bincount(
                users,
                weights=ratings - self.global_mean - self.item_bias[items],
                minlength=self.n_users,
            )
            self.user_bias = user_sums / (user_counts + self.bias_reg_lambda)
            item_sums = np.bincount(
                items,
                weights=ratings - self.global_mean - self.user_bias[users],
                minlength=self.n_items,
            )
            self.item_bias = item_sums / (item_counts + self.bias_reg_lambda)

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

    def observed_rmse(self, rating_matrix=None) -> float:
        """Compute RMSE over observed raw ratings, never missing positions."""
        matrix = self.rating_matrix if rating_matrix is None else _as_explicit_csr(rating_matrix)
        if matrix.shape != (self.n_users, self.n_items):
            raise ValueError("rating_matrix shape does not match the fitted model")
        rows, cols = matrix.nonzero()
        errors = matrix.data - self.predict(rows, cols)
        return float(np.sqrt(np.mean(np.square(errors))))
