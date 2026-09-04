"""Observed-only non-negative matrix factorization for explicit ratings.

The reference chapter uses non-negative multiplicative updates, a mask for
observed ratings, and optional L2 factor regularization. This implementation
evaluates the masked products directly on CSR coordinates:

    P <- P * ((M * R) Q) / ((M * (P Q.T)) Q + lambda P + epsilon)
    Q <- Q * ((M * R).T P) / ((M * (P Q.T)).T P + lambda Q + epsilon)

Consequently, an absent CSR entry never acts like an observed rating of zero.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import sparse

from .als import _as_explicit_csr


class NMF:
    """Masked NMF with non-negative multiplicative factor updates."""

    def __init__(
        self,
        n_factors: int = 50,
        max_iter: int = 100,
        tol: float = 1e-4,
        epsilon: float = 1e-10,
        seed: int = 42,
        reg_lambda: float = 0.0,
    ):
        if n_factors <= 0:
            raise ValueError("n_factors must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.seed = seed
        self.reg_lambda = reg_lambda

    def fit(self, rating_matrix) -> "NMF":
        """Fit non-negative factors using only stored explicit ratings."""
        self.train_matrix = _as_explicit_csr(rating_matrix)
        self.n_users, self.n_items = self.train_matrix.shape
        self._rows, self._cols = self.train_matrix.nonzero()

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.uniform(
            self.epsilon, 1.0, (self.n_users, self.n_factors)
        )
        self.item_factors = rng.uniform(
            self.epsilon, 1.0, (self.n_items, self.n_factors)
        )

        previous_rmse = np.inf
        self.training_error_: list[float] = []
        self.training_objective_: list[float] = []
        self.n_iterations_ = 0
        for iteration in range(self.max_iter):
            self._update_user_factors()
            self._update_item_factors()
            self.n_iterations_ = iteration + 1

            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                error = self.observed_loss()
                self.training_error_.append(error)
                self.training_objective_.append(self.regularized_objective())
                # observed_loss is a sum, so converge on RMSE instead: tol then
                # stays in rating units and does not scale with the nnz count.
                rmse = np.sqrt(error / self.train_matrix.nnz)
                if abs(previous_rmse - rmse) < self.tol:
                    break
                previous_rmse = rmse
        return self

    def _masked_prediction_matrix(self) -> sparse.csr_matrix:
        predictions = np.sum(
            self.user_factors[self._rows] * self.item_factors[self._cols], axis=1
        )
        return sparse.csr_matrix(
            (predictions, (self._rows, self._cols)),
            shape=(self.n_users, self.n_items),
        )

    def _update_user_factors(self) -> None:
        numerator = np.asarray(self.train_matrix @ self.item_factors)
        denominator = np.asarray(
            self._masked_prediction_matrix() @ self.item_factors
        ) + self.reg_lambda * self.user_factors
        self.user_factors *= numerator / (denominator + self.epsilon)

    def _update_item_factors(self) -> None:
        numerator = np.asarray(self.train_matrix.T @ self.user_factors)
        denominator = np.asarray(
            self._masked_prediction_matrix().T @ self.user_factors
        ) + self.reg_lambda * self.item_factors
        self.item_factors *= numerator / (denominator + self.epsilon)

    def predict(self, user_idx: int | Iterable[int], item_idx: int | Iterable[int]):
        users = np.asarray(user_idx, dtype=np.int64)
        items = np.asarray(item_idx, dtype=np.int64)
        predictions = np.sum(self.user_factors[users] * self.item_factors[items], axis=-1)
        return float(predictions) if predictions.ndim == 0 else predictions

    def observed_loss(self, rating_matrix=None) -> float:
        """Return squared reconstruction error over observed ratings only."""
        matrix = self.train_matrix if rating_matrix is None else _as_explicit_csr(rating_matrix)
        if matrix.shape != (self.n_users, self.n_items):
            raise ValueError("rating_matrix shape does not match the fitted model")
        rows, cols = matrix.nonzero()
        residuals = matrix.data - self.predict(rows, cols)
        return float(np.sum(np.square(residuals)))

    def regularized_objective(self, rating_matrix=None) -> float:
        """Return observed reconstruction loss plus L2 factor penalty."""
        penalty = self.reg_lambda * (
            np.sum(np.square(self.user_factors))
            + np.sum(np.square(self.item_factors))
        )
        return float(self.observed_loss(rating_matrix) + penalty)

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
