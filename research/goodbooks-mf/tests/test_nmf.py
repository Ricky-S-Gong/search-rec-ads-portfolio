import numpy as np
import pytest
from scipy import sparse

from goodbooks_mf.nmf import NMF


def explicit_ratings(extra_items: int = 0) -> sparse.csr_matrix:
    rows = np.array([0, 0, 1, 1, 2, 2])
    cols = np.array([0, 1, 0, 2, 1, 2])
    values = np.array([5.0, 3.0, 4.0, 1.0, 2.0, 5.0])
    return sparse.csr_matrix((values, (rows, cols)), shape=(3, 3 + extra_items))


def test_masked_nmf_keeps_factors_nonnegative_and_reduces_observed_loss():
    ratings = explicit_ratings()
    one_step = NMF(n_factors=2, max_iter=1, tol=0.0, seed=5).fit(ratings)
    trained = NMF(n_factors=2, max_iter=40, tol=0.0, seed=5).fit(ratings)

    assert np.all(trained.user_factors >= 0)
    assert np.all(trained.item_factors >= 0)
    assert trained.observed_loss(ratings) < one_step.observed_loss(ratings)


def test_masked_nmf_ignores_unobserved_cells_in_updates_and_loss():
    compact_matrix = explicit_ratings()
    expanded_matrix = explicit_ratings(extra_items=4)
    compact = NMF(n_factors=2, max_iter=15, tol=0.0, seed=19).fit(compact_matrix)
    expanded = NMF(n_factors=2, max_iter=15, tol=0.0, seed=19).fit(expanded_matrix)

    rows, cols = compact_matrix.nonzero()
    np.testing.assert_allclose(
        compact.predict(rows, cols),
        expanded.predict(rows, cols),
    )
    np.testing.assert_allclose(
        compact.observed_loss(compact_matrix),
        expanded.observed_loss(expanded_matrix),
    )


def test_masked_nmf_observed_loss_excludes_missing_positions():
    model = NMF(n_factors=1, max_iter=1, seed=3)
    model.n_users = 2
    model.n_items = 2
    model.user_factors = np.array([[1.0], [100.0]])
    model.item_factors = np.array([[2.0], [100.0]])
    ratings = sparse.csr_matrix(([3.0], ([0], [0])), shape=(2, 2))

    # Only (0, 0) is observed: (3 - 1*2)^2 = 1. The huge prediction at
    # missing (1, 1) must not enter the objective.
    assert model.observed_loss(ratings) == 1.0


def test_regularized_nmf_updates_match_l2_multiplicative_formula():
    ratings = explicit_ratings()
    model = NMF(
        n_factors=2,
        max_iter=1,
        reg_lambda=0.5,
        epsilon=1e-12,
        seed=7,
    )
    model.train_matrix = ratings
    model.n_users, model.n_items = ratings.shape
    model._rows, model._cols = ratings.nonzero()
    model.user_factors = np.array(
        [[0.2, 0.8], [0.5, 0.4], [0.7, 0.3]], dtype=np.float64
    )
    model.item_factors = np.array(
        [[0.6, 0.2], [0.3, 0.9], [0.8, 0.5]], dtype=np.float64
    )

    user_numerator = np.asarray(ratings @ model.item_factors)
    user_denominator = np.asarray(
        model._masked_prediction_matrix() @ model.item_factors
    ) + model.reg_lambda * model.user_factors
    expected_users = model.user_factors * user_numerator / (
        user_denominator + model.epsilon
    )
    model._update_user_factors()
    np.testing.assert_allclose(model.user_factors, expected_users)

    item_numerator = np.asarray(ratings.T @ model.user_factors)
    item_denominator = np.asarray(
        model._masked_prediction_matrix().T @ model.user_factors
    ) + model.reg_lambda * model.item_factors
    expected_items = model.item_factors * item_numerator / (
        item_denominator + model.epsilon
    )
    model._update_item_factors()
    np.testing.assert_allclose(model.item_factors, expected_items)


def test_regularized_nmf_objective_adds_factor_penalty_only():
    ratings = sparse.csr_matrix(([3.0], ([0], [0])), shape=(2, 2))
    model = NMF(n_factors=1, max_iter=1, reg_lambda=0.25, seed=3)
    model.n_users = 2
    model.n_items = 2
    model.user_factors = np.array([[1.0], [2.0]])
    model.item_factors = np.array([[2.0], [3.0]])

    expected = model.observed_loss(ratings) + 0.25 * (1.0 + 4.0 + 4.0 + 9.0)
    assert model.regularized_objective(ratings) == expected


def test_regularized_nmf_rejects_negative_lambda():
    with pytest.raises(ValueError, match="reg_lambda"):
        NMF(reg_lambda=-0.1)
