import numpy as np
from scipy import sparse

from goodbooks_mf.bias_aware_als import BiasAwareALS


def biased_ratings(extra_items: int = 0) -> sparse.csr_matrix:
    rows = np.array([0, 0, 1, 1, 2, 2])
    cols = np.array([0, 1, 0, 2, 1, 2])
    values = np.array([5.0, 4.0, 3.0, 1.0, 2.0, 4.0])
    return sparse.csr_matrix((values, (rows, cols)), shape=(3, 3 + extra_items))


def test_bias_aware_als_uses_observed_ratings_for_regularized_biases():
    ratings = biased_ratings()
    model = BiasAwareALS(
        n_factors=2,
        reg_lambda=0.1,
        n_iterations=1,
        bias_reg_lambda=1.0,
        bias_iterations=10,
        seed=7,
    ).fit(ratings)

    assert model.global_mean == np.mean(ratings.data)
    assert not np.allclose(model.user_bias, 0)
    assert not np.allclose(model.item_bias, 0)
    assert np.any(model.residual_matrix.data < 0)
    assert model.residual_matrix.nnz == ratings.nnz


def test_bias_aware_als_prediction_restores_baseline_and_residual_factors():
    model = BiasAwareALS(
        n_factors=2,
        reg_lambda=0.2,
        n_iterations=3,
        bias_reg_lambda=1.0,
        bias_iterations=10,
        seed=11,
    ).fit(biased_ratings())

    users = np.array([0, 1, 2])
    items = np.array([1, 2, 1])
    expected = (
        model.global_mean
        + model.user_bias[users]
        + model.item_bias[items]
        + np.sum(model.user_factors[users] * model.item_factors[items], axis=1)
    )

    np.testing.assert_allclose(model.predict(users, items), expected)


def test_bias_aware_als_ignores_unobserved_cells_and_has_stable_cold_item_score():
    compact = BiasAwareALS(
        n_factors=2,
        reg_lambda=0.1,
        n_iterations=4,
        bias_reg_lambda=1.0,
        bias_iterations=10,
        seed=17,
    ).fit(biased_ratings())
    expanded = BiasAwareALS(
        n_factors=2,
        reg_lambda=0.1,
        n_iterations=4,
        bias_reg_lambda=1.0,
        bias_iterations=10,
        seed=17,
    ).fit(biased_ratings(extra_items=2))

    rows, cols = biased_ratings().nonzero()
    np.testing.assert_allclose(compact.predict(rows, cols), expanded.predict(rows, cols))
    assert expanded.predict(0, 4) == expanded.global_mean + expanded.user_bias[0]
