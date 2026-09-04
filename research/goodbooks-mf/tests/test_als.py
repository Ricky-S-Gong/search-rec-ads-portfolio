import numpy as np
from scipy import sparse

from goodbooks_mf.als import ALS


def explicit_ratings(extra_items: int = 0) -> sparse.csr_matrix:
    rows = np.array([0, 0, 1, 1, 2, 2])
    cols = np.array([0, 1, 0, 2, 1, 2])
    values = np.array([5.0, 3.0, 4.0, 1.0, 2.0, 5.0])
    return sparse.csr_matrix((values, (rows, cols)), shape=(3, 3 + extra_items))


def test_als_user_update_matches_reference_closed_form():
    ratings = explicit_ratings()
    model = ALS(n_factors=2, reg_lambda=0.1, n_iterations=1, seed=7)
    model.train_matrix = ratings
    model.n_users, model.n_items = ratings.shape
    model.reg_eye = model.reg_lambda * np.eye(model.n_factors)
    model.item_factors = np.array(
        [
            [0.2, 0.8],
            [0.6, 0.4],
            [0.9, 0.1],
        ]
    )
    model.user_factors = np.zeros((model.n_users, model.n_factors))

    item_vectors = model.item_factors[[0, 1]]
    expected = np.linalg.solve(
        item_vectors.T @ item_vectors + model.reg_eye,
        item_vectors.T @ np.array([5.0, 3.0]),
    )

    model._update_user_factors()

    np.testing.assert_allclose(model.user_factors[0], expected)


def test_als_reduces_error_only_over_observed_ratings():
    ratings = explicit_ratings()
    model = ALS(n_factors=2, reg_lambda=0.1, n_iterations=1, seed=11)
    model.fit(ratings)
    first_error = model.training_rmse_[0]

    model = ALS(n_factors=2, reg_lambda=0.1, n_iterations=8, seed=11)
    model.fit(ratings)

    assert model.training_rmse_[-1] < first_error
    assert len(model.training_rmse_) == 8


def test_als_does_not_treat_unobserved_cells_as_zero_ratings():
    compact = ALS(n_factors=2, reg_lambda=0.1, n_iterations=4, seed=17).fit(
        explicit_ratings()
    )
    expanded = ALS(n_factors=2, reg_lambda=0.1, n_iterations=4, seed=17).fit(
        explicit_ratings(extra_items=4)
    )

    rows, cols = explicit_ratings().nonzero()
    np.testing.assert_allclose(
        compact.predict(rows, cols),
        expanded.predict(rows, cols),
    )
