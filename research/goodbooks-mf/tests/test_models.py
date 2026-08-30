import numpy as np
import pandas as pd

from goodbooks_mf.models import BasicMF, FunkSVD, rmse


def toy_ratings():
    train = pd.DataFrame(
        [
            (0, 0, 5), (0, 1, 4), (0, 2, 1),
            (1, 0, 4), (1, 1, 5), (1, 2, 1),
            (2, 0, 1), (2, 1, 1), (2, 2, 5),
        ],
        columns=["user_idx", "item_idx", "rating"],
    )
    validation = pd.DataFrame(
        [(0, 0, 5), (1, 1, 5), (2, 2, 5)],
        columns=["user_idx", "item_idx", "rating"],
    )
    return train, validation


def test_basic_mf_is_deterministic_for_a_fixed_seed():
    train, _ = toy_ratings()
    left = BasicMF(n_users=3, n_items=4, n_factors=3, n_epochs=8, seed=7).fit(train)
    right = BasicMF(n_users=3, n_items=4, n_factors=3, n_epochs=8, seed=7).fit(train)

    assert np.allclose(left.user_factors, right.user_factors)
    assert np.allclose(left.item_factors, right.item_factors)


def test_funksvd_learns_biases_and_restores_best_validation_epoch():
    train, validation = toy_ratings()
    model = FunkSVD(
        n_users=3,
        n_items=4,
        n_factors=3,
        learning_rate=0.02,
        reg_lambda=0.01,
        n_epochs=200,
        patience=15,
        seed=11,
    ).fit(train, validation)

    predictions = model.predict(validation["user_idx"], validation["item_idx"])
    baseline = np.repeat(train["rating"].mean(), len(validation))
    assert rmse(validation["rating"], predictions) < rmse(validation["rating"], baseline)
    assert model.best_epoch is not None
    assert model.n_epochs_trained <= 200
    assert not np.allclose(model.user_bias, 0)


def test_recommend_excludes_seen_items_and_orders_by_raw_score():
    train, _ = toy_ratings()
    model = FunkSVD(n_users=3, n_items=4, n_factors=2, n_epochs=20, seed=3).fit(train)

    recommendations = model.recommend(0, candidate_item_idxs=[0, 1, 2, 3], seen_item_idxs=[0, 1], k=2)

    assert [item for item, _ in recommendations] == [
        item for item, _ in sorted(recommendations, key=lambda pair: (-pair[1], pair[0]))
    ]
    assert {item for item, _ in recommendations}.isdisjoint({0, 1})
