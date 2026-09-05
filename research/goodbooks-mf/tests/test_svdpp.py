import numpy as np
import pandas as pd

from goodbooks_mf.svdpp import SVDPP


def toy_interactions():
    train = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1],
            "item_idx": [0, 1, 2, 0, 1, 2],
            "rating": [5, 4, 0, 1, 2, 0],
            "is_read": [True, True, True, True, True, True],
            "is_reviewed": [False, False, False, False, False, False],
        }
    )
    validation = pd.DataFrame(
        {"user_idx": [0, 1], "item_idx": [1, 0], "rating": [4, 1]}
    )
    return train, validation


def test_svdpp_prediction_matches_the_documented_formula():
    model = SVDPP(n_users=1, n_items=2, n_factors=2)
    model.global_mean = 3.0
    model.user_bias = np.array([0.2])
    model.item_bias = np.array([0.0, -0.1])
    model.user_factors = np.array([[0.5, 1.0]])
    model.item_factors = np.array([[1.0, 1.0], [2.0, -1.0]])
    model.implicit_factors = np.array([[1.0, 0.0], [0.0, 2.0]])
    model.implicit_history = [np.array([0, 1])]
    model._refresh_user_representations()

    normalized_history = np.array([1.0, 2.0]) / np.sqrt(2)
    expected = 3.0 + 0.2 - 0.1 + np.array([2.0, -1.0]) @ (
        np.array([0.5, 1.0]) + normalized_history
    )

    assert model.predict(0, 1) == expected


def test_svdpp_fit_is_deterministic_and_predicts_finite_scores():
    train, validation = toy_interactions()
    kwargs = dict(
        n_users=2,
        n_items=3,
        n_factors=2,
        learning_rate=0.01,
        reg_lambda=0.02,
        n_epochs=4,
        patience=2,
        seed=7,
    )

    left = SVDPP(**kwargs).fit(train, validation)
    right = SVDPP(**kwargs).fit(train, validation)

    left_scores = left.predict([0, 1], [2, 2])
    right_scores = right.predict([0, 1], [2, 2])
    assert np.isfinite(left_scores).all()
    assert np.allclose(left_scores, right_scores)


def test_svdpp_uses_train_implicit_feedback_without_treating_zero_as_a_rating():
    train, _ = toy_interactions()
    without_implicit_only = train.loc[train["rating"] > 0].copy()
    kwargs = dict(
        n_users=2,
        n_items=3,
        n_factors=2,
        learning_rate=0.01,
        reg_lambda=0.02,
        n_epochs=2,
        seed=13,
    )

    with_history = SVDPP(**kwargs).fit(train)
    ratings_only = SVDPP(**kwargs).fit(without_implicit_only)

    assert with_history.global_mean == ratings_only.global_mean == 3.0
    assert not np.allclose(
        with_history.predict([0, 1], [2, 2]),
        ratings_only.predict([0, 1], [2, 2]),
    )


def test_svdpp_updates_implicit_item_factors_from_explicit_errors():
    train, _ = toy_interactions()
    seed = 17
    rng = np.random.default_rng(seed)
    rng.normal(0, 0.1, (2, 2))  # user factors
    rng.normal(0, 0.1, (3, 2))  # item factors
    initial_implicit = rng.normal(0, 0.1, (3, 2))
    model = SVDPP(
        n_users=2,
        n_items=3,
        n_factors=2,
        learning_rate=0.01,
        reg_lambda=0.02,
        n_epochs=1,
        seed=seed,
    ).fit(train)

    assert not np.allclose(model.implicit_factors, initial_implicit)


def test_svdpp_validation_interactions_never_enter_implicit_history():
    train, validation = toy_interactions()
    future_signals = validation.assign(is_read=True, is_reviewed=True)
    no_future_signals = validation.assign(is_read=False, is_reviewed=False)
    kwargs = dict(
        n_users=2,
        n_items=3,
        n_factors=2,
        learning_rate=0.01,
        reg_lambda=0.02,
        n_epochs=3,
        patience=2,
        seed=19,
    )

    left = SVDPP(**kwargs).fit(train, future_signals)
    right = SVDPP(**kwargs).fit(train, no_future_signals)

    assert np.allclose(left.predict([0, 1], [2, 2]), right.predict([0, 1], [2, 2]))


def test_svdpp_handles_a_user_with_no_train_history():
    train, _ = toy_interactions()
    model = SVDPP(
        n_users=3,
        n_items=3,
        n_factors=2,
        n_epochs=1,
        seed=23,
    ).fit(train)

    assert np.isfinite(model.predict(2, 1))


def test_svdpp_restores_the_complete_best_validation_checkpoint():
    train, validation = toy_interactions()
    common = dict(
        n_users=2,
        n_items=3,
        n_factors=2,
        learning_rate=0.03,
        reg_lambda=0.02,
        seed=29,
    )
    selected = SVDPP(**common, n_epochs=20, patience=2).fit(train, validation)
    assert selected.best_epoch is not None
    replay = SVDPP(**common, n_epochs=selected.best_epoch + 1).fit(train)

    assert np.allclose(
        selected.predict([0, 1], [0, 2]),
        replay.predict([0, 1], [0, 2]),
    )


def test_svdpp_recommend_excludes_seen_items_and_breaks_score_ties_by_item():
    train, _ = toy_interactions()
    model = SVDPP(
        n_users=2,
        n_items=3,
        n_factors=0,
        learning_rate=0,
        n_epochs=1,
        seed=31,
    ).fit(train)

    recommendations = model.recommend(0, [2, 1, 0], seen_item_idxs=[0], k=2)

    assert [item for item, _ in recommendations] == [1, 2]
