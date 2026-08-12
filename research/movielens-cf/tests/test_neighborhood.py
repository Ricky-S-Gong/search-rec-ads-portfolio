import numpy as np
import pandas as pd

from movielens_cf.neighborhood import NeighborhoodCF, shrink_similarity


def fixture_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, 10, 5.0), (1, 20, 1.0), (1, 30, 4.0),
            (2, 10, 5.0), (2, 20, 1.0), (2, 40, 5.0),
            (3, 10, 1.0), (3, 20, 5.0), (3, 40, 1.0),
            (4, 10, 4.0), (4, 20, 2.0), (4, 30, 5.0), (4, 40, 4.0),
        ],
        columns=["user_id", "movie_id", "rating"],
    )


def test_similarity_shrinkage_penalizes_low_support_more_strongly():
    assert shrink_similarity(0.8, support=2, shrinkage=10) < shrink_similarity(
        0.8, support=20, shrinkage=10
    )
    assert shrink_similarity(0.8, support=20, shrinkage=0) == 0.8


def test_user_cf_predicts_from_positive_centered_neighbors_and_reports_support():
    model = NeighborhoodCF(
        mode="user", k=3, min_support=2, shrinkage=0, min_neighbors=1
    ).fit(fixture_ratings())

    prediction = model.predict(1, 40)

    assert prediction.used_fallback is False
    assert prediction.neighbor_count >= 1
    assert 4.0 < prediction.rating <= 5.0


def test_item_cf_excludes_seen_items_and_is_deterministic():
    model = NeighborhoodCF(
        mode="item", k=3, min_support=2, shrinkage=0, min_neighbors=1
    ).fit(fixture_ratings())

    first = model.recommend(1, n=2)
    second = model.recommend(1, n=2)

    assert first == second
    assert {item.movie_id for item in first}.isdisjoint({10, 20, 30})
    assert first[0].movie_id == 40


def test_unknown_user_uses_the_shared_fallback_scores():
    model = NeighborhoodCF(
        mode="user", k=3, min_support=2, shrinkage=0, min_neighbors=1
    ).fit(fixture_ratings())

    recommendations = model.recommend(999, n=2)

    assert len(recommendations) == 2
    assert all(item.used_fallback for item in recommendations)
    assert np.isfinite([item.score for item in recommendations]).all()


def test_batched_recommendations_match_single_user_results():
    model = NeighborhoodCF(
        mode="item", k=3, min_support=2, shrinkage=0, min_neighbors=1
    ).fit(fixture_ratings())

    batched = model.recommend_many([1, 2], n=2)

    assert batched[1] == model.recommend(1, n=2)
    assert batched[2] == model.recommend(2, n=2)
