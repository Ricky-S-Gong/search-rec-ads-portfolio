import numpy as np
import pandas as pd

from movielens_cf.baselines import BiasBaseline
from movielens_cf.neighborhood import NeighborhoodCF, select_top_indices, shrink_similarity


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


def test_top_n_uses_unclipped_scores_before_movie_id_tie_breaking():
    raw_scores = np.array([5.10, 5.45, 5.25])
    movie_ids = np.array([10, 30, 20])

    selected = select_top_indices(raw_scores, movie_ids, n=3)

    assert movie_ids[selected].tolist() == [30, 20, 10]


def test_top_n_uses_meaningful_secondary_scores_before_movie_id():
    raw_scores = np.array([5.0, 5.0, 5.0, 5.0])
    bayesian_scores = np.array([4.1, 4.8, 4.5, 4.3])
    evidence_strength = np.array([2.0, 1.0, 4.0, 3.0])
    movie_ids = np.array([10, 40, 20, 30])

    selected = select_top_indices(
        raw_scores,
        movie_ids,
        n=3,
        secondary_scores=bayesian_scores,
        evidence_strength=evidence_strength,
    )

    assert movie_ids[selected].tolist() == [40, 20, 30]


def test_bias_baseline_exposes_unclipped_predictions_and_training_residuals():
    ratings = fixture_ratings()
    baseline = BiasBaseline(regularization=5.0, iterations=5).fit(ratings)

    raw = baseline.predict_raw(1, 10)
    residuals = baseline.residuals(ratings)

    assert residuals.shape == (len(ratings),)
    assert residuals[0] == np.float64(ratings.iloc[0]["rating"] - raw)
    assert baseline.predict(1, 10) == np.clip(raw, 1, 5)


def test_bias_aware_item_cf_does_not_reduce_all_five_star_evidence_to_movie_id_order():
    ratings = pd.DataFrame(
        [
            (1, 10, 5.0), (1, 20, 5.0),
            (2, 10, 5.0), (2, 20, 4.0), (2, 30, 5.0), (2, 40, 2.0),
            (3, 10, 4.0), (3, 20, 5.0), (3, 30, 4.0), (3, 40, 1.0),
            (4, 10, 2.0), (4, 20, 3.0), (4, 30, 2.0), (4, 40, 5.0),
            (5, 10, 1.0), (5, 20, 2.0), (5, 30, 1.0), (5, 40, 4.0),
        ],
        columns=["user_id", "movie_id", "rating"],
    )
    model = NeighborhoodCF(
        mode="item",
        variant="bias_aware",
        k=3,
        min_support=2,
        shrinkage=0,
        min_neighbors=1,
    ).fit(ratings)

    recommendations = model.recommend(1, n=2)

    assert len({item.ranking_score for item in recommendations}) == 2
    assert recommendations[0].ranking_score > recommendations[1].ranking_score


def test_recommendation_exposes_raw_rank_score_and_clipped_rating_estimate():
    model = NeighborhoodCF(
        mode="user", k=3, min_support=2, shrinkage=0, min_neighbors=1
    ).fit(fixture_ratings())

    recommendation = model.recommend(1, n=1)[0]

    assert recommendation.ranking_score >= recommendation.rating_estimate
    assert 1.0 <= recommendation.rating_estimate <= 5.0
    assert recommendation.similarity_weight_sum > 0


def test_fallback_recommendation_has_no_neighborhood_confidence_weight():
    model = NeighborhoodCF(
        mode="user", k=3, min_support=2, shrinkage=0, min_neighbors=1
    ).fit(fixture_ratings())

    recommendation = model.recommend(999, n=1)[0]

    assert recommendation.used_fallback is True
    assert recommendation.similarity_weight_sum == 0.0
