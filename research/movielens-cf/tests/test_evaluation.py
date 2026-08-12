import math

import pandas as pd

from movielens_cf.evaluation import rating_metrics, ranking_metrics, recommendation_stats


def test_ranking_metrics_measure_recall_and_discount_rank():
    truth = {1: {10, 20}, 2: {30}}
    recommendations = {1: [10, 99, 20], 2: [99, 30, 98]}

    metrics, per_user = ranking_metrics(truth, recommendations, k=3)

    expected_user_one = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    expected_user_two = (1 / math.log2(3)) / 1
    assert metrics["recall_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == pytest.approx((expected_user_one + expected_user_two) / 2)
    assert set(per_user.columns) == {"user_id", "recall_at_10", "ndcg_at_10"}


def test_rating_metrics_return_rmse_and_mae():
    actual = pd.Series([5.0, 3.0, 1.0])
    predicted = pd.Series([4.0, 3.0, 3.0])

    metrics = rating_metrics(actual, predicted)

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == pytest.approx(math.sqrt(5 / 3))


def test_recommendation_stats_report_catalog_exposure_and_long_tail_share():
    recommendations = {1: [1, 3], 2: [3, 4]}
    item_counts = pd.Series({1: 100, 2: 50, 3: 10, 4: 1})

    metrics = recommendation_stats(recommendations, item_counts, catalog_size=4)

    assert metrics["catalog_coverage"] == 0.75
    assert metrics["long_tail_share"] == 0.75
    assert metrics["novelty"] > 0


import pytest
