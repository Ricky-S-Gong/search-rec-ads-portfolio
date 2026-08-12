import math

import pandas as pd
import pytest

from movielens_cf.evaluation import (
    paired_bootstrap_interval,
    ranking_metrics,
    ranking_tie_stats,
    rating_metrics,
    recommendation_stats,
)


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


def test_paired_bootstrap_preserves_user_level_pairing():
    baseline = pd.DataFrame({"user_id": [1, 2, 3], "ndcg_at_10": [0.1, 0.2, 0.3]})
    candidate = pd.DataFrame({"user_id": [1, 2, 3], "ndcg_at_10": [0.0, 0.1, 0.2]})

    interval = paired_bootstrap_interval(candidate, baseline, "ndcg_at_10")

    assert interval["users"] == 3
    assert interval["meanDifference"] == pytest.approx(-0.1)
    assert interval["confidence95"] == pytest.approx([-0.1, -0.1])


def test_ranking_tie_stats_distinguish_full_and_partial_exact_ties():
    recommendations = {
        1: [5.0, 5.0, 5.0],
        2: [5.0, 5.0, 4.5],
        3: [4.9, 4.8, 4.7],
    }

    stats = ranking_tie_stats(recommendations)

    assert stats["lists"] == 3
    assert stats["fullyTiedListShare"] == pytest.approx(1 / 3)
    assert stats["entriesInExactTieShare"] == pytest.approx(5 / 9)
    assert stats["largestTieGroup"] == 3


import pytest
