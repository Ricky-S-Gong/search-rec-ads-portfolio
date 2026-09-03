import math

import numpy as np
import pandas as pd
import pytest

from goodbooks_mf.evaluation import (
    evaluate_ranking,
    evaluate_ratings,
    mae,
    ndcg_at_k,
    precision_at_k,
    prepare_ranking_data,
    recall_at_k,
    rmse,
)


class ScoreModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, user_idx, item_idx):
        users = np.asarray(user_idx, dtype=int)
        items = np.asarray(item_idx, dtype=int)
        self.calls.append((users.copy(), items.copy()))
        return np.array([self.scores.get((int(user), int(item)), 0.0) for user, item in zip(users, items)])


def test_scalar_metrics_match_hand_calculations():
    assert rmse([5, 3, 1], [4, 3, 3]) == pytest.approx(math.sqrt(5 / 3))
    assert mae([5, 3, 1], [4, 3, 3]) == 1.0
    assert precision_at_k([10, 99, 20], {10, 20}, 3) == pytest.approx(2 / 3)
    assert recall_at_k([10, 99, 20], {10, 20, 30}, 3) == pytest.approx(2 / 3)
    expected_ndcg = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([10, 99, 20], {10, 20}, 3) == pytest.approx(expected_ndcg)


def test_rating_evaluation_ignores_zero_ratings_and_clips_predictions():
    test = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "item_idx": [0, 1, 2],
            "rating": [5, 0, 1],
        }
    )
    model = ScoreModel({(0, 0): 7, (0, 1): 4, (1, 2): -2})

    result = evaluate_ratings(model, test)

    assert result == {
        "rmse": 0.0,
        "mae": 0.0,
        "evaluated_rating_count": 2,
        "evaluated_rating_users": 2,
    }
    assert model.calls[0][1].tolist() == [0, 2]


def test_ranking_data_uses_full_train_catalog_and_fixed_relevance():
    train = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 2],
            "item_idx": [0, 1, 2, 3],
            "rating": [5, 0, 4, 3],
        }
    )
    test = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 3],
            "item_idx": [2, 3, 0, 3, 1, 2],
            "rating": [4, 0, 0, 3, 2, 5],
            "is_read": [False, True, True, True, True, False],
        }
    )

    left = prepare_ranking_data(train, test)
    right = prepare_ranking_data(train, test)

    assert left.catalog.tolist() == [0, 1, 2, 3]
    assert left.users.tolist() == [0, 1, 3]
    assert left.seen_by_user == {0: {0, 1}, 1: {2}, 2: {3}}
    assert left.relevant_by_user == {0: {2, 3}, 1: {0}, 3: {2}}
    assert np.array_equal(left.catalog, right.catalog)
    assert np.array_equal(left.users, right.users)


def test_ranking_evaluation_shares_candidates_uses_raw_scores_and_breaks_ties_by_item():
    train = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "item_idx": [0, 1, 2, 3],
            "rating": [5, 4, 5, 4],
        }
    )
    test = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "item_idx": [2, 0],
            "rating": [5, 5],
            "is_read": [True, True],
        }
    )
    ranking_data = prepare_ranking_data(train, test)
    scores = {(0, 2): 9, (0, 3): 9, (1, 0): 2, (1, 1): 1}
    first = ScoreModel(scores)
    second = ScoreModel(scores)

    result = evaluate_ranking(first, ranking_data, k_values=(1, 2, 3))
    evaluate_ranking(second, ranking_data, k_values=(1, 2, 3))

    assert result["candidate_policy"] == "full_train_catalog_excluding_seen"
    assert result["catalog_size"] == 4
    assert result["evaluated_ranking_users"] == 2
    assert result["precision_at_1"] == 1.0
    assert result["recall_at_1"] == 1.0
    assert result["ndcg_at_1"] == 1.0
    assert result["precision_at_2"] == 0.5
    assert result["recall_at_2"] == 1.0
    assert result["ndcg_at_2"] == 1.0
    assert [call[1].tolist() for call in first.calls] == [[2, 3], [0, 1]]
    assert [call[1].tolist() for call in first.calls] == [call[1].tolist() for call in second.calls]


def test_ranking_uses_raw_scores_instead_of_clipped_rating_predictions():
    train = pd.DataFrame(
        {"user_idx": [0, 1, 2], "item_idx": [0, 1, 2], "rating": [5, 5, 5]}
    )
    test = pd.DataFrame(
        {"user_idx": [0], "item_idx": [2], "rating": [5], "is_read": [True]}
    )
    ranking_data = prepare_ranking_data(train, test)
    model = ScoreModel({(0, 1): 6, (0, 2): 7})

    result = evaluate_ranking(model, ranking_data, k_values=(1,))

    assert result["recall_at_1"] == 1.0


def test_users_without_relevant_candidates_are_excluded():
    train = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 1], "rating": [5, 5]})
    test = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "item_idx": [1, 0],
            "rating": [3, 0],
            "is_read": [True, False],
        }
    )

    ranking_data = prepare_ranking_data(train, test)

    assert ranking_data.users.tolist() == []
    assert evaluate_ranking(ScoreModel({}), ranking_data)["evaluated_ranking_users"] == 0
