import pandas as pd

from goodbooks_mf.split import per_user_temporal_split


def test_temporal_split_preserves_order_and_places_explicit_ratings_in_every_part():
    rows = []
    for user_idx in (0, 1):
        for position in range(10):
            rows.append(
                {
                    "user_idx": user_idx,
                    "item_idx": user_idx * 3 + position % 3,
                    "rating": 0 if position in (1, 4) else (position % 5) + 1,
                    "event_time": pd.Timestamp("2017-01-01", tz="UTC")
                    + pd.Timedelta(days=position),
                }
            )
    frame = pd.DataFrame(rows)

    train, validation, test = per_user_temporal_split(frame)

    for user_idx in (0, 1):
        user_train = train[train["user_idx"] == user_idx]
        user_validation = validation[validation["user_idx"] == user_idx]
        user_test = test[test["user_idx"] == user_idx]
        assert (user_train["rating"] > 0).any()
        assert (user_validation["rating"] > 0).any()
        assert (user_test["rating"] > 0).any()
        assert user_train["event_time"].max() < user_validation["event_time"].min()
        assert user_validation["event_time"].max() < user_test["event_time"].min()


def test_temporal_split_removes_validation_and_test_items_missing_from_train_catalog():
    frame = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 0, 0],
            "item_idx": [1, 2, 1, 2, 99, 1],
            "rating": [5, 4, 5, 4, 5, 4],
            "event_time": pd.date_range("2017-01-01", periods=6, tz="UTC"),
        }
    )

    train, validation, test = per_user_temporal_split(frame, train_fraction=0.5, validation_fraction=0.25)

    known = set(train["item_idx"])
    assert set(validation["item_idx"]).issubset(known)
    assert set(test["item_idx"]).issubset(known)
    assert 99 not in set(validation["item_idx"]) | set(test["item_idx"])
