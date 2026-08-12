import pandas as pd

from movielens_cf.split import per_user_temporal_split


def test_temporal_split_is_deterministic_and_keeps_future_events_out_of_training():
    ratings = pd.DataFrame(
        [
            (user_id, movie_id, float((movie_id % 5) + 1), 1_000 + movie_id)
            for user_id in (1, 2)
            for movie_id in range(1, 21)
        ],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    ).sample(frac=1, random_state=9)

    first = per_user_temporal_split(ratings)
    second = per_user_temporal_split(ratings)

    for first_part, second_part in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_part, second_part)
    train, validation, test = first
    assert train.groupby("user_id").size().to_dict() == {1: 16, 2: 16}
    assert validation.groupby("user_id").size().to_dict() == {1: 2, 2: 2}
    assert test.groupby("user_id").size().to_dict() == {1: 2, 2: 2}
    assert train.groupby("user_id").timestamp.max().lt(
        validation.groupby("user_id").timestamp.min()
    ).all()
    assert validation.groupby("user_id").timestamp.max().lt(
        test.groupby("user_id").timestamp.min()
    ).all()
