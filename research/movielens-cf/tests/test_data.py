from pathlib import Path

import pandas as pd
import pytest

from movielens_cf.data import load_movielens, validate_ratings


def test_load_movielens_parses_official_double_colon_files(tmp_path: Path):
    (tmp_path / "ratings.dat").write_text(
        "1::10::5::100\n1::20::4::200\n2::10::3::300\n", encoding="utf-8"
    )
    (tmp_path / "movies.dat").write_text(
        "10::Alpha (2000)::Drama|Romance\n20::Beta (2001)::Comedy\n",
        encoding="latin-1",
    )

    ratings, movies = load_movielens(tmp_path)

    assert ratings.to_dict("records")[0] == {
        "user_id": 1,
        "movie_id": 10,
        "rating": 5.0,
        "timestamp": 100,
    }
    assert movies.loc[movies["movie_id"] == 10, "genres"].item() == "Drama|Romance"


def test_validate_ratings_rejects_duplicate_user_movie_pairs():
    ratings = pd.DataFrame(
        [(1, 10, 5.0, 100), (1, 10, 4.0, 200)],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_ratings(ratings)
