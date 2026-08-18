from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]


def load_movielens(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the official MovieLens 1M double-colon files."""
    ratings_path = data_dir / "ratings.dat"
    movies_path = data_dir / "movies.dat"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError("MovieLens ratings.dat and movies.dat are required")
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=RATING_COLUMNS,
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"},
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=MOVIE_COLUMNS,
        encoding="latin-1",
        dtype={"movie_id": "int32", "title": "string", "genres": "string"},
    )
    validate_ratings(ratings)
    return ratings, movies


def validate_ratings(ratings: pd.DataFrame) -> None:
    missing = set(RATING_COLUMNS).difference(ratings.columns)
    if missing:
        raise ValueError(f"missing rating columns: {', '.join(sorted(missing))}")
    if ratings.empty:
        raise ValueError("ratings must not be empty")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("duplicate user-movie ratings are not allowed")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("ratings must be in the inclusive range [1, 5]")
    if ratings[["user_id", "movie_id", "timestamp"]].isna().any().any():
        raise ValueError("rating identifiers and timestamps must not be missing")


def dataset_profile(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """Return compact, JSON-safe profile values used by the website."""
    users = int(ratings["user_id"].nunique())
    rated_movies = int(ratings["movie_id"].nunique())
    observed = int(len(ratings))
    per_user = ratings.groupby("user_id", observed=True).size()
    per_movie = ratings.groupby("movie_id", observed=True).size().sort_values(ascending=False)
    genre_counts = Counter(
        genre for value in movies["genres"].dropna() for genre in str(value).split("|")
    )
    distribution = ratings["rating"].value_counts().sort_index()
    return {
        "users": users,
        "moviesInMetadata": int(len(movies)),
        "ratedMovies": rated_movies,
        "ratings": observed,
        "ratingScale": [1, 5],
        "density": round(observed / (users * rated_movies), 6),
        "sparsity": round(1 - observed / (users * rated_movies), 6),
        "timestampRange": [int(ratings["timestamp"].min()), int(ratings["timestamp"].max())],
        "ratingsPerUser": {
            "minimum": int(per_user.min()),
            "median": round(float(per_user.median()), 1),
            "mean": round(float(per_user.mean()), 1),
            "maximum": int(per_user.max()),
        },
        "ratingsPerMovie": {
            "minimum": int(per_movie.min()),
            "median": round(float(per_movie.median()), 1),
            "mean": round(float(per_movie.mean()), 1),
            "maximum": int(per_movie.max()),
        },
        "ratingDistribution": {
            str(int(stars)): {"count": int(count), "share": round(float(count / observed), 6)}
            for stars, count in distribution.items()
        },
        "popularity": {
            "top1PercentShare": round(float(per_movie.iloc[: max(1, round(rated_movies * 0.01))].sum() / observed), 6),
            "top10PercentShare": round(float(per_movie.iloc[: max(1, round(rated_movies * 0.10))].sum() / observed), 6),
            "moviesBelow10Ratings": int((per_movie < 10).sum()),
        },
        "genres": {"count": len(genre_counts), "top": genre_counts.most_common(8)},
        "fields": [
            {
                "table": "ratings",
                "name": name,
                "dtype": str(ratings[name].dtype),
                "example": ratings[name].iloc[0],
            }
            for name in RATING_COLUMNS
        ] + [
            {
                "table": "movies",
                "name": name,
                "dtype": str(movies[name].dtype),
                "example": movies[name].iloc[0],
            }
            for name in MOVIE_COLUMNS
        ],
    }
