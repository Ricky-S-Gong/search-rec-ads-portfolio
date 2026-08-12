from __future__ import annotations

import numpy as np
import pandas as pd


def per_user_temporal_split(
    ratings: pd.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split every user's timeline deterministically into train/validation/test."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave positive train, validation, and test parts")
    ordered = ratings.sort_values(["user_id", "timestamp", "movie_id"], kind="stable").copy()
    position = ordered.groupby("user_id", observed=True).cumcount().to_numpy()
    count = ordered.groupby("user_id", observed=True)["user_id"].transform("size").to_numpy()
    if (count < 3).any():
        raise ValueError("every user needs at least three interactions")
    train_end = np.floor(count * train_fraction).astype(int)
    validation_end = np.floor(count * (train_fraction + validation_fraction)).astype(int)
    train_end = np.clip(train_end, 1, count - 2)
    validation_end = np.clip(validation_end, train_end + 1, count - 1)
    train = ordered.loc[position < train_end]
    validation = ordered.loc[(position >= train_end) & (position < validation_end)]
    test = ordered.loc[position >= validation_end]
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))
