from __future__ import annotations

import numpy as np
import pandas as pd


def per_user_temporal_split(
    interactions: pd.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each timeline at explicit-rating boundaries, then remove cold items."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave positive train, validation, and test parts")
    ordered = interactions.sort_values(
        ["user_idx", "event_time", "item_idx"], kind="stable"
    ).reset_index(drop=True)
    pieces: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    for _, group in ordered.groupby("user_idx", observed=True, sort=True):
        group = group.reset_index(drop=True)
        explicit_positions = np.flatnonzero(group["rating"].to_numpy() > 0)
        if len(explicit_positions) < 3:
            raise ValueError("every user needs at least three explicit ratings")
        train_end = int(np.clip(np.floor(len(explicit_positions) * train_fraction), 1, len(explicit_positions) - 2))
        validation_end = int(
            np.clip(
                np.floor(len(explicit_positions) * (train_fraction + validation_fraction)),
                train_end + 1,
                len(explicit_positions) - 1,
            )
        )
        last_train_position = explicit_positions[train_end - 1]
        last_validation_position = explicit_positions[validation_end - 1]
        pieces["train"].append(group.iloc[: last_train_position + 1])
        pieces["validation"].append(group.iloc[last_train_position + 1 : last_validation_position + 1])
        pieces["test"].append(group.iloc[last_validation_position + 1 :])
    train = pd.concat(pieces["train"], ignore_index=True)
    known_items = set(train["item_idx"])
    validation = pd.concat(pieces["validation"], ignore_index=True)
    test = pd.concat(pieces["test"], ignore_index=True)
    validation = validation[validation["item_idx"].isin(known_items)].reset_index(drop=True)
    test = test[test["item_idx"].isin(known_items)].reset_index(drop=True)
    return train, validation, test
