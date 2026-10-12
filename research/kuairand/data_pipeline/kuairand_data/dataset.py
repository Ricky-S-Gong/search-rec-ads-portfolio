"""Leakage-safe positive-example batches for the KuaiRand two-tower model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


CATEGORICAL_COLUMNS = (
    "item_id", "author_id", "video_type", "upload_type", "visible_status",
    "music_id", "music_type", "tag",
)
NUMERIC_COLUMNS = ("video_duration", "server_width", "server_height", "upload_day")
REQUIRED_ITEM_COLUMNS = set(CATEGORICAL_COLUMNS).union({"upload_dt", *NUMERIC_COLUMNS[:3]})


@dataclass(frozen=True)
class FeatureSchema:
    categorical_cardinalities: tuple[int, ...]
    numeric_feature_count: int = len(NUMERIC_COLUMNS)


class _PositiveExamples(Dataset[tuple[int, list[int]]]):
    def __init__(self, item_ids: list[int], histories: list[list[int]]) -> None:
        self.item_ids = item_ids
        self.histories = histories

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, index: int) -> tuple[int, list[int]]:
        return self.item_ids[index], self.histories[index]


class _ItemFeatureStore:
    def __init__(self, features: pd.DataFrame) -> None:
        missing = REQUIRED_ITEM_COLUMNS.difference(features.columns)
        if missing:
            raise ValueError(f"item features missing columns: {sorted(missing)}")
        if features["item_id"].duplicated().any():
            raise ValueError("item features contain duplicate item_id values")

        self.item_rows = {int(item_id): row for row, item_id in enumerate(features["item_id"])}
        categorical = np.zeros((len(features), len(CATEGORICAL_COLUMNS)), dtype=np.int64)
        cardinalities: list[int] = []
        for index, column in enumerate(CATEGORICAL_COLUMNS):
            values = features[column].astype("string").fillna("__missing__")
            vocabulary = {value: token + 1 for token, value in enumerate(sorted(values.unique()))}
            categorical[:, index] = values.map(vocabulary).to_numpy(dtype=np.int64)
            cardinalities.append(len(vocabulary) + 1)

        upload_dates = pd.to_datetime(features["upload_dt"], errors="coerce")
        if upload_dates.isna().any():
            raise ValueError("item features contain invalid upload_dt values")
        numeric = features.loc[:, NUMERIC_COLUMNS[:3]].apply(pd.to_numeric, errors="coerce")
        numeric["upload_day"] = (upload_dates - upload_dates.min()).dt.days
        numeric = numeric.fillna(numeric.median()).fillna(0.0).astype("float32")
        values = numeric.to_numpy()
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std == 0] = 1.0

        self.categorical = categorical
        self.numeric = (values - mean) / std
        self.schema = FeatureSchema(tuple(cardinalities))

    def validate_item_ids(self, item_ids: Iterable[int]) -> None:
        missing = {int(item_id) for item_id in item_ids}.difference(self.item_rows)
        if missing:
            raise ValueError(f"item features missing {len(missing)} referenced item IDs")

    def encode(self, item_ids: list[int | None]) -> tuple[torch.Tensor, torch.Tensor]:
        categorical = np.zeros((len(item_ids), len(CATEGORICAL_COLUMNS)), dtype=np.int64)
        numeric = np.zeros((len(item_ids), len(NUMERIC_COLUMNS)), dtype=np.float32)
        real_ids = [item_id for item_id in item_ids if item_id is not None]
        self.validate_item_ids(real_ids)
        for output_index, item_id in enumerate(item_ids):
            if item_id is not None:
                row = self.item_rows[int(item_id)]
                categorical[output_index] = self.categorical[row]
                numeric[output_index] = self.numeric[row]
        return torch.from_numpy(categorical), torch.from_numpy(numeric)


def _history_list(value: object) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("history_item_ids must be a list")
    return [int(item_id) for item_id in value]


def build_dataloader(
    events_path: str | Path,
    histories_path: str | Path,
    item_features_path: str | Path,
    *,
    split: str,
    batch_size: int,
    max_history_length: int = 50,
    shuffle: bool = False,
) -> tuple[DataLoader[tuple[int, list[int]]], FeatureSchema]:
    """Build batches from the frozen training contract for one split."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if batch_size < 1 or max_history_length < 1:
        raise ValueError("batch_size and max_history_length must be positive")

    events = pd.read_parquet(events_path, columns=["request_id", "item_id", "long_view", "is_random", "split"])
    positives = events.loc[
        (events["split"] == split) & events["long_view"] & (~events["is_random"]),
        ["request_id", "item_id"],
    ]
    histories = pd.read_parquet(histories_path, columns=["request_id", "history_item_ids"])
    if histories["request_id"].duplicated().any():
        raise ValueError("user histories contain duplicate request_id values")
    examples = positives.merge(histories, on="request_id", how="left", validate="one_to_one", indicator=True)
    missing_histories = examples.loc[examples["_merge"] != "both", "request_id"]
    if not missing_histories.empty:
        raise ValueError(f"missing histories for {len(missing_histories)} positive events")

    item_store = _ItemFeatureStore(pd.read_parquet(item_features_path))
    item_ids = examples["item_id"].astype("int64").tolist()
    histories_by_example = [_history_list(value) for value in examples["history_item_ids"]]
    item_store.validate_item_ids(item_ids)
    item_store.validate_item_ids(item_id for history in histories_by_example for item_id in history)
    dataset = _PositiveExamples(item_ids, histories_by_example)

    def collate(examples: list[tuple[int, list[int]]]) -> dict[str, torch.Tensor]:
        target_ids = [item_id for item_id, _ in examples]
        history_ids: list[int | None] = []
        history_mask = torch.zeros((len(examples), max_history_length), dtype=torch.bool)
        for row, (_, history) in enumerate(examples):
            recent = history[-max_history_length:]
            start = max_history_length - len(recent)
            history_ids.extend([None] * start + recent)
            history_mask[row, start:] = True
        target_categorical, target_numeric = item_store.encode(target_ids)
        history_categorical, history_numeric = item_store.encode(history_ids)
        return {
            "target_categorical": target_categorical,
            "target_numeric": target_numeric,
            "history_categorical": history_categorical.reshape(len(examples), max_history_length, -1),
            "history_numeric": history_numeric.reshape(len(examples), max_history_length, -1),
            "history_mask": history_mask,
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate), item_store.schema
