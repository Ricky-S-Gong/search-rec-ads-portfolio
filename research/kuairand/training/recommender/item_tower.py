"""Adapt Ziqi's pinned ItemTower to raw IDs and persisted training-only features."""

import numpy as np
import torch
from torch import nn

from .vendor.ziqi.item_tower import ItemTower

CATEGORICAL = (
    "item_id",
    "author_id",
    "video_type",
    "upload_type",
    "visible_status",
    "music_id",
    "music_type",
    "tag",
)
NUMERIC = ("video_duration", "server_width", "server_height")


def fit_item_encoding(table, training_item_ids):
    """Fit vocabularies/median/mean/std only on training-referenced static items.

    All candidate items are transformed with the same frozen parameters. Unknown
    categories and padding map to zero. This corrects upstream full-catalog fit.
    """
    table = table.sort_by([("item_id", "ascending")])
    ids = np.array(table["item_id"])
    fit = np.isin(ids, training_item_ids)
    if not fit.any():
        raise ValueError("No training items to fit feature encoding")
    categorical, vocabularies = [], []
    for name in CATEGORICAL:
        values = ["__missing__" if v is None else str(v) for v in table[name].to_pylist()]
        vocabulary = {
            v: i + 1 for i, v in enumerate(sorted({v for v, keep in zip(values, fit) if keep}))
        }
        categorical.append([vocabulary.get(v, 0) for v in values])
        vocabularies.append(vocabulary)
    dates = np.array(table["upload_dt"].to_pylist(), dtype="datetime64[D]")
    if np.isnat(dates).any():
        raise ValueError("Invalid upload date")
    date_origin = dates[fit].min()
    numeric = np.column_stack(
        [np.array(table[n], dtype=float) for n in NUMERIC] + [(dates - date_origin).astype(float)]
    )
    if np.isinf(numeric).any():
        raise ValueError("Infinite item metadata")
    medians = np.array(
        [
            np.nanmedian(numeric[fit, i]) if np.isfinite(numeric[fit, i]).any() else 0
            for i in range(4)
        ]
    )
    numeric = np.where(np.isnan(numeric), medians, numeric)
    means, std = numeric[fit].mean(axis=0), numeric[fit].std(axis=0)
    std[std == 0] = 1
    normalized = ((numeric - means) / std).astype(np.float32)
    return {
        "item_ids": ids.tolist(),
        "categorical": np.array(categorical).T.tolist(),
        "numeric": normalized.tolist(),
        "cardinalities": [len(v) + 1 for v in vocabularies],
        "vocabularies": vocabularies,
        "median": medians.tolist(),
        "mean": means.tolist(),
        "std": std.tolist(),
        "upload_day_origin": str(date_origin),
        "fit_item_ids": ids[fit].tolist(),
        "fit_scope": "train events only; no validation/test fitting",
    }


class ItemAdapter(nn.Module):
    def __init__(self, config, manifest):
        super().__init__()
        encoding = manifest["item_encoding"]
        self.register_buffer("raw_item_ids", torch.tensor(encoding["item_ids"], dtype=torch.long))
        # Row zero is genuine padding, distinct from real raw item ID zero.
        self.register_buffer(
            "categorical",
            torch.cat(
                [
                    torch.zeros(1, 8, dtype=torch.long),
                    torch.tensor(encoding["categorical"], dtype=torch.long),
                ]
            ),
        )
        self.register_buffer(
            "numeric",
            torch.cat([torch.zeros(1, 4), torch.tensor(encoding["numeric"], dtype=torch.float32)]),
        )
        self.tower = ItemTower(
            encoding["cardinalities"],
            4,
            config["model"]["output_dim"],
            config["model"]["item_categorical_embedding_dim"],
        )

    def encode_items(self, features):
        ids = features["item_id"]
        positions = torch.searchsorted(self.raw_item_ids, ids)
        valid = (positions < len(self.raw_item_ids)) & (ids >= 0)
        safe = positions.clamp(max=len(self.raw_item_ids) - 1)
        valid &= self.raw_item_ids[safe] == ids
        if ((ids >= 0) & ~valid).any():
            raise ValueError("Unknown candidate ID")
        rows = torch.where(valid, positions + 1, 0)
        return self.tower(self.categorical[rows], self.numeric[rows])


def build_item_tower(config, manifest):
    """Shared factory; use the verified prepared manifest, never refit on resume."""
    return ItemAdapter(config, manifest)
