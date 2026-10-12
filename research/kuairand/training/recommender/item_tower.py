"""Adapt Ziqi's pinned ItemTower to raw IDs and persisted training-only features."""

import numpy as np
import torch
from torch import nn

from .vendor.data_pipeline_snapshot.item_tower import ItemTower

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


def fit_item_encoding(
    table, training_item_ids, categorical_names=CATEGORICAL, numeric_names=(*NUMERIC, "upload_day")
):
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
    for name in categorical_names:
        values = ["__missing__" if v is None else str(v) for v in table[name].to_pylist()]
        vocabulary = {
            v: i + 1 for i, v in enumerate(sorted({v for v, keep in zip(values, fit) if keep}))
        }
        categorical.append([vocabulary.get(v, 0) for v in values])
        vocabularies.append(vocabulary)
    date_origin = None
    columns = []
    for name in numeric_names:
        if name == "upload_day":
            dates = np.array(table["upload_dt"].to_pylist(), dtype="datetime64[D]")
            if np.isnat(dates).any():
                raise ValueError("Invalid upload date")
            date_origin = dates[fit].min()
            columns.append((dates - date_origin).astype(float))
        else:
            columns.append(np.array(table[name], dtype=float))
    numeric = np.column_stack(columns) if columns else np.empty((len(ids), 0))
    if np.isinf(numeric).any():
        raise ValueError("Infinite item metadata")
    medians = np.array(
        [
            np.nanmedian(numeric[fit, i]) if np.isfinite(numeric[fit, i]).any() else 0
            for i in range(len(numeric_names))
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
        "upload_day_origin": str(date_origin) if date_origin is not None else None,
        "categorical_names": list(categorical_names),
        "numeric_names": list(numeric_names),
        "fit_item_ids": ids[fit].tolist(),
        "fit_scope": "train events only; no validation/test fitting",
    }


class ItemAdapter(nn.Module):
    def __init__(self, config, manifest):
        super().__init__()
        encoding = manifest["item_encoding"]
        for key, stored in (
            ("item_categorical_features", "categorical_names"),
            ("item_numeric_features", "numeric_names"),
        ):
            if key in config["model"] and config["model"][key] != encoding.get(stored):
                raise ValueError("Configured item features differ from frozen manifest")
        self.register_buffer("raw_item_ids", torch.tensor(encoding["item_ids"], dtype=torch.long))
        # Row zero is genuine padding, distinct from real raw item ID zero.
        self.register_buffer(
            "categorical",
            torch.cat(
                [
                    torch.zeros(1, len(encoding["cardinalities"]), dtype=torch.long),
                    torch.tensor(encoding["categorical"], dtype=torch.long),
                ]
            ),
        )
        self.register_buffer(
            "numeric",
            torch.cat(
                [
                    torch.zeros(1, len(encoding["median"])),
                    torch.tensor(encoding["numeric"], dtype=torch.float32),
                ]
            ),
        )
        numeric_count = len(encoding["median"])
        if numeric_count:
            self.tower = ItemTower(
                encoding["cardinalities"],
                numeric_count,
                config["model"]["output_dim"],
                config["model"]["item_categorical_embedding_dim"],
            )
        else:
            if encoding.get("categorical_names") != ["item_id"]:
                raise ValueError("ID-only tower requires exactly item_id")
            self.tower = IdItemTower(
                encoding["cardinalities"][0], config["model"]["item_categorical_embedding_dim"]
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


class IdItemTower(nn.Module):
    """The same two-layer item MLP with only an ID embedding as input."""

    def __init__(self, cardinality, width):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, width, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 64))

    def forward(self, categorical, numeric):
        return torch.nn.functional.normalize(self.mlp(self.embedding(categorical[..., 0])), dim=-1)
