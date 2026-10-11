"""The static-feature Item Tower shared by target and history videos."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ItemTower(nn.Module):
    """Encode item metadata into L2-normalized retrieval vectors."""

    def __init__(
        self,
        categorical_cardinalities: list[int] | tuple[int, ...],
        numeric_feature_count: int,
        embedding_dim: int = 64,
        categorical_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if not categorical_cardinalities or min(categorical_cardinalities) < 2:
            raise ValueError("categorical cardinalities must reserve index 0 and include one value")
        if numeric_feature_count < 1:
            raise ValueError("numeric_feature_count must be positive")
        self.categorical_feature_count = len(categorical_cardinalities)
        self.numeric_feature_count = numeric_feature_count
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, categorical_embedding_dim, padding_idx=0) for cardinality in categorical_cardinalities]
        )
        input_dim = self.categorical_feature_count * categorical_embedding_dim + numeric_feature_count
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        if categorical.shape[:-1] != numeric.shape[:-1]:
            raise ValueError("categorical and numeric leading dimensions must match")
        if categorical.shape[-1] != self.categorical_feature_count:
            raise ValueError("unexpected categorical feature count")
        if numeric.shape[-1] != self.numeric_feature_count:
            raise ValueError("unexpected numeric feature count")
        embedded = [layer(categorical[..., index].long()) for index, layer in enumerate(self.embeddings)]
        features = torch.cat([*embedded, numeric.float()], dim=-1)
        return F.normalize(self.mlp(features), p=2, dim=-1)
