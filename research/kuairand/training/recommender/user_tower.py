"""User-ID/context encoder with optional masked mean of shared item history vectors."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SafeBatchNorm(nn.BatchNorm1d):
    """Use running statistics for a singleton final batch rather than dropping it."""

    def forward(self, x):
        if self.training and x.shape[0] == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(x)


class UserTower(nn.Module):
    """Map raw IDs and request context to normalized [B,64] vectors.

    user_ids is the training-only vocabulary; unknown IDs use index zero.
    time_ms is Unix milliseconds; context uses the configured fixed UTC offset.
    history/mask enable masked mean pooling when configured; empty histories use zero.
    """

    def __init__(self, user_ids, config):
        super().__init__()
        self.register_buffer("user_ids", torch.tensor(sorted(set(user_ids)), dtype=torch.long))
        self.history_pooling = config.get("history_pooling", "none")
        self.history_dim = config["output_dim"] if self.history_pooling == "mean" else 0
        self.offset = config["timezone_offset_minutes"]
        self.tabs = config["num_tabs"]
        self.user_embedding = nn.Embedding(len(self.user_ids) + 1, config["user_embedding_dim"])
        self.tab_embedding = nn.Embedding(self.tabs + 1, config["context_embedding_dim"])
        width = (
            config["user_embedding_dim"] + config["context_embedding_dim"] + 4 + self.history_dim
        )
        layers = []
        for hidden in config["hidden_dims"]:
            layers.append(nn.Linear(width, hidden))
            if config["batch_norm"]:
                layers.append(SafeBatchNorm(hidden))
            layers.extend([nn.ReLU(), nn.Dropout(config["dropout"])])
            width = hidden
        layers.append(nn.Linear(width, config["output_dim"]))
        self.mlp = nn.Sequential(*layers)

    def encode_users(self, user_features, history=None, mask=None):
        """Encode aligned user_id/time_ms/tab vectors; empty histories are valid."""
        users = user_features["user_id"].long()
        tabs = user_features["tab"].long()
        timestamps = user_features["time_ms"]
        if users.ndim != 1 or tabs.shape != users.shape or timestamps.shape != users.shape:
            raise ValueError("user_id/time_ms/tab must be aligned [B] tensors")
        if mask is not None and (mask.ndim != 2 or mask.shape[0] != len(users)):
            raise ValueError("history mask must have shape [B,L]")
        if len(self.user_ids):
            positions = torch.searchsorted(self.user_ids, users)
            safe = positions.clamp(max=len(self.user_ids) - 1)
            indices = torch.where(
                (positions < len(self.user_ids)) & (self.user_ids[safe] == users), positions + 1, 0
            )
        else:
            indices = torch.zeros_like(users)
        tab_indices = torch.where((tabs >= 0) & (tabs < self.tabs), tabs + 1, 0)
        seconds = timestamps.double() / 1000 + self.offset * 60
        day = torch.remainder(seconds, 86400) / 86400 * (2 * math.pi)
        # Unix epoch was Thursday; shift to Monday for weekly context.
        week = torch.remainder(seconds + 3 * 86400, 7 * 86400) / (7 * 86400) * (2 * math.pi)
        context = torch.stack([day.sin(), day.cos(), week.sin(), week.cos()], dim=1).float()
        x = torch.cat([self.user_embedding(indices), self.tab_embedding(tab_indices), context], 1)
        if self.history_dim:
            pooled = torch.zeros((len(users), self.history_dim), device=x.device)
            if history is not None:
                if mask is None or history.shape != (*mask.shape, self.history_dim):
                    raise ValueError("History vectors/mask must be aligned [B,L,64]/[B,L]")
                pooled = (history.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(
                    1, keepdim=True
                ).clamp(min=1)
            x = torch.cat([x, pooled], dim=1)
        return F.normalize(self.mlp(x).float(), dim=1)

    forward = encode_users
