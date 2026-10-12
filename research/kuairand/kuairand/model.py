"""Combine public tower interfaces and mask duplicate targets in batch negatives."""

import importlib

import torch
from torch import nn
from torch.nn import functional as F

from .user_tower import UserTower


def load_item_factory(spec):
    """Resolve Ziqi's module:function factory, accepting config and manifest."""
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


class TwoTower(nn.Module):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def encode_users(self, features, history=None, mask=None):
        return self.user_tower.encode_users(features, history, mask)

    def encode_items(self, features):
        vectors = self.item_tower.encode_items(features)
        if vectors.shape != (len(features["item_id"]), 64):
            raise ValueError("Item tower must return [N,64]")
        if not torch.isfinite(vectors).all():
            raise ValueError("Non-finite item vectors")
        return F.normalize(vectors.float(), dim=1)

    def forward(self, batch):
        history = None
        if "history_item_ids" in batch:
            history = self.item_tower.encode_items({"item_id": batch["history_item_ids"]})
        users = self.encode_users(batch, history=history, mask=batch.get("history_mask"))
        return users, self.encode_items(batch)


def build_model(config, manifest):
    """Build both towers; item implementation is provided by Ziqi, not duplicated."""
    factory = load_item_factory(config["model"]["item_factory"])
    return TwoTower(UserTower(manifest["user_ids"], config["model"]), factory(config, manifest))


def batch_loss(user_vectors, item_vectors, item_ids, temperature):
    """Positive diagonal CE; same-ID off-diagonal entries cannot be negatives."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = user_vectors.float() @ item_vectors.float().T / temperature
    repeated = item_ids[:, None] == item_ids[None, :]
    repeated.fill_diagonal_(False)
    logits = logits.masked_fill(repeated, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(item_ids), device=logits.device))
