"""Synthetic item tower for correctness tests only; never used in real runs."""

from torch import nn
from torch.nn import functional as F


class FixtureItem(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 64)

    def encode_items(self, features):
        return F.normalize(self.embedding(features["item_id"] % 32), dim=1)


def build(config, manifest):
    return FixtureItem()
