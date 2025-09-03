"""Position-wise feed-forward network."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    """Two linear layers with a GELU in between, expanding to ``4 * dim``."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
