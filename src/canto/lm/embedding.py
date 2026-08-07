"""Token and positional embeddings for the language model."""

from __future__ import annotations

import math

import torch
from torch import nn


class ScaledEmbedding(nn.Module):
    """Token embedding scaled by ``sqrt(dim)`` as in the Transformer paper."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.scale = math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positions added to the token embeddings."""

    table: torch.Tensor

    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        table = torch.zeros(max_len, dim)
        table[:, 0::2] = torch.sin(positions * div)
        table[:, 1::2] = torch.cos(positions * div[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positions to ``x`` of shape ``(B, T, dim)``."""
        return x + self.table[:, : x.shape[1], :]
