"""Transformer blocks for the acoustic-token language model."""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm transformer decoder block: attention then feed-forward."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
