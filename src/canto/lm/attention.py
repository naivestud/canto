"""Causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Standard multi-head attention restricted to past positions.

    The upper triangle of the attention matrix is masked so position ``t`` can
    only attend to positions ``<= t``, which is what autoregressive generation
    requires.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: (B, length), True where the position is padding.
            pad_mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(pad_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.resid_drop(self.proj(out))
