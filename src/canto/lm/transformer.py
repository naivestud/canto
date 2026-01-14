"""Transformer blocks for the acoustic-token language model."""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .feedforward import FeedForward
from .kv_cache import KVCache


class TransformerBlock(nn.Module):
    """Pre-norm transformer decoder block: attention then feed-forward."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x),
            key_padding_mask=key_padding_mask,
            kv_cache=kv_cache,
            layer_index=layer_index,
        )
        x = x + self.ff(self.ln2(x))
        return x


class TransformerBackbone(nn.Module):
    """A stack of :class:`TransformerBlock` followed by a final layer norm."""

    def __init__(self, dim: int, num_layers: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, num_heads, dropout) for _ in range(num_layers)
        )
        self.ln_f = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        for index, block in enumerate(self.blocks):
            x = block(
                x,
                key_padding_mask=key_padding_mask,
                kv_cache=kv_cache,
                layer_index=index,
            )
        return self.ln_f(x)
