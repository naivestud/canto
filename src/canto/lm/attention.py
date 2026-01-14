"""Causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn

from .kv_cache import KVCache


def causal_mask(length: int, device: torch.device, cached_len: int = 0) -> torch.Tensor:
    """Boolean mask over keys; True marks positions that must not attend.

    ``cached_len`` accounts for positions already held in a key-value cache:
    query ``i`` (absolute position ``cached_len + i``) may attend to every
    position up to and including its own. With no cache this reduces to the
    usual upper-triangular mask.
    """
    rows = torch.arange(length, device=device)[:, None]
    cols = torch.arange(cached_len + length, device=device)[None, :]
    return cols > cached_len + rows


_MASK_CACHE_LIMIT = 64
_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}


def cached_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """A shared square :func:`causal_mask`, memoised per ``(length, device)``.

    Generation rebuilds the mask every step even though the shape rarely
    changes, so caching removes that allocation from the hot loop. The cache
    is cleared wholesale once it grows past a small limit.
    """
    key = (length, device)
    mask = _mask_cache.get(key)
    if mask is None:
        if len(_mask_cache) >= _MASK_CACHE_LIMIT:
            _mask_cache.clear()
        mask = causal_mask(length, device)
        _mask_cache[key] = mask
    return mask


class CausalSelfAttention(nn.Module):
    """Standard multi-head attention restricted to past positions.

    The upper triangle of the attention matrix is masked so position ``t`` can
    only attend to positions ``<= t``, which is what autoregressive generation
    requires. Pass a :class:`KVCache` to reuse keys and values computed for
    earlier positions instead of recomputing them.
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

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        cached_len = 0
        if kv_cache is not None:
            cached_len = kv_cache.length
            k, v = kv_cache.append(layer_index, k, v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if cached_len == 0:
            mask = cached_causal_mask(length, x.device)
        else:
            mask = causal_mask(length, x.device, cached_len)
        scores = scores.masked_fill(mask, float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: (B, length), True where the position is padding.
            pad_mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(pad_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.resid_drop(self.proj(out))
