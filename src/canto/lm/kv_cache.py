"""Key-value cache for autoregressive generation.

During decoding, the keys and values of past positions never change, so
recomputing them at every step is wasted work. :class:`KVCache` stores them
per transformer layer as ``(B, heads, T, head_dim)`` tensors and grows by one
position per generated token.
"""

from __future__ import annotations

import torch


class KVCache:
    """Per-layer store of attention keys and values."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    @property
    def length(self) -> int:
        """Number of cached positions (``0`` before the first append)."""
        first = self._keys[0]
        return 0 if first is None else first.shape[2]

    def append(
        self, layer: int, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate new keys/values with the cached ones for ``layer``.

        Returns the full ``(B, heads, T_total, head_dim)`` pair to attend over.
        """
        prev_keys, prev_values = self._keys[layer], self._values[layer]
        if prev_keys is not None and prev_values is not None:
            keys = torch.cat([prev_keys, keys], dim=2)
            values = torch.cat([prev_values, values], dim=2)
        self._keys[layer] = keys
        self._values[layer] = values
        return keys, values

    def crop(self, max_length: int) -> None:
        """Keep only the most recent ``max_length`` positions in every layer."""
        for layer in range(self.num_layers):
            keys, values = self._keys[layer], self._values[layer]
            if keys is not None and values is not None and keys.shape[2] > max_length:
                self._keys[layer] = keys[:, :, -max_length:, :]
                self._values[layer] = values[:, :, -max_length:, :]

    def reset(self) -> None:
        """Drop all cached positions."""
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers
