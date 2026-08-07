"""Convolutional encoder that maps a waveform to a compact latent sequence."""

from __future__ import annotations

import torch
from torch import nn

from .layers import ResidualConvBlock, wn_conv1d


class Encoder(nn.Module):
    """Down-sampling convolutional encoder.

    Produces one latent vector per ``prod(strides)`` input samples. The channel
    count stays at ``dim`` throughout so the latent is easy to quantise.
    """

    def __init__(
        self,
        in_channels: int = 1,
        dim: int = 64,
        strides: tuple[int, ...] = (2, 4, 5, 8),
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.strides = tuple(strides)

        layers: list[nn.Module] = [wn_conv1d(in_channels, dim, kernel_size=7, padding=3)]
        for stride in self.strides:
            for _ in range(num_blocks):
                layers.append(ResidualConvBlock(dim))
            layers.append(nn.ELU())
            layers.append(
                wn_conv1d(dim, dim, kernel_size=2 * stride, stride=stride, padding=stride // 2)
            )
        layers.append(nn.ELU())
        layers.append(wn_conv1d(dim, dim, kernel_size=7, padding=3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` of shape ``(B, 1, T)`` to latents ``(B, dim, T')``."""
        return self.net(x)
