"""Convolutional decoder that renders a latent sequence back into a waveform."""

from __future__ import annotations

import torch
from torch import nn

from .layers import ResidualConvBlock, wn_conv1d, wn_conv_transpose1d


class Decoder(nn.Module):
    """Up-sampling convolutional decoder, the mirror of :class:`Encoder`.

    Each stride becomes a transposed convolution that increases the temporal
    resolution; a final ``tanh`` keeps samples in ``[-1, 1]``.
    """

    def __init__(
        self,
        out_channels: int = 1,
        dim: int = 64,
        strides: tuple[int, ...] = (8, 5, 4, 2),
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.strides = tuple(strides)

        layers: list[nn.Module] = [wn_conv1d(dim, dim, kernel_size=7, padding=3)]
        for stride in self.strides:
            layers.append(nn.ELU())
            layers.append(
                wn_conv_transpose1d(
                    dim,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2,
                )
            )
            for _ in range(num_blocks):
                layers.append(ResidualConvBlock(dim))
        layers.append(nn.ELU())
        layers.append(wn_conv1d(dim, out_channels, kernel_size=7, padding=3))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latents ``(B, dim, T')`` to a waveform ``(B, 1, T)``."""
        return self.net(x)
