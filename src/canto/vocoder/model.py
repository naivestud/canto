"""Token-to-waveform vocoder.

The vocoder embeds discrete acoustic tokens and up-samples them with transposed
convolutions until the temporal resolution matches the original waveform. The
product of ``upsample_rates`` must equal the codec hop length so that a frame
sequence of length ``T`` becomes ``T * hop`` samples.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..codec.layers import wn_conv1d, wn_conv_transpose1d


class Vocoder(nn.Module):
    """A HiFi-GAN-style generator conditioned on discrete tokens."""

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int = 32,
        channels: int = 64,
        upsample_rates: tuple[int, ...] = (4, 2),
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.upsample_factor = math.prod(upsample_rates)
        self.embedding = nn.Embedding(num_tokens, embed_dim)

        pad = (kernel_size - 1) // 2
        layers: list[nn.Module] = [wn_conv1d(embed_dim, channels, kernel_size, padding=pad)]
        for rate in upsample_rates:
            layers.append(nn.ELU())
            layers.append(
                wn_conv_transpose1d(
                    channels,
                    channels,
                    kernel_size=2 * rate,
                    stride=rate,
                    padding=rate // 2 + rate % 2,
                    output_padding=rate % 2,
                )
            )
        layers.append(nn.ELU())
        layers.append(wn_conv1d(channels, 1, kernel_size, padding=pad))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Render token indices ``(B, T)`` into a waveform ``(B, 1, T * hop)``."""
        x = self.embedding(tokens).transpose(1, 2)
        return self.net(x)
