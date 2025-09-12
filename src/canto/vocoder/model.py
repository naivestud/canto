"""Token-to-waveform vocoder.

The vocoder embeds discrete acoustic tokens and up-samples them with transposed
convolutions until the temporal resolution matches the original waveform. The
product of ``upsample_rates`` must equal the codec hop length so that a frame
sequence of length ``T`` becomes ``T * hop`` samples. A multi-period refinement
stage then smooths the up-sampled signal before the final projection.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..codec.layers import wn_conv1d, wn_conv_transpose1d


class MultiPeriodRefinement(nn.Module):
    """Dilated residual convolutions at several periods, summed together.

    Each branch sees the signal at a different effective period through its
    dilation, which helps the generator commit to consistent local structure at
    multiple time scales.
    """

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in dilations:
            kernel = 5
            pad = (kernel * dilation - dilation) // 2
            self.branches.append(
                nn.Sequential(
                    nn.ELU(),
                    wn_conv1d(channels, channels, kernel, dilation=dilation, padding=pad),
                    nn.ELU(),
                    wn_conv1d(channels, channels, kernel, dilation=1, padding=(kernel - 1) // 2),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = x
        for branch in self.branches:
            refined = refined + branch(refined)
        return refined


class Vocoder(nn.Module):
    """A HiFi-GAN-style generator conditioned on discrete tokens."""

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int = 32,
        channels: int = 64,
        upsample_rates: tuple[int, ...] = (4, 2),
        kernel_size: int = 7,
        refinements: tuple[int, ...] = (1, 3, 5),
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
        self.upsample = nn.Sequential(*layers)
        self.refine = MultiPeriodRefinement(channels, refinements)
        self.head = nn.Sequential(
            nn.ELU(),
            wn_conv1d(channels, 1, kernel_size, padding=pad),
            nn.Tanh(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Render token indices ``(B, T)`` into a waveform ``(B, 1, T * hop)``."""
        x = self.embedding(tokens).transpose(1, 2)
        x = self.upsample(x)
        x = self.refine(x)
        return self.head(x)

    @classmethod
    def from_config(cls, config, num_tokens: int) -> Vocoder:
        """Build a vocoder from a :class:`VocoderConfig` and a token count."""
        return cls(
            num_tokens=num_tokens,
            embed_dim=config.embed_dim,
            channels=config.channels,
            upsample_rates=config.upsample_rates,
            refinements=config.refinements,
        )
