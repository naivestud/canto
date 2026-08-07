"""Convolutional building blocks shared by the codec encoder and decoder."""

from __future__ import annotations

import torch
from torch import nn


def wn_conv1d(*args, **kwargs) -> nn.Conv1d:
    """A weight-normalised 1-D convolution."""
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def wn_conv_transpose1d(*args, **kwargs) -> nn.ConvTranspose1d:
    """A weight-normalised 1-D transposed convolution."""
    return nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualConvBlock(nn.Module):
    """Two dilated convolutions with a residual connection.

    The block keeps the channel count and temporal resolution unchanged, so it
    can be stacked freely between the (de)convolutional layers that change
    resolution.
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size * dilation - dilation) // 2
        self.net = nn.Sequential(
            nn.ELU(),
            wn_conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad),
            nn.ELU(),
            wn_conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
