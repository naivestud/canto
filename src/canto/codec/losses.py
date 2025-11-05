"""Losses for vector quantisation."""

from __future__ import annotations

import torch


def quantization_loss(quantized: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Commitment plus codebook loss between codes and encoder outputs.

    ``quantized`` are the looked-up codebook vectors and ``z`` the projected
    encoder latents. The commitment term pulls the encoder towards the codes;
    the codebook term pulls the codes towards the encoder.
    """
    commitment = torch.mean((quantized.detach() - z) ** 2)
    codebook = torch.mean((quantized - z.detach()) ** 2)
    return commitment + codebook
