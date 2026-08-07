"""Vector quantisation for the codec latent space."""

from __future__ import annotations

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    """A single codebook mapping latent vectors to discrete codes.

    The latent is projected to ``codebook_dim``, snapped to the nearest codebook
    entry, and projected back. Gradients flow through the non-differentiable
    lookup with the straight-through estimator.
    """

    def __init__(self, dim: int, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.in_proj = nn.Conv1d(dim, codebook_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(codebook_dim, dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        # (B, dim, T) -> (B, T, codebook_dim)
        return self.in_proj(x).transpose(1, 2)

    def _nearest(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, codebook_dim) -> indices (B, T)
        emb = self.codebook.weight  # (K, D)
        dist = (
            z.pow(2).sum(-1, keepdim=True)
            - 2.0 * z @ emb.t()
            + emb.pow(2).sum(-1, keepdim=True).t()
        )
        return dist.argmin(dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map latents ``(B, dim, T)`` to codebook indices ``(B, T)``."""
        return self._nearest(self._project(x))

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map indices ``(B, T)`` back to latents ``(B, dim, T)``."""
        quantized = self.codebook(indices)  # (B, T, codebook_dim)
        return self.out_proj(quantized.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(quantized, indices, loss)`` for latents ``(B, dim, T)``."""
        z = self._project(x)
        indices = self._nearest(z)
        quantized = self.codebook(indices)

        # Straight-through: copy gradients of the quantised signal to z.
        quantized_st = z + (quantized - z).detach()

        commitment = torch.mean((quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((quantized - z.detach()) ** 2)
        loss = commitment + codebook_loss

        out = self.out_proj(quantized_st.transpose(1, 2))
        return out, indices, loss
