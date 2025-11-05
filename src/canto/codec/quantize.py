"""Vector quantisation for the codec latent space."""

from __future__ import annotations

import torch
from torch import nn

from .losses import quantization_loss


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

        loss = quantization_loss(quantized, z)

        out = self.out_proj(quantized_st.transpose(1, 2))
        return out, indices, loss


class ResidualVectorQuantizer(nn.Module):
    """A stack of :class:`VectorQuantizer` applied to successive residuals.

    Each level quantises what the previous levels could not represent, so more
    levels trade bandwidth for fidelity. Indices come back as
    ``(B, num_quantizers, T)``.
    """

    def __init__(self, dim: int, codebook_size: int, codebook_dim: int, num_quantizers: int) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList(
            VectorQuantizer(dim, codebook_size, codebook_dim) for _ in range(num_quantizers)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(quantized, indices, loss)`` for latents ``(B, dim, T)``."""
        residual = x
        quantized_sum = torch.zeros_like(x)
        total_loss = x.new_zeros(())
        all_indices = []
        for quantizer in self.quantizers:
            quantized, indices, loss = quantizer(residual)
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            total_loss = total_loss + loss
            all_indices.append(indices)
        return quantized_sum, torch.stack(all_indices, dim=1), total_loss

    def encode(self, x: torch.Tensor, num_quantizers: int | None = None) -> torch.Tensor:
        """Encode latents to indices ``(B, Q, T)`` using the first ``Q`` levels."""
        n = self.num_quantizers if num_quantizers is None else num_quantizers
        residual = x
        all_indices = []
        for quantizer in self.quantizers[:n]:
            indices = quantizer.encode(residual)
            residual = residual - quantizer.decode_indices(indices)
            all_indices.append(indices)
        return torch.stack(all_indices, dim=1)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices ``(B, Q, T)`` back into latents ``(B, dim, T)``."""
        quantized_sum = None
        for level, quantizer in enumerate(self.quantizers[: indices.shape[1]]):
            quantized = quantizer.decode_indices(indices[:, level, :])
            quantized_sum = quantized if quantized_sum is None else quantized_sum + quantized
        assert quantized_sum is not None
        return quantized_sum
