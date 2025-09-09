"""Sampling utilities for drawing the next token from LM logits."""

from __future__ import annotations

import torch


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Pick the most likely token. ``logits`` is ``(B, vocab)``; returns ``(B,)``."""
    return logits.argmax(dim=-1)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out every logit outside the top ``k`` entries."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, k, dim=-1)
    cutoff = values[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def top_k_sample(
    logits: torch.Tensor,
    k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from the top ``k`` entries of the distribution."""
    filtered = top_k_filter(logits, k)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
