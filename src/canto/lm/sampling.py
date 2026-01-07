"""Sampling utilities for drawing the next token from LM logits."""

from __future__ import annotations

from collections.abc import Callable

import torch

# A sampler maps a ``(B, vocab)`` logits tensor to ``(B,)`` token ids.
Sampler = Callable[[torch.Tensor], torch.Tensor]


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


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability is >= ``p``."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Shift right so the token that crosses the threshold is kept.
    remove = cumulative - probs >= p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


def min_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep tokens whose probability is at least ``p`` times the top token's.

    Unlike top-p the cutoff adapts to how peaked the distribution is: a
    confident distribution keeps few tokens, a flat one keeps many. ``p <= 0``
    disables the filter.
    """
    if p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    top = probs.max(dim=-1, keepdim=True).values
    keep = probs >= p * top
    return torch.where(keep, logits, torch.full_like(logits, float("-inf")))


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sharpen (``< 1``) or flatten (``> 1``) the distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive; use greedy_sample for argmax")
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor, previous_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Down-weight tokens that already appear in ``previous_tokens``.

    ``penalty`` of 1.0 disables the effect; larger values discourage repeats.
    """
    if penalty == 1.0 or previous_tokens.numel() == 0:
        return logits
    logits = logits.clone()
    gathered = torch.gather(logits, -1, previous_tokens)
    gathered = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
    return logits.scatter(-1, previous_tokens, gathered)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One-stop sampling with temperature and top-k / top-p truncation."""
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
