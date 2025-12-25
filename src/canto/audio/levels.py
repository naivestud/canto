"""Amplitude measurement and normalisation helpers."""

from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    """Root-mean-square level of a signal; ``0.0`` for empty input."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def peak(x: np.ndarray) -> float:
    """Peak absolute amplitude of a signal; ``0.0`` for empty input."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def normalize_peak(x: np.ndarray, target: float = 1.0) -> np.ndarray:
    """Scale ``x`` so its peak amplitude equals ``target``.

    Silent input has no defined peak and is returned unchanged. The result is
    a new ``float32`` array with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float32)
    current = peak(x)
    if current == 0.0:
        return x.copy()
    return x * (target / current)
