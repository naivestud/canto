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


def trim_silence(x: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Drop leading and trailing samples whose magnitude stays below ``threshold``.

    Useful for tightening voice prompts before cloning. Input that never
    crosses the threshold comes back as an empty array.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    loud = np.flatnonzero(np.abs(flat) >= threshold)
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)
    return flat[loud[0] : loud[-1] + 1].copy()
