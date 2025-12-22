"""Sample-rate conversion helpers.

canto's models run at a fixed rate, so callers with audio from elsewhere need
a lightweight converter. Linear interpolation is far from audiophile grade but
is dependency-free and adequate for voice prompts.
"""

from __future__ import annotations

import numpy as np


def resample(x: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample a 1-D float signal to ``target_rate`` via linear interpolation.

    Returns a new ``float32`` array. The output length follows from preserving
    the signal's duration, so ``len(out)`` is roughly
    ``len(x) * target_rate / orig_rate``.
    """
    if orig_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if orig_rate == target_rate or x.size == 0:
        return x.copy()

    n_out = int(round((x.size - 1) * target_rate / orig_rate)) + 1
    positions = np.linspace(0.0, x.size - 1, n_out)
    return np.interp(positions, np.arange(x.size), x).astype(np.float32)
