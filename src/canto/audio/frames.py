"""Framing helpers used by spectral analysis."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def hann_window(length: int) -> np.ndarray:
    """A periodic Hann window of ``length`` samples."""
    return np.hanning(length + 1)[:-1].astype(np.float32)


def num_frames(n_samples: int, frame_length: int, hop_length: int, center: bool = True) -> int:
    """Number of frames ``frame_signal`` will produce for a given input length."""
    if center:
        # Centre padding adds frame_length samples in total, so the count is
        # 1 + n // hop; note this is not ceil(n / hop), which is one short on
        # exact multiples of the hop length.
        return 1 + n_samples // hop_length
    return 1 + (n_samples - frame_length) // hop_length


def frame_signal(
    x: np.ndarray,
    frame_length: int,
    hop_length: int,
    center: bool = True,
) -> np.ndarray:
    """Split ``x`` into overlapping frames of shape ``(num_frames, frame_length)``.

    With ``center=True`` the signal is reflect-padded by ``frame_length // 2`` on
    each side so that frame ``t`` is centred on ``x[t * hop_length]``.
    """
    x = np.asarray(x, dtype=np.float32)
    if center:
        pad = frame_length // 2
        x = np.pad(x, (pad, pad), mode="reflect")

    n = x.shape[0]
    count = 1 + (n - frame_length) // hop_length
    if count <= 0:
        return np.zeros((0, frame_length), dtype=np.float32)

    # A strided view avoids materialising the (count, frame_length) index
    # matrix the gather used; copy so callers own their frames.
    windows = sliding_window_view(x, frame_length)
    return np.ascontiguousarray(windows[::hop_length])
