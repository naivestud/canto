"""Minimal WAV reading and writing on top of the standard-library ``wave`` module.

Only 16-bit PCM is supported, which is what canto produces and consumes. Float
audio is normalised to ``[-1.0, 1.0]``.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..errors import AudioIOError, UnsupportedFormatError


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file.

    Returns a ``(samples, sample_rate)`` tuple where ``samples`` is a 1-D
    ``float32`` array in ``[-1.0, 1.0]``. Multi-channel audio is down-mixed by
    averaging channels.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            n_channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            n_frames = handle.getnframes()
            raw = handle.readframes(n_frames)
    except (wave.Error, FileNotFoundError, OSError) as exc:
        raise AudioIOError(f"could not read wav file: {path}") from exc

    if sample_width != 2:
        raise UnsupportedFormatError(f"expected 16-bit PCM, got {sample_width * 8}-bit")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write a 1-D float array in ``[-1.0, 1.0]`` as a 16-bit PCM WAV file.

    Values are clipped to the valid range before quantisation. Returns the
    path that was written.
    """
    path = Path(path)
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(pcm)
    except OSError as exc:
        raise AudioIOError(f"could not write wav file: {path}") from exc
    return path
