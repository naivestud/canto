"""Mel-spectrogram analysis utilities.

These are plain-numpy so they can be used for inspection and for scoring codec
reconstruction quality without pulling in a heavy DSP dependency.
"""

from __future__ import annotations

import numpy as np

from .frames import frame_signal, hann_window

# Floor applied to mel energies before the logarithm to avoid ``log(0)``.
MEL_FLOOR = 1e-10


def hz_to_mel(hz: float) -> float:
    return 2595.0 * float(np.log10(1.0 + hz / 700.0))


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Return a ``(n_mels, n_fft // 2 + 1)`` triangular mel filterbank."""
    fmax = fmax if fmax is not None else sample_rate / 2.0
    n_bins = n_fft // 2 + 1

    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    bank = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, centre, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        if centre > left:
            bank[m, left:centre] = (np.arange(left, centre) - left) / (centre - left)
        if right > centre:
            bank[m, centre:right] = (right - np.arange(centre, right)) / (right - centre)
    return bank


def log_mel_spectrogram(
    x: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Compute a ``(n_mels, num_frames)`` log-mel spectrogram of ``x``."""
    frames = frame_signal(x, n_fft, hop_length, center=True)
    window = hann_window(n_fft)
    spectrum = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=-1))
    bank = mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)
    mel = spectrum.astype(np.float32) @ bank.T
    return np.log(np.maximum(mel, MEL_FLOOR)).T.astype(np.float32)


def spectral_distance(
    a: np.ndarray,
    b: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> float:
    """Mean absolute log-mel distance between two waveforms.

    The shorter signal sets the comparison length so a codec round-trip can be
    scored even if padding shifts the output by a few samples.
    """
    n = min(a.shape[-1], b.shape[-1])
    spec_a = log_mel_spectrogram(
        a[..., :n], sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    spec_b = log_mel_spectrogram(
        b[..., :n], sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    t = min(spec_a.shape[1], spec_b.shape[1])
    return float(np.mean(np.abs(spec_a[:, :t] - spec_b[:, :t])))
