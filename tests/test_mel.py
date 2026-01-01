import numpy as np
import pytest

from canto.audio import log_mel_spectrogram, mel_filterbank
from canto.audio.mel import hz_to_mel, mel_to_hz


def test_filterbank_shape_and_nonnegative():
    bank = mel_filterbank(n_mels=40, n_fft=512, sample_rate=16_000)
    assert bank.shape == (40, 257)
    assert (bank >= 0).all()
    # Every mel band should have some energy.
    assert (bank.sum(axis=1) > 0).all()


def test_log_mel_shape():
    x = np.random.default_rng(0).standard_normal(16_000).astype(np.float32)
    spec = log_mel_spectrogram(x, 16_000, n_fft=512, hop_length=160, n_mels=40)
    assert spec.shape[0] == 40
    assert spec.ndim == 2


def test_log_mel_deterministic():
    x = np.random.default_rng(1).standard_normal(8_000).astype(np.float32)
    a = log_mel_spectrogram(x, 16_000, n_mels=40)
    b = log_mel_spectrogram(x, 16_000, n_mels=40)
    np.testing.assert_array_equal(a, b)


def test_silence_hits_mel_floor():
    x = np.zeros(4_000, dtype=np.float32)
    spec = log_mel_spectrogram(x, 16_000, n_mels=40)
    # log(floor) is a large negative constant; silence should sit at it.
    assert np.allclose(spec, np.log(1e-10), atol=1e-4)


def test_tiny_signal_stays_finite():
    # Very small energies must be clamped to the floor, never log(0) = -inf.
    x = np.full(4_000, 1e-12, dtype=np.float32)
    spec = log_mel_spectrogram(x, 16_000, n_mels=40)
    assert np.isfinite(spec).all()


def test_mel_hz_roundtrip():
    for hz in (0.0, 100.0, 1_000.0, 8_000.0):
        assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, rel=1e-6)


def test_filterbank_is_cached():
    # The same parameter set must reuse the cached array.
    a = mel_filterbank(n_mels=40, n_fft=512, sample_rate=16_000)
    b = mel_filterbank(n_mels=40, n_fft=512, sample_rate=16_000)
    assert a is b


def test_filterbank_fmax_limits_bands():
    limited = mel_filterbank(n_mels=40, n_fft=512, sample_rate=16_000, fmax=2_000.0)
    # With fmax at 2 kHz nothing above bin 64 (2 kHz at 512/16k) has energy.
    assert limited[:, 64:].sum() == 0.0
    assert limited.sum() > 0.0
