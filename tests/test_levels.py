import numpy as np
import pytest

from canto.audio.levels import normalize_peak, peak, rms, trim_silence


def test_rms_of_known_signal():
    # A full-scale square wave has RMS 1.0; silence has RMS 0.0.
    assert rms(np.ones(100, dtype=np.float32)) == 1.0
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0


def test_rms_of_sine_is_half_sqrt_two():
    t = np.arange(100_000, dtype=np.float32) / 100_000.0
    x = np.sin(2 * np.pi * t).astype(np.float32)
    assert abs(rms(x) - np.sqrt(0.5)) < 1e-3


def test_rms_empty_is_zero():
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_peak_finds_largest_absolute_sample():
    x = np.array([0.1, -0.9, 0.4], dtype=np.float32)
    assert peak(x) == pytest.approx(0.9)
    assert peak(np.zeros(0, dtype=np.float32)) == 0.0


def test_normalize_peak_scales_to_target():
    x = np.array([0.0, -0.5, 0.25], dtype=np.float32)
    out = normalize_peak(x, target=1.0)
    assert peak(out) == 1.0
    np.testing.assert_allclose(out, x * 2.0)


def test_normalize_peak_custom_target():
    x = np.array([0.0, 0.4], dtype=np.float32)
    out = normalize_peak(x, target=0.5)
    assert abs(peak(out) - 0.5) < 1e-6


def test_normalize_peak_silence_unchanged():
    x = np.zeros(8, dtype=np.float32)
    out = normalize_peak(x)
    np.testing.assert_array_equal(out, x)
    assert out is not x


def test_normalize_peak_does_not_mutate_input():
    x = np.array([0.5, 0.25], dtype=np.float32)
    normalize_peak(x)
    np.testing.assert_array_equal(x, np.array([0.5, 0.25], dtype=np.float32))


def test_trim_silence_removes_quiet_edges():
    x = np.array([0.0, 0.0, 0.5, -0.6, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(trim_silence(x), np.array([0.5, -0.6], dtype=np.float32))


def test_trim_silence_keeps_interior_quiet_samples():
    x = np.array([0.9, 0.0, 0.9], dtype=np.float32)
    assert trim_silence(x).shape == (3,)


def test_trim_silence_all_quiet_returns_empty():
    x = np.full(10, 1e-6, dtype=np.float32)
    assert trim_silence(x).shape == (0,)


def test_trim_silence_respects_threshold():
    x = np.array([0.001, 0.5, 0.001], dtype=np.float32)
    # With a low threshold the 0.001 samples count as loud.
    assert trim_silence(x, threshold=1e-4).shape == (3,)
    assert trim_silence(x, threshold=1e-2).shape == (1,)


def test_trim_silence_negative_threshold_raises():
    with pytest.raises(ValueError):
        trim_silence(np.zeros(4, dtype=np.float32), threshold=-1.0)
