import numpy as np
import pytest

from canto.audio.levels import normalize_peak, peak, rms


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
