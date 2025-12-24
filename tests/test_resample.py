import numpy as np
import pytest

from canto.audio.resample import resample


def test_same_rate_returns_equal_copy():
    x = np.random.default_rng(0).standard_normal(100).astype(np.float32)
    out = resample(x, 16_000, 16_000)
    np.testing.assert_array_equal(out, x)
    assert out is not x


def test_upsample_length_scales():
    x = np.zeros(1_000, dtype=np.float32)
    out = resample(x, 8_000, 16_000)
    # Doubling the rate doubles the duration, within one sample of rounding.
    assert abs(len(out) - 1_999) <= 1


def test_downsample_length_scales():
    x = np.zeros(1_600, dtype=np.float32)
    out = resample(x, 16_000, 8_000)
    assert abs(len(out) - 800) <= 1


def test_dc_signal_stays_constant():
    x = np.full(500, 0.25, dtype=np.float32)
    out = resample(x, 8_000, 24_000)
    np.testing.assert_allclose(out, 0.25, atol=1e-6)


def test_sine_frequency_is_preserved():
    # A 100 Hz tone at 1 kHz should still look like 100 Hz at 2 kHz.
    rate_in, rate_out, freq = 1_000, 2_000, 100.0
    t_in = np.arange(rate_in, dtype=np.float32) / rate_in
    x = np.sin(2 * np.pi * freq * t_in).astype(np.float32)
    out = resample(x, rate_in, rate_out)
    t_out = np.arange(len(out), dtype=np.float32) / rate_out
    expected = np.sin(2 * np.pi * freq * t_out)
    # Ignore the endpoints where interpolation has less support.
    np.testing.assert_allclose(out[10:-10], expected[10:-10], atol=0.05)


def test_empty_input_returns_empty():
    out = resample(np.zeros(0, dtype=np.float32), 8_000, 16_000)
    assert out.shape == (0,)


def test_single_sample_survives():
    out = resample(np.array([0.5], dtype=np.float32), 8_000, 16_000)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(0.5)


def test_invalid_rates_raise():
    x = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        resample(x, 0, 16_000)
    with pytest.raises(ValueError):
        resample(x, 16_000, -1)
