import numpy as np

from canto.pipeline import CantoTTS


def test_synthesize_returns_bounded_waveform():
    tts = CantoTTS()
    result = tts.synthesize("hello there", seed=0, max_frames=12)
    assert result.sample_rate == tts.sample_rate
    assert result.wav.ndim == 1
    assert result.wav.shape[0] > 0
    assert np.abs(result.wav).max() <= 1.0


def test_synthesize_deterministic_with_seed():
    tts = CantoTTS()
    a = tts.synthesize("repeat me", seed=7, max_frames=10)
    b = tts.synthesize("repeat me", seed=7, max_frames=10)
    np.testing.assert_array_equal(a.wav, b.wav)


def test_synthesize_length_scales_with_frames():
    tts = CantoTTS()
    short = tts.synthesize("a", seed=1, max_frames=4)
    long = tts.synthesize("a", seed=1, max_frames=8)
    # More generated frames means more samples, up to early EOS.
    assert long.wav.shape[0] >= short.wav.shape[0]
