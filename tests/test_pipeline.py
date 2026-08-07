from pathlib import Path

import numpy as np
import pytest

from canto.audio import read_wav
from canto.errors import EmptyTextError
from canto.pipeline import CantoTTS

FIXTURES = Path(__file__).parent / "fixtures"


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


def test_clone_voice_with_prompt():
    prompt, rate = read_wav(FIXTURES / "tone_1s.wav")
    tts = CantoTTS()
    result = tts.clone_voice(prompt, "clone this voice", seed=3, max_frames=10)
    assert result.sample_rate == tts.sample_rate
    assert result.wav.ndim == 1
    assert result.wav.shape[0] > 0
    assert np.abs(result.wav).max() <= 1.0


def test_clone_voice_deterministic():
    prompt, _ = read_wav(FIXTURES / "tone_1s.wav")
    tts = CantoTTS()
    a = tts.clone_voice(prompt, "hi", seed=11, max_frames=8)
    b = tts.clone_voice(prompt, "hi", seed=11, max_frames=8)
    np.testing.assert_array_equal(a.wav, b.wav)


def test_empty_text_raises():
    tts = CantoTTS()
    with pytest.raises(EmptyTextError):
        tts.synthesize("   ")
    with pytest.raises(EmptyTextError):
        tts.synthesize("")
