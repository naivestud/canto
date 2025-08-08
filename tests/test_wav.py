import numpy as np
import pytest

from canto.audio import read_wav, write_wav
from canto.errors import AudioIOError, UnsupportedFormatError


def test_roundtrip_preserves_signal(tmp_path):
    rate = 16_000
    t = np.arange(rate, dtype=np.float32) / rate
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    path = write_wav(tmp_path / "tone.wav", tone, rate)
    loaded, loaded_rate = read_wav(path)

    assert loaded_rate == rate
    assert loaded.shape == tone.shape
    # 16-bit quantisation keeps error well below 1e-3.
    assert np.max(np.abs(loaded - tone)) < 1e-3


def test_write_clips_out_of_range(tmp_path):
    samples = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    path = write_wav(tmp_path / "clip.wav", samples, 8_000)
    loaded, _ = read_wav(path)
    assert loaded.max() <= 1.0
    assert loaded.min() >= -1.0


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(AudioIOError):
        read_wav(tmp_path / "does_not_exist.wav")


def test_read_non_pcm_raises(tmp_path):
    path = tmp_path / "notawav.wav"
    path.write_bytes(b"this is not a riff file")
    with pytest.raises((AudioIOError, UnsupportedFormatError)):
        read_wav(path)
