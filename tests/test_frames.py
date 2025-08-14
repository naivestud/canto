import numpy as np

from canto.audio.frames import frame_signal, hann_window, num_frames


def test_hann_window_endpoints_near_zero():
    w = hann_window(256)
    assert w.shape == (256,)
    assert w[0] < 1e-3
    assert w.max() > 0.99


def test_frame_shape_center():
    x = np.zeros(1000, dtype=np.float32)
    frames = frame_signal(x, frame_length=400, hop_length=160, center=True)
    expected = num_frames(1000, 400, 160, center=True)
    assert frames.shape == (expected, 400)


def test_frame_no_center():
    x = np.arange(640, dtype=np.float32)
    frames = frame_signal(x, frame_length=320, hop_length=160, center=False)
    # 1 + (640 - 320) // 160 == 3 frames.
    assert frames.shape == (3, 320)


def test_short_signal_returns_empty_when_uncentered():
    x = np.zeros(100, dtype=np.float32)
    frames = frame_signal(x, frame_length=400, hop_length=160, center=False)
    assert frames.shape == (0, 400)


def test_frames_reference_original_samples():
    x = np.arange(800, dtype=np.float32)
    frames = frame_signal(x, frame_length=200, hop_length=100, center=False)
    # Each frame start aligns with a hop multiple of the original signal.
    np.testing.assert_allclose(frames[1, 0], 100.0)
    np.testing.assert_allclose(frames[2, 0], 200.0)
