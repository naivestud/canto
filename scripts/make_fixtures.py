#!/usr/bin/env python
"""Generate the small WAV fixtures used by the test-suite.

Run from the repository root:

    python scripts/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from canto.audio import write_wav

SAMPLE_RATE = 16_000
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def tone(duration: float, frequency: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * frequency * t)


def silence(duration: float) -> np.ndarray:
    return np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)


def sweep(duration: float, f0: float = 220.0, f1: float = 880.0) -> np.ndarray:
    t = np.arange(int(duration * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * duration))
    return 0.5 * np.sin(phase)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    write_wav(FIXTURE_DIR / "tone_1s.wav", tone(1.0), SAMPLE_RATE)
    write_wav(FIXTURE_DIR / "silence_1s.wav", silence(1.0), SAMPLE_RATE)
    write_wav(FIXTURE_DIR / "sweep_2s.wav", sweep(2.0), SAMPLE_RATE)
    print(f"wrote fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
