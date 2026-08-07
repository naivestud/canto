#!/usr/bin/env python
"""Synthesise a sentence and write it to a WAV file.

Usage:
    python examples/synthesize.py --text "hello world" --out hello.wav
"""

from __future__ import annotations

import argparse

from canto import CantoTTS
from canto.audio import write_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="hello from canto")
    parser.add_argument("--out", default="synthesis.wav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    tts = CantoTTS()
    result = tts.synthesize(args.text, seed=args.seed, max_frames=args.max_frames)
    write_wav(args.out, result.wav, result.sample_rate)
    print(f"wrote {args.out}: {result.duration:.2f}s, {result.num_frames} frames")


if __name__ == "__main__":
    main()
