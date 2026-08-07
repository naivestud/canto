#!/usr/bin/env python
"""Zero-shot voice cloning: speak new text in the voice of a reference clip.

Usage:
    python examples/voice_clone.py --prompt ref.wav --text "clone me" --out clone.wav
"""

from __future__ import annotations

import argparse

from canto import CantoTTS
from canto.audio import read_wav, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="reference wav (a few seconds)")
    parser.add_argument("--text", required=True, help="text to speak in the prompt voice")
    parser.add_argument("--out", default="clone.wav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    prompt, _ = read_wav(args.prompt)
    tts = CantoTTS()
    result = tts.clone_voice(prompt, args.text, seed=args.seed, max_frames=args.max_frames)
    write_wav(args.out, result.wav, result.sample_rate)
    print(f"wrote {args.out}: {result.duration:.2f}s, {result.num_frames} frames")


if __name__ == "__main__":
    main()
