#!/usr/bin/env python
"""Round-trip a WAV through the codec: encode to tokens, then decode back.

Usage:
    python examples/encode_decode.py --wav input.wav --out reconstruction.wav
"""

from __future__ import annotations

import argparse

from canto import AudioCodec, AudioTokenizer, CantoConfig
from canto.audio import read_wav, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, help="input wav to round-trip")
    parser.add_argument("--out", default="reconstruction.wav")
    args = parser.parse_args()

    config = CantoConfig.tiny()
    tokenizer = AudioTokenizer(AudioCodec(config.codec))

    wav, _ = read_wav(args.wav)
    codes = tokenizer.encode(wav)
    recon = tokenizer.decode(codes)
    write_wav(args.out, recon, tokenizer.sample_rate)
    print(f"encoded {len(wav)} samples -> {codes.shape} codes -> {len(recon)} samples")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
