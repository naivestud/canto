"""Command-line interface for canto."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .audio import write_wav


def _add_synthesis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", required=True, help="text to synthesise")
    parser.add_argument("--out", required=True, help="output wav path")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--max-frames", type=int, default=400, help="cap on generated frames")


def cmd_synthesize(args: argparse.Namespace) -> int:
    from .pipeline import CantoTTS

    tts = CantoTTS()
    result = tts.synthesize(args.text, seed=args.seed, max_frames=args.max_frames)
    write_wav(args.out, result.wav, result.sample_rate)
    print(f"wrote {args.out} ({result.duration:.2f}s, {result.num_frames} frames)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="canto",
        description="Zero-shot text-to-speech over a neural audio codec.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="print the library version")

    synth = sub.add_parser("synthesize", help="synthesise speech from text")
    _add_synthesis_options(synth)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "synthesize":
        return cmd_synthesize(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
