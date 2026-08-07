"""Command-line interface for canto."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .audio import write_wav
from .errors import CantoError


def _add_synthesis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", required=True, help="text to synthesise")
    parser.add_argument("--out", required=True, help="output wav path")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--max-frames", type=int, default=400, help="cap on generated frames")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="expected output sample rate (must match the model rate)",
    )


def _check_sample_rate(requested: int | None, model_rate: int) -> int | None:
    """Return an error code if ``requested`` disagrees with the model rate."""
    if requested is not None and requested != model_rate:
        print(f"canto: model runs at {model_rate} Hz, got --sample-rate {requested}", file=sys.stderr)
        return 2
    return None


def cmd_synthesize(args: argparse.Namespace) -> int:
    from .pipeline import CantoTTS

    tts = CantoTTS()
    error = _check_sample_rate(args.sample_rate, tts.sample_rate)
    if error is not None:
        return error
    result = tts.synthesize(args.text, seed=args.seed, max_frames=args.max_frames)
    write_wav(args.out, result.wav, result.sample_rate)
    print(f"wrote {args.out} ({result.duration:.2f}s, {result.num_frames} frames)")
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    from .audio import read_wav
    from .pipeline import CantoTTS

    prompt, sample_rate = read_wav(args.prompt)
    tts = CantoTTS()
    error = _check_sample_rate(args.sample_rate, tts.sample_rate)
    if error is not None:
        return error
    if sample_rate != tts.sample_rate:
        print(f"warning: prompt rate {sample_rate} != model rate {tts.sample_rate}", file=sys.stderr)
    result = tts.clone_voice(prompt, args.text, seed=args.seed, max_frames=args.max_frames)
    write_wav(args.out, result.wav, result.sample_rate)
    print(f"wrote {args.out} ({result.duration:.2f}s, {result.num_frames} frames)")
    return 0


def _build_tokenizer():
    from .codec import AudioCodec
    from .config import CantoConfig
    from .tokens import AudioTokenizer

    config = CantoConfig.tiny()
    return AudioTokenizer(AudioCodec(config.codec))


def cmd_encode(args: argparse.Namespace) -> int:
    import numpy as np

    from .audio import read_wav

    wav, _ = read_wav(args.wav)
    codes = _build_tokenizer().encode(wav)
    np.save(args.out, codes)
    print(f"wrote {args.out} with codes of shape {codes.shape}")
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    import numpy as np

    codes = np.load(args.tokens)
    tokenizer = _build_tokenizer()
    wav = tokenizer.decode(codes)
    write_wav(args.out, wav, tokenizer.sample_rate)
    print(f"wrote {args.out} ({len(wav)} samples)")
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

    clone = sub.add_parser("clone", help="synthesise speech in the voice of a prompt")
    clone.add_argument("--prompt", required=True, help="reference wav to clone")
    _add_synthesis_options(clone)

    encode = sub.add_parser("encode", help="tokenise a wav into discrete codes")
    encode.add_argument("--wav", required=True, help="input wav path")
    encode.add_argument("--out", required=True, help="output .npy path for codes")

    decode = sub.add_parser("decode", help="render codes back into a wav")
    decode.add_argument("--tokens", required=True, help="input .npy codes path")
    decode.add_argument("--out", required=True, help="output wav path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    handlers = {
        "synthesize": cmd_synthesize,
        "clone": cmd_clone,
        "encode": cmd_encode,
        "decode": cmd_decode,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return handler(args)
    except CantoError as exc:
        print(f"canto: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
