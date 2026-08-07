"""Command-line interface for canto."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="canto",
        description="Zero-shot text-to-speech over a neural audio codec.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="print the library version")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
