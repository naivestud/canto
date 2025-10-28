from pathlib import Path

import numpy as np
import pytest

from canto.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_version_prints(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.count(".") == 2


def test_encode_decode_roundtrip(tmp_path):
    codes_path = tmp_path / "codes.npy"
    wav_path = tmp_path / "recon.wav"
    assert main(["encode", "--wav", str(FIXTURES / "tone_1s.wav"), "--out", str(codes_path)]) == 0
    assert codes_path.exists()
    assert main(["decode", "--tokens", str(codes_path), "--out", str(wav_path)]) == 0
    assert wav_path.exists()


def test_synthesize_writes_wav(tmp_path):
    out = tmp_path / "out.wav"
    rc = main(["synthesize", "--text", "hello", "--out", str(out), "--seed", "0", "--max-frames", "6"])
    assert rc == 0
    assert out.exists()


def test_clone_writes_wav(tmp_path):
    out = tmp_path / "clone.wav"
    rc = main(
        [
            "clone",
            "--prompt",
            str(FIXTURES / "tone_1s.wav"),
            "--text",
            "hi",
            "--out",
            str(out),
            "--seed", "0",
            "--max-frames", "6",
        ]
    )
    assert rc == 0
    assert out.exists()


def test_unknown_command_returns_zero_and_help(capsys):
    # With no subcommand we print help and exit cleanly.
    assert main([]) == 0


def test_missing_input_file_returns_nonzero(tmp_path, capsys):
    rc = main(["encode", "--wav", str(tmp_path / "nope.wav"), "--out", str(tmp_path / "c.npy")])
    assert rc == 2
    assert "canto:" in capsys.readouterr().err


def test_missing_prompt_returns_nonzero(tmp_path):
    rc = main(
        [
            "clone",
            "--prompt",
            str(tmp_path / "missing.wav"),
            "--text",
            "hi",
            "--out",
            str(tmp_path / "o.wav"),
        ]
    )
    assert rc == 2


def test_sample_rate_mismatch_returns_error(tmp_path):
    rc = main(
        [
            "synthesize",
            "--text",
            "hi",
            "--out",
            str(tmp_path / "o.wav"),
            "--sample-rate",
            "16000",
        ]
    )
    assert rc == 2
