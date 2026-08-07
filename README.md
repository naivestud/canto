# canto

[![CI](https://github.com/naivestud/canto/actions/workflows/ci.yml/badge.svg)](https://github.com/naivestud/canto/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Zero-shot text-to-speech over a neural audio codec.

canto models speech as a sequence of discrete acoustic tokens. A neural audio
codec turns audio into those tokens and back, a decoder-only language model
generates them from text plus a short voice prompt, and a neural vocoder
renders them into a waveform. Cloning a voice is just priming the model with a
few seconds of reference audio.

## How it works

```
text ──▶ acoustic-token LM ──▶ tokens ──▶ vocoder ──▶ waveform
prompt ─┘        ▲
                 └── the audio codec turns the prompt into tokens
```

The [architecture doc](docs/architecture.md) walks through each stage.

> **Status:** the architecture, data flow, CLI, and test-suite are complete.
> No pretrained weights ship with the repo yet, so out-of-the-box output is
> noise-like — the plumbing is the point for now. See the
> [design notes](docs/design-notes.md).

## Features

- **Neural audio codec** — convolutional encoder/decoder with residual vector
  quantisation; configurable codebooks and bandwidth.
- **Audio tokenisation** — waveform ⇄ discrete codes via `AudioTokenizer`.
- **Acoustic-token language model** — causal transformer over a joint text +
  audio vocabulary with prompt conditioning for zero-shot cloning.
- **Neural vocoder** — token-to-waveform generator with multi-period
  refinement.
- **Voice cloning** — `clone_voice()` continues in the voice of a reference
  clip.
- **CLI** — `synthesize`, `clone`, `encode`, `decode`.
- Fully typed, tested on CPU, no heavyweight DSP dependencies.

## Install

```bash
pip install -e .
```

For development (tests, lint, types):

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch 2.1+.

## Quickstart

```python
import canto
from canto.audio import write_wav

tts = canto.CantoTTS()
result = tts.synthesize("hello from canto", seed=0)
write_wav("hello.wav", result.wav, result.sample_rate)
```

Clone a voice from a reference clip:

```python
from canto.audio import read_wav

prompt, _ = read_wav("reference.wav")
result = tts.clone_voice(prompt, "speak like me", seed=1)
```

Command line:

```bash
canto synthesize --text "hello world" --out hello.wav --seed 0
canto clone --prompt ref.wav --text "clone me" --out clone.wav
canto encode --wav input.wav --out codes.npy
canto decode --tokens codes.npy --out recon.wav
```

## Documentation

- [Usage guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Design notes](docs/design-notes.md)
- [API reference](docs/api-reference.md)
- [Examples](examples/README.md)

## License

MIT — see [LICENSE](LICENSE).
