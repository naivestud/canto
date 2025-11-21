# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-08

### Added

- Zero-shot voice cloning: `CantoTTS.clone_voice()` and prompt conditioning in
  the language model.
- `synthesize_chunked()` for long-form text.
- CLI `synthesize`, `clone`, `encode`, and `decode` subcommands.
- `SynthesisResult` now carries the generated codes, frame count, and duration.
- Sampling options: temperature, top-p (nucleus), and a repetition penalty.

## [0.1.0] - 2026-08-08

### Added

- Neural audio codec with a convolutional encoder/decoder and residual vector
  quantisation (`canto.codec`).
- `AudioTokenizer` for waveform ⇄ discrete-token conversion.
- Acoustic-token language model with prompt conditioning and greedy / top-k /
  top-p sampling (`canto.lm`).
- Token-to-waveform vocoder with multi-period refinement (`canto.vocoder`).
- `CantoTTS` pipeline with `synthesize` (`canto.pipeline`).
- Checkpoint save/load directories (`canto.checkpoint`).
- CLI `version` subcommand.
- Test-suite covering every module, runnable on CPU.
