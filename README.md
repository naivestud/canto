# canto

Zero-shot text-to-speech over a neural audio codec.

The plan is a small but complete pipeline:

- a neural audio codec that turns waveforms into discrete acoustic tokens,
- an autoregressive model that generates those tokens from text plus a voice prompt,
- a vocoder that renders tokens back into audio,
- a CLI for synthesis and voice cloning.

Work in progress.
