# Architecture

canto is a zero-shot text-to-speech system built around a neural audio codec.
Instead of predicting mel spectrograms, it models speech as a sequence of
discrete acoustic tokens, which lets a plain decoder-only language model do the
heavy lifting.

## The three stages

```
text ──▶ [ AcousticTokenLM ] ──▶ acoustic tokens
prompt ─┘                              │
                                       ▼
                                [ Vocoder ] ──▶ waveform

waveform ──▶ [ AudioCodec encoder ] ──▶ latents ──▶ [ RVQ ] ──▶ tokens
tokens    ──▶ [ RVQ lookup ] ──▶ latents ──▶ [ AudioCodec decoder ] ──▶ waveform
```

### 1. Neural audio codec (`canto.codec`)

The codec compresses a waveform into a compact latent sequence and back.

- `Encoder` — strided 1-D convolutions down-sample the waveform by
  `prod(enc_strides)` (320× by default → 75 frames/second at 24 kHz).
- `ResidualVectorQuantizer` — a stack of `VectorQuantizer` codebooks. Each
  level quantises the residual left by the previous levels, so more levels
  trade bandwidth for fidelity.
- `Decoder` — transposed convolutions up-sample the quantised latents back to
  a waveform, ending in `tanh`.

`AudioTokenizer` wraps the codec with a numpy-friendly `encode` / `decode`.

### 2. Acoustic-token language model (`canto.lm`)

A decoder-only transformer over a joint vocabulary of text graphemes and
acoustic tokens. The sequence is laid out as:

```
[text ...] [audio_bos] [prompt tokens ...] [generated tokens ...] [audio_eos]
```

The model is trained with teacher forcing to predict the next token. At
inference it is primed with the text and the prompt tokens, then sampled
autoregressively until it emits `audio_eos`. Because the prompt tokens carry
the reference speaker's acoustic content, generation continues in that voice —
this is what makes cloning zero-shot.

Generation is restricted to the acoustic-token range, and decoding supports
greedy, top-k, top-p (nucleus), temperature, and a repetition penalty.

### 3. Vocoder (`canto.vocoder`)

A HiFi-GAN-style generator that embeds the discrete tokens and up-samples them
with transposed convolutions. The product of its `upsample_rates` equals the
codec hop length, so a `T`-frame token sequence becomes `T * hop` samples. A
multi-period refinement stage smooths the up-sampled signal before the final
projection.

## The pipeline (`canto.pipeline`)

`CantoTTS` ties the three stages together and owns the text front-end
(normalisation + grapheme encoding). `synthesize` runs text → tokens → waveform;
`clone_voice` adds a prompt; `synthesize_chunked` handles long inputs.

## Data flow and shapes

| Tensor | Shape | Notes |
| --- | --- | --- |
| waveform | `(B, 1, T)` | float in `[-1, 1]` |
| codec latents | `(B, dim, T / hop)` | encoder output |
| acoustic codes | `(B, Q, T / hop)` | one row per codebook level |
| LM token ids | `(B, L)` | joint text + audio vocabulary |
| vocoder input | `(B, T / hop)` | first-codebook codes |
