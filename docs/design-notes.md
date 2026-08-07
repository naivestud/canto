# Design notes

A few decisions and the reasoning behind them.

## Why tokens instead of mel spectrograms?

Modelling speech as discrete tokens lets a single decoder-only language model
handle the acoustic generation, with no duration model, no attention alignment,
and no separate acoustic feature regression. Tokens also make the voice prompt
trivial to inject: it is just a prefix of the same token sequence.

## Why a grapheme vocabulary instead of phonemes?

A phonemiser (G2P) is a heavy, language-specific dependency. For a small,
dependency-light library, graphemes are deterministic, cover any input, and are
good enough to build and exercise the pipeline. The `TextFrontend` is the
seam where a real G2P would slot in; a tiny digit-to-word fallback shows the
shape of that extension.

## Why does the language model only generate the first codebook level?

The autoregressive stage models the coarse acoustic tokens (codebook level 0).
That keeps the generation problem one-dimensional and is the standard coarse
stage in coarse-to-fine codec TTS systems. The codec itself implements full
residual vector quantisation and is tested with several levels; a refinement
model over the remaining levels would be a natural next step.

## Why a separate vocoder if the codec already decodes?

The codec decoder reconstructs audio from quantised latents, which is what you
want for a round-trip. At synthesis time only the coarse tokens exist, so a
vocoder that maps tokens directly to a waveform is the cleaner path and avoids
fabricating the missing codebook levels.

## Why tiny configs everywhere in tests?

The models are real PyTorch modules, but tests run on CPU in seconds, so the
default preset shrinks channels, layers, and codebooks. Shape, determinism, and
round-trip properties are meaningful regardless of width; audio quality is not
what the test-suite asserts.

## Known limitations

- No training loop ships yet; the checkpoint format is ready for one.
- `generate` stops the whole batch when any sequence emits the end marker, so
  batched synthesis is conservative. Single-sequence synthesis (the common
  case) is unaffected.
- No streaming; synthesis is offline and returns the full waveform.
