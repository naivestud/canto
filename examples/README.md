# Examples

Small, runnable scripts showing the main flows. They use the untrained `tiny`
configuration, so the audio is noise-like; the point is the plumbing, not the
sound. Train or load real weights for usable speech.

## Synthesize

```bash
python examples/synthesize.py --text "hello from canto" --out hello.wav
```

## Voice cloning

Provide a few seconds of reference audio and the text to speak:

```bash
python examples/voice_clone.py --prompt ref.wav --text "clone me" --out clone.wav
```

## Codec round-trip

Encode a WAV to discrete tokens and decode it back:

```bash
python examples/encode_decode.py --wav input.wav --out reconstruction.wav
```

## Notes

- All scripts accept `--seed` to make runs reproducible.
- Synthesis length is capped with `--max-frames` (one frame is one codec step,
  8 samples at the tiny preset), so short caps keep these demos fast.
- `voice_clone.py` works best with a clean prompt of roughly two to five
  seconds at the model sample rate (24 kHz).
