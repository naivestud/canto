# API reference

Public surface lives in the top-level `canto` namespace.

## `CantoTTS`

```python
CantoTTS(config: CantoConfig | None = None)
```

End-to-end pipeline. Defaults to `CantoConfig.tiny()`.

- `synthesize(text, prompt_wav=None, max_frames=400, seed=None) -> SynthesisResult`
- `clone_voice(prompt_wav, text, max_frames=400, seed=None) -> SynthesisResult`
- `synthesize_chunked(text, prompt_wav=None, max_chars=80, max_frames=400, seed=None) -> SynthesisResult`
- `encode_audio(wav) -> np.ndarray` — waveform to codes `(Q, frames)`
- `decode_codes(codes) -> np.ndarray` — codes back to a waveform
- `eval()` — set every component to inference mode

## `SynthesisResult`

Dataclass with fields `wav`, `sample_rate`, `codes` and properties
`num_frames` and `duration`.

## `AudioCodec`

```python
AudioCodec(config: CodecConfig)
```

- `encode(wav, num_quantizers=None) -> Tensor` — `(B,1,T)` to `(B,Q,T')`
- `decode_indices(indices) -> Tensor` — `(B,Q,T')` to `(B,1,T)`
- `forward(wav) -> (recon, indices, loss)`
- `bandwidth_kbps(num_quantizers=None) -> float`

## `AudioTokenizer`

```python
AudioTokenizer(codec: AudioCodec, device=None)
```

- `encode(wav) -> np.ndarray` — 1-D waveform to `(Q, frames)` int codes
- `decode(codes) -> np.ndarray` — codes back to a 1-D waveform
- properties `sample_rate`, `frame_rate`

## `AcousticTokenLM`

```python
AcousticTokenLM(config: LMConfig, text_vocab: GraphemeVocab | None = None)
```

- `forward(text_ids, prompt_codes=None, target_codes=None) -> logits`
- `generate(text_ids, prompt_codes=None, max_new_tokens=200, sampler=None) -> codes`
- `pack_inputs(text_ids, prompt_codes, target_codes) -> sequence`
- `audio_to_lm(codes)` / `lm_to_audio(ids)` — shift between codec and LM spaces

## `Vocoder`

```python
Vocoder(num_tokens, embed_dim=32, channels=64, upsample_rates=(4,2), ...)
Vocoder.from_config(config, num_tokens)
```

- `forward(tokens) -> Tensor` — `(B,T)` to `(B,1,T*hop)`

## Configs

`CantoConfig`, `CodecConfig`, `LMConfig`, `VocoderConfig` are keyword-only
dataclasses. `CantoConfig.tiny()` returns a minimal preset; `validate()` checks
cross-component consistency.

## Errors

All exceptions derive from `CantoError`: `AudioIOError`, `ConfigError`,
`CheckpointError`, `EmptyTextError`, `UnsupportedFormatError`.

## Checkpoints

- `save_checkpoint(directory, config, models)`
- `load_checkpoint(directory) -> (config, state_dicts)`
