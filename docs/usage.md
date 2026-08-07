# Usage

Everything below uses the public API in `canto`. The untrained `tiny` preset
produces noise-like audio; the calls are what matter here.

## Synthesize speech

```python
import canto
from canto.audio import write_wav

tts = canto.CantoTTS()
result = tts.synthesize("hello from canto", seed=0)
write_wav("hello.wav", result.wav, result.sample_rate)
```

`result` is a `SynthesisResult` carrying the waveform, the generated codes, the
frame count, and the duration.

## Clone a voice

Provide a few seconds of reference audio; the model continues in that voice.

```python
import canto
from canto.audio import read_wav, write_wav

prompt, _ = read_wav("reference.wav")
tts = canto.CantoTTS()
result = tts.clone_voice(prompt, "speak like me", seed=1)
write_wav("clone.wav", result.wav, result.sample_rate)
```

## Tokenize and reconstruct audio

```python
import canto
from canto.audio import read_wav, write_wav

config = canto.CantoConfig.tiny()
tokenizer = canto.AudioTokenizer(canto.AudioCodec(config.codec))

wav, _ = read_wav("input.wav")
codes = tokenizer.encode(wav)          # (num_quantizers, frames)
recon = tokenizer.decode(codes)        # waveform
write_wav("recon.wav", recon, tokenizer.sample_rate)
```

## Long text

`synthesize_chunked` splits long input on word boundaries and concatenates the
pieces, keeping every language-model call within a comfortable context length.

```python
tts = canto.CantoTTS()
result = tts.synthesize_chunked(long_text, max_chars=80)
```
