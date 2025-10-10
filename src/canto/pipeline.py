"""End-to-end synthesis pipeline tying the codec, language model, and vocoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .codec import AudioCodec
from .config import CantoConfig
from .errors import EmptyTextError
from .lm.model import AcousticTokenLM
from .text import TextFrontend
from .tokens import AudioTokenizer
from .vocoder import Vocoder


@dataclass
class SynthesisResult:
    """The output of a synthesis call."""

    wav: np.ndarray
    sample_rate: int
    codes: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.codes.shape[-1]) if self.codes.size else 0

    @property
    def duration(self) -> float:
        return float(self.wav.shape[-1]) / self.sample_rate


class CantoTTS:
    """Zero-shot text-to-speech built on a neural audio codec.

    The codec turns audio into discrete tokens, the language model generates
    those tokens from text plus an optional voice prompt, and the vocoder
    renders the tokens back into a waveform.
    """

    def __init__(self, config: CantoConfig | None = None) -> None:
        self.config = config or CantoConfig.tiny()
        self.config.validate()
        self.sample_rate = self.config.sample_rate

        self.codec = AudioCodec(self.config.codec)
        self.tokenizer = AudioTokenizer(self.codec)
        self.lm = AcousticTokenLM(self.config.lm)
        self.vocoder = Vocoder.from_config(
            self.config.vocoder, num_tokens=self.config.codec.codebook_size
        )
        self.frontend = TextFrontend()
        self.eval()

    def eval(self) -> None:
        """Put every component in inference mode."""
        self.codec.eval()
        self.lm.eval()
        self.vocoder.eval()

    def encode_audio(self, wav: np.ndarray) -> np.ndarray:
        """Tokenise a waveform into codes of shape ``(Q, frames)``."""
        return self.tokenizer.encode(wav)

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        """Render codes back into a waveform with the codec decoder."""
        return self.tokenizer.decode(codes)

    def _prompt_codes(self, prompt_wav: np.ndarray) -> torch.Tensor:
        """Tokenise a reference waveform and keep the first codebook level.

        The first level carries the coarse acoustic content that steers the
        language model towards the prompt's voice.
        """
        codes = self.tokenizer.encode(prompt_wav)[0]
        return torch.tensor(codes, dtype=torch.long)[None, :]

    def synthesize(
        self,
        text: str,
        prompt_wav: np.ndarray | None = None,
        max_frames: int = 400,
        seed: int | None = None,
    ) -> SynthesisResult:
        """Synthesise speech for ``text``, optionally cloning ``prompt_wav``.

        ``prompt_wav`` is a short reference waveform whose first-codebook codes
        prime the language model so the output follows its voice.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not self.frontend.normalize(text):
            raise EmptyTextError("cannot synthesise empty text")

        text_ids = self.frontend.to_ids(text)
        text_tensor = torch.tensor([text_ids], dtype=torch.long)

        prompt_codes = None
        if prompt_wav is not None:
            prompt_codes = self._prompt_codes(prompt_wav)

        with torch.no_grad():
            codes = self.lm.generate(text_tensor, prompt_codes, max_new_tokens=max_frames)
            if codes.shape[1] == 0:
                # Fall back to a single silent frame so callers always get audio.
                codes = torch.zeros(1, 1, dtype=torch.long)
            wav = self.vocoder(codes)

        return SynthesisResult(
            wav=wav[0, 0].cpu().numpy(),
            sample_rate=self.sample_rate,
            codes=codes[0].cpu().numpy(),
        )

    def clone_voice(
        self,
        prompt_wav: np.ndarray,
        text: str,
        max_frames: int = 400,
        seed: int | None = None,
    ) -> SynthesisResult:
        """Synthesise ``text`` in the voice of ``prompt_wav`` (zero-shot)."""
        return self.synthesize(text, prompt_wav=prompt_wav, max_frames=max_frames, seed=seed)

    def synthesize_chunked(
        self,
        text: str,
        prompt_wav: np.ndarray | None = None,
        max_chars: int = 80,
        max_frames: int = 400,
        seed: int | None = None,
    ) -> SynthesisResult:
        """Synthesise long text by splitting it into chunks and concatenating.

        Splitting keeps each language-model call within a comfortable context
        length and bounds memory for very long inputs.
        """
        chunks = _chunk_text(text, max_chars)
        results = [
            self.synthesize(chunk, prompt_wav=prompt_wav, max_frames=max_frames, seed=seed)
            for chunk in chunks
        ]
        if results:
            wav = np.concatenate([r.wav for r in results])
            codes = np.concatenate([r.codes for r in results], axis=-1)
        else:
            wav = np.zeros(0, dtype=np.float32)
            codes = np.zeros(0, dtype=np.int64)
        return SynthesisResult(wav=wav, sample_rate=self.sample_rate, codes=codes)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` on word boundaries into chunks of at most ``max_chars``."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > max_chars:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + (1 if length else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks
