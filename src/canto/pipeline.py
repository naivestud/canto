"""End-to-end synthesis pipeline tying the codec, language model, and vocoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .codec import AudioCodec
from .config import CantoConfig
from .lm.model import AcousticTokenLM
from .text import TextFrontend
from .tokens import AudioTokenizer
from .vocoder import Vocoder


@dataclass
class SynthesisResult:
    """The output of a synthesis call."""

    wav: np.ndarray
    sample_rate: int


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

        text_ids = self.frontend.to_ids(text)
        text_tensor = torch.tensor([text_ids], dtype=torch.long)

        prompt_codes = None
        if prompt_wav is not None:
            codes = self.tokenizer.encode(prompt_wav)[0]
            prompt_codes = torch.tensor(codes, dtype=torch.long)[None, :]

        with torch.no_grad():
            codes = self.lm.generate(text_tensor, prompt_codes, max_new_tokens=max_frames)
            if codes.shape[1] == 0:
                # Fall back to a single silent frame so callers always get audio.
                codes = torch.zeros(1, 1, dtype=torch.long)
            wav = self.vocoder(codes)

        return SynthesisResult(wav=wav[0, 0].cpu().numpy(), sample_rate=self.sample_rate)
