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
