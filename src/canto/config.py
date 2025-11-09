"""Dataclass configuration for the codec, language model, and full pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import DEFAULT_CODEBOOK_SIZE, DEFAULT_NUM_QUANTIZERS, SAMPLE_RATE
from .errors import ConfigError


@dataclass(kw_only=True)
class CodecConfig:
    """Geometry of the neural audio codec."""

    dim: int = 64
    enc_strides: tuple[int, ...] = (2, 4, 5, 8)
    dec_strides: tuple[int, ...] = (8, 5, 4, 2)
    num_quantizers: int = DEFAULT_NUM_QUANTIZERS
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    codebook_dim: int = 32

    @property
    def hop_length(self) -> int:
        """Total down-sampling factor, i.e. samples per codec frame."""
        return math.prod(self.enc_strides)

    @property
    def frame_rate(self) -> float:
        return SAMPLE_RATE / self.hop_length

    def validate(self) -> None:
        if self.dim <= 0:
            raise ConfigError(f"codec dim must be positive, got {self.dim}")
        if math.prod(self.enc_strides) != math.prod(self.dec_strides):
            raise ConfigError("encoder and decoder stride products must match")
        if self.num_quantizers < 1:
            raise ConfigError("num_quantizers must be >= 1")
        if self.codebook_size < 2:
            raise ConfigError("codebook_size must be >= 2")


@dataclass(kw_only=True)
class LMConfig:
    """Geometry of the acoustic-token language model."""

    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    context: int = 512
    dropout: float = 0.0
    codebook_size: int = DEFAULT_CODEBOOK_SIZE

    def validate(self) -> None:
        if self.dim <= 0 or self.dim % self.num_heads != 0:
            raise ConfigError("lm dim must be positive and divisible by num_heads")
        if self.num_layers < 1:
            raise ConfigError("num_layers must be >= 1")
        if self.context < 2:
            raise ConfigError("context must be >= 2")


@dataclass(kw_only=True)
class VocoderConfig:
    """Geometry of the token-to-waveform vocoder."""

    embed_dim: int = 32
    channels: int = 64
    upsample_rates: tuple[int, ...] = (8, 5, 4, 2)
    refinements: tuple[int, ...] = (1, 3, 5)

    @property
    def upsample_factor(self) -> int:
        return math.prod(self.upsample_rates)

    def validate(self) -> None:
        if self.channels <= 0 or self.embed_dim <= 0:
            raise ConfigError("vocoder channels and embed_dim must be positive")
        if self.upsample_factor < 1:
            raise ConfigError("vocoder upsample_factor must be >= 1")


@dataclass(kw_only=True)
class CantoConfig:
    """Top-level configuration bundling every component."""

    sample_rate: int = SAMPLE_RATE
    codec: CodecConfig = field(default_factory=CodecConfig)
    lm: LMConfig = field(default_factory=LMConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)

    def validate(self) -> None:
        self.codec.validate()
        self.lm.validate()
        self.vocoder.validate()
        if self.lm.codebook_size != self.codec.codebook_size:
            raise ConfigError("lm and codec codebook_size must match")
        if self.vocoder.upsample_factor != self.codec.hop_length:
            raise ConfigError("vocoder upsample_factor must equal the codec hop length")

    @classmethod
    def tiny(cls) -> CantoConfig:
        """A minimal configuration for tests and quick local runs."""
        codec = CodecConfig(
            dim=16,
            enc_strides=(2, 4),
            dec_strides=(4, 2),
            num_quantizers=1,
            codebook_size=64,
            codebook_dim=8,
        )
        lm = LMConfig(dim=32, num_layers=2, num_heads=2, context=128, codebook_size=64)
        vocoder = VocoderConfig(embed_dim=8, channels=16, upsample_rates=(4, 2))
        return cls(codec=codec, lm=lm, vocoder=vocoder)
