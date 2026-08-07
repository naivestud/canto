"""The full neural audio codec model."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import CodecConfig
from ..constants import SAMPLE_RATE
from .decoder import Decoder
from .encoder import Encoder
from .quantize import ResidualVectorQuantizer


class AudioCodec(nn.Module):
    """Waveform in, discrete codes out, waveform back again.

    The encoder compresses audio into a latent sequence, the residual vector
    quantiser turns it into discrete codes, and the decoder reconstructs audio
    from the quantised latents.
    """

    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.sample_rate = SAMPLE_RATE
        self.hop_length = config.hop_length

        self.encoder = Encoder(in_channels=1, dim=config.dim, strides=config.enc_strides)
        self.quantizer = ResidualVectorQuantizer(
            config.dim, config.codebook_size, config.codebook_dim, config.num_quantizers
        )
        self.decoder = Decoder(out_channels=1, dim=config.dim, strides=config.dec_strides)

    def encode(self, wav: torch.Tensor, num_quantizers: int | None = None) -> torch.Tensor:
        """Encode ``(B, 1, T)`` audio into codes ``(B, Q, T')``.

        ``num_quantizers`` selects how many codebook levels to keep, trading
        bandwidth for fidelity; it defaults to every level the codec was built
        with.
        """
        latents = self.encoder(wav)
        return self.quantizer.encode(latents, num_quantizers)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codes ``(B, Q, T')`` back into ``(B, 1, T)`` audio."""
        quantized = self.quantizer.decode_indices(indices)
        wav = self.decoder(quantized)
        # The decoder already ends in tanh; clamp defensively so downstream code
        # can rely on the [-1, 1] contract even if the head is swapped out.
        return wav.clamp(-1.0, 1.0)

    def bandwidth_kbps(self, num_quantizers: int | None = None) -> float:
        """Bit-rate in kbit/s for a given number of codebook levels."""
        n = self.config.num_quantizers if num_quantizers is None else num_quantizers
        frame_rate = self.sample_rate / self.hop_length
        bits_per_frame = n * math.log2(self.config.codebook_size)
        return frame_rate * bits_per_frame / 1000.0

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, indices, loss)`` for a batch of audio."""
        latents = self.encoder(wav)
        quantized, indices, loss = self.quantizer(latents)
        reconstruction = self.decoder(quantized)
        return reconstruction, indices, loss
