"""Turn waveforms into discrete acoustic tokens and back again."""

from __future__ import annotations

import numpy as np
import torch

from ..codec import AudioCodec


class AudioTokenizer:
    """A numpy-friendly wrapper around :class:`AudioCodec`.

    ``encode`` maps a 1-D float waveform to a ``(num_quantizers, frames)`` int
    array of codes; ``decode`` maps codes back to a waveform. The tokenizer owns
    no state of its own, so it is cheap to construct around a shared codec.
    """

    def __init__(self, codec: AudioCodec, device: str | torch.device | None = None) -> None:
        self.codec = codec
        self.device = torch.device(device) if device is not None else next(
            codec.parameters()
        ).device
        self.codec.to(self.device)
        self.codec.eval()

    @property
    def sample_rate(self) -> int:
        return self.codec.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.codec.sample_rate / self.codec.hop_length

    def encode(self, wav: np.ndarray) -> np.ndarray:
        """Encode a 1-D float waveform into codes of shape ``(Q, frames)``."""
        wav = np.asarray(wav, dtype=np.float32).reshape(1, 1, -1)
        tensor = torch.from_numpy(wav).to(self.device)
        with torch.no_grad():
            indices = self.codec.encode(tensor)
        return indices[0].to(torch.long).cpu().numpy()

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes of shape ``(Q, frames)`` back into a 1-D float waveform."""
        codes = np.asarray(codes)
        if codes.ndim == 1:
            codes = codes[None, :]
        tensor = torch.from_numpy(codes).to(torch.long).to(self.device)[None, ...]
        with torch.no_grad():
            wav = self.codec.decode_indices(tensor)
        return wav[0, 0].cpu().numpy()
