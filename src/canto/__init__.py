"""canto: zero-shot text-to-speech over a neural audio codec.

The public surface is intentionally small:

* :class:`CantoTTS` — end-to-end synthesis and voice cloning.
* :class:`AudioCodec` / :class:`AudioTokenizer` — waveform <-> discrete tokens.
* :class:`AcousticTokenLM` — generates acoustic tokens from text and a prompt.
* :class:`Vocoder` — renders acoustic tokens back into a waveform.
"""

from .codec import AudioCodec
from .config import CantoConfig, CodecConfig, LMConfig, VocoderConfig
from .errors import (
    AudioIOError,
    CantoError,
    CheckpointError,
    ConfigError,
    EmptyTextError,
    UnsupportedFormatError,
)
from .lm.model import AcousticTokenLM
from .pipeline import CantoTTS, SynthesisResult
from .tokens import AudioTokenizer
from .vocoder import Vocoder

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AudioCodec",
    "AudioTokenizer",
    "AcousticTokenLM",
    "Vocoder",
    "CantoTTS",
    "SynthesisResult",
    "CantoConfig",
    "CodecConfig",
    "LMConfig",
    "VocoderConfig",
    "CantoError",
    "AudioIOError",
    "ConfigError",
    "CheckpointError",
    "EmptyTextError",
    "UnsupportedFormatError",
]
