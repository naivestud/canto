"""Shared constants used across canto."""

from __future__ import annotations

# Default sampling rate for all audio handled by the library. The codec and
# vocoder are tuned for this rate; resampling is the caller's responsibility.
SAMPLE_RATE = 24_000

# Default codec geometry. A frame rate of SAMPLE_RATE / prod(strides) frames
# per second; with the default strides this is 24000 / 320 = 75 fps.
DEFAULT_CODEBOOK_SIZE = 1024
DEFAULT_NUM_QUANTIZERS = 1
