"""Exception hierarchy for canto.

Every error raised by the library derives from :class:`CantoError` so callers
can catch a single base type when they do not care about the specifics.
"""

from __future__ import annotations


class CantoError(Exception):
    """Base class for all canto errors."""


class AudioIOError(CantoError):
    """Raised when audio cannot be read or written."""


class ConfigError(CantoError):
    """Raised for invalid or inconsistent configuration values."""


class CheckpointError(CantoError):
    """Raised when a checkpoint cannot be saved or loaded."""


class EmptyTextError(CantoError):
    """Raised when synthesis is requested with no usable text."""


class UnsupportedFormatError(CantoError):
    """Raised for unsupported audio or token formats."""
