"""Text chunking helpers for processing long inputs.

Long passages are easier to synthesise and embed when they are broken into
bounded pieces. :func:`chunk_text` splits on word boundaries; the split is
greedy so chunks stay as close to the limit as possible.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences at terminators followed by whitespace.

    The terminator stays with its sentence. Text without terminators comes
    back as a single sentence; whitespace-only input gives no sentences.
    """
    parts = [part.strip() for part in _SENTENCE_END.split(text)]
    return [part for part in parts if part]


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` on word boundaries into chunks of at most ``max_chars``.

    Words longer than ``max_chars`` are kept whole rather than broken, so no
    information is lost. Whitespace is normalised to single spaces.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
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
