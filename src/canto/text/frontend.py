"""Text normalisation and conversion to token ids."""

from __future__ import annotations

import re

from .vocab import GraphemeVocab


class TextFrontend:
    """Normalises free text and turns it into grapheme ids.

    The normaliser is deliberately conservative: it lower-cases, collapses runs
    of whitespace, and trims the ends. Language-specific grapheme-to-phoneme
    fallbacks can be layered on later without changing callers.
    """

    def __init__(self, vocab: GraphemeVocab | None = None) -> None:
        self.vocab = vocab or GraphemeVocab()

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def to_ids(self, text: str) -> list[int]:
        """Normalise ``text`` and return ids framed by BOS and EOS."""
        normalised = self.normalize(text)
        ids = self.vocab.encode(normalised)
        return [self.vocab.bos_id, *ids, self.vocab.eos_id]
