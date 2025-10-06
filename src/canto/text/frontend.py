"""Text normalisation and conversion to token ids."""

from __future__ import annotations

import re

from .vocab import GraphemeVocab

# A deliberately small fallback so digits do not reach the model as raw
# characters; a real deployment would swap in a proper G2P here.
_DIGITS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


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
        text = "".join(_DIGITS.get(char, char) for char in text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def to_ids(self, text: str) -> list[int]:
        """Normalise ``text`` and return ids framed by BOS and EOS."""
        normalised = self.normalize(text)
        ids = self.vocab.encode(normalised)
        return [self.vocab.bos_id, *ids, self.vocab.eos_id]
