"""Text normalisation and conversion to token ids."""

from __future__ import annotations

import re
import string
from typing import Literal

from .numbers import number_to_words, spell_digits
from .vocab import GraphemeVocab

NumberStyle = Literal["digits", "cardinal"]

_MAX_CARDINAL_RUN = 6  # digit runs longer than this are spelled out instead

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _expand_cardinal(match: re.Match[str]) -> str:
    run = match.group(0)
    if len(run) > _MAX_CARDINAL_RUN:
        return spell_digits(run)
    return number_to_words(int(run))


class TextFrontend:
    """Normalises free text and turns it into grapheme ids.

    The normaliser is deliberately conservative: it lower-cases, collapses runs
    of whitespace, and trims the ends. Language-specific grapheme-to-phoneme
    fallbacks can be layered on later without changing callers.

    ``number_style`` controls how digit runs are verbalised: ``"digits"``
    (the default) spells each digit, which suits phone numbers and ids, while
    ``"cardinal"`` reads short runs as quantities, e.g. ``12`` as
    ``twelve``. With ``keep_punctuation=False`` ASCII punctuation is dropped,
    which some acoustic models prefer.
    """

    def __init__(
        self,
        vocab: GraphemeVocab | None = None,
        number_style: NumberStyle = "digits",
        keep_punctuation: bool = True,
    ) -> None:
        if number_style not in ("digits", "cardinal"):
            raise ValueError(f"unknown number_style: {number_style!r}")
        self.vocab = vocab or GraphemeVocab()
        self.number_style = number_style
        self.keep_punctuation = keep_punctuation

    def normalize(self, text: str) -> str:
        text = text.lower()
        # Spell digits out so they never reach the model as raw characters;
        # a real deployment would swap in a proper G2P here.
        if self.number_style == "cardinal":
            text = re.sub(r"\d+", _expand_cardinal, text)
        else:
            text = spell_digits(text)
        if not self.keep_punctuation:
            text = text.translate(_PUNCTUATION_TABLE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def to_ids(self, text: str) -> list[int]:
        """Normalise ``text`` and return ids framed by BOS and EOS."""
        normalised = self.normalize(text)
        ids = self.vocab.encode(normalised)
        return [self.vocab.bos_id, *ids, self.vocab.eos_id]
