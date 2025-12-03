"""Number-to-words helpers for the text front-end.

Speech models expect words, not digits, so the front-end rewrites digit runs
before encoding. Two styles are offered: spelling each digit individually
(``spell_digits``), which suits phone numbers and ids, and cardinal expansion
(``number_to_words``), which suits quantities.
"""

from __future__ import annotations

DIGIT_WORDS = {
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

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]

# Cardinal expansion is supported up to (but excluding) one million; longer
# digit runs are better spelled digit by digit.
MAX_CARDINAL = 1_000_000


def spell_digits(text: str) -> str:
    """Replace every digit character with its word, leaving other text alone."""
    return "".join(DIGIT_WORDS.get(char, char) for char in text)


def _below_hundred(value: int) -> str:
    if value < 20:
        return _ONES[value]
    tens, ones = divmod(value, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_ONES[ones]}"


def _below_thousand(value: int) -> str:
    hundreds, remainder = divmod(value, 100)
    words = _below_hundred(remainder) if remainder else ""
    if hundreds:
        words = f"{_ONES[hundreds]} hundred" + (f" {words}" if words else "")
    return words


def number_to_words(value: int) -> str:
    """Spell an integer as cardinal English words.

    Supports ``-999_999`` to ``999_999``; raises :class:`ValueError` outside
    that range so callers can fall back to digit spelling.
    """
    if not -MAX_CARDINAL < value < MAX_CARDINAL:
        raise ValueError(f"value out of range for cardinal expansion: {value}")
    if value < 0:
        return f"minus {number_to_words(-value)}"
    if value == 0:
        return _ONES[0]

    thousands, remainder = divmod(value, 1000)
    words = _below_thousand(remainder) if remainder else ""
    if thousands:
        words = f"{_below_thousand(thousands)} thousand" + (f" {words}" if words else "")
    return words
