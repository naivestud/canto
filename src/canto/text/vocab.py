"""A small grapheme vocabulary for the text front-end.

Keeping the unit as a grapheme (one character) avoids a phonemiser dependency
while still giving the language model a compact, deterministic input space.
"""

from __future__ import annotations


class GraphemeVocab:
    """Maps printable ASCII characters to integer ids and back."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self) -> None:
        self.specials = [self.PAD, self.UNK, self.BOS, self.EOS]
        self.graphemes = [chr(code) for code in range(32, 127)]
        self.itos = self.specials + self.graphemes
        self.stoi = {symbol: index for index, symbol in enumerate(self.itos)}

        self.pad_id = self.stoi[self.PAD]
        self.unk_id = self.stoi[self.UNK]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of grapheme ids; unknowns map to UNK."""
        return [self.stoi.get(char, self.unk_id) for char in text]

    def decode(self, ids: list[int]) -> str:
        """Convert ids back to a string, dropping special tokens."""
        n_specials = len(self.specials)
        return "".join(self.itos[i] for i in ids if i >= n_specials)
