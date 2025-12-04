import pytest

from canto.text.numbers import number_to_words, spell_digits


def test_spell_digits_replaces_only_digits():
    assert spell_digits("call 911") == "call nineoneone"
    assert spell_digits("no digits") == "no digits"
    assert spell_digits("0") == "zero"


def test_number_to_words_small_values():
    assert number_to_words(0) == "zero"
    assert number_to_words(7) == "seven"
    assert number_to_words(13) == "thirteen"
    assert number_to_words(42) == "forty two"


def test_number_to_words_hundreds():
    assert number_to_words(100) == "one hundred"
    assert number_to_words(105) == "one hundred five"
    assert number_to_words(350) == "three hundred fifty"
    assert number_to_words(999) == "nine hundred ninety nine"


def test_number_to_words_thousands():
    assert number_to_words(1000) == "one thousand"
    assert number_to_words(12_345) == "twelve thousand three hundred forty five"
    assert number_to_words(20_006) == "twenty thousand six"


def test_number_to_words_negative():
    assert number_to_words(-5) == "minus five"
    assert number_to_words(-1200) == "minus one thousand two hundred"


def test_number_to_words_out_of_range_raises():
    with pytest.raises(ValueError):
        number_to_words(1_000_000)
    with pytest.raises(ValueError):
        number_to_words(-1_000_000)
