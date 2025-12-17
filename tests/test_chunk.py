import pytest

from canto.text.chunk import chunk_text, split_sentences


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", max_chars=80) == ["hello world"]


def test_empty_text_is_no_chunks():
    assert chunk_text("", max_chars=80) == []
    assert chunk_text("   ", max_chars=80) == []


def test_chunks_respect_limit():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text(text, max_chars=20)
    assert all(len(chunk) <= 20 for chunk in chunks)
    # Joining the chunks back together recovers the original words.
    assert " ".join(chunks) == text


def test_long_word_kept_whole():
    # A word longer than the limit is not broken apart.
    chunks = chunk_text("abcdefghij short", max_chars=4)
    assert chunks[0] == "abcdefghij"


def test_whitespace_normalised():
    assert chunk_text("  a   b  ", max_chars=80) == ["a b"]


def test_invalid_max_chars_raises():
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=0)


def test_split_sentences_basic():
    text = "Hello there. How are you? Fine!"
    assert split_sentences(text) == ["Hello there.", "How are you?", "Fine!"]


def test_split_sentences_keeps_terminator():
    parts = split_sentences("Wait... really? Yes.")
    assert parts[0] == "Wait..."


def test_split_sentences_no_terminator_is_single():
    assert split_sentences("no terminators here") == ["no terminators here"]


def test_split_sentences_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_ignores_mid_sentence_dots():
    # A dot not followed by whitespace does not end the sentence.
    assert split_sentences("a 1.5x speedup") == ["a 1.5x speedup"]
