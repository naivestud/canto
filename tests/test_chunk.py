import pytest

from canto.text.chunk import chunk_text


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
