from canto.text import GraphemeVocab, TextFrontend


def test_roundtrip_ascii():
    vocab = GraphemeVocab()
    text = "Hello, world!"
    ids = vocab.encode(text)
    assert vocab.decode(ids) == text


def test_unknown_maps_to_unk():
    vocab = GraphemeVocab()
    # An emoji is outside printable ASCII.
    ids = vocab.encode("\U0001F600")
    assert ids == [vocab.unk_id]


def test_special_ids_are_stable():
    vocab = GraphemeVocab()
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1
    assert vocab.bos_id == 2
    assert vocab.eos_id == 3


def test_decode_drops_specials():
    vocab = GraphemeVocab()
    ids = [vocab.bos_id] + vocab.encode("hi") + [vocab.eos_id]
    assert vocab.decode(ids) == "hi"


def test_size_covers_ascii():
    vocab = GraphemeVocab()
    assert vocab.size == 4 + 95


def test_frontend_lowercases_and_collapses_whitespace():
    frontend = TextFrontend()
    assert frontend.normalize("  Hello   WORLD ") == "hello world"


def test_frontend_expands_digits():
    frontend = TextFrontend()
    assert frontend.normalize("call 911") == "call nineoneone"


def test_frontend_to_ids_frames_with_bos_eos():
    frontend = TextFrontend()
    ids = frontend.to_ids("hi")
    vocab = frontend.vocab
    assert ids[0] == vocab.bos_id
    assert ids[-1] == vocab.eos_id
    assert vocab.decode(ids) == "hi"
