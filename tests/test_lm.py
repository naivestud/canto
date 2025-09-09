import torch

from canto.config import LMConfig
from canto.lm.model import AcousticTokenLM
from canto.text import GraphemeVocab


def tiny_lm() -> AcousticTokenLM:
    config = LMConfig(dim=32, num_layers=2, num_heads=2, context=64, codebook_size=16)
    return AcousticTokenLM(config)


def test_forward_logits_shape():
    lm = tiny_lm()
    vocab = GraphemeVocab()
    text_ids = torch.tensor([vocab.encode("hi")])
    prompt = torch.randint(0, 16, (1, 5))
    target = torch.randint(0, 16, (1, 7))
    logits = lm(text_ids, prompt, target)
    # seq = 2 text + 1 audio_bos + 5 prompt + 7 target + 1 audio_eos = 16
    assert logits.shape == (1, 15, lm.vocab_size)


def test_vocab_layout():
    lm = tiny_lm()
    vocab = GraphemeVocab()
    assert lm.audio_offset == vocab.size
    assert lm.audio_bos_id == vocab.size + 16
    assert lm.audio_eos_id == vocab.size + 17
    assert lm.vocab_size == vocab.size + 18


def test_audio_id_roundtrip():
    lm = tiny_lm()
    codes = torch.randint(0, 16, (2, 4))
    lm_ids = lm.audio_to_lm(codes)
    assert lm.lm_to_audio(lm_ids).equal(codes)
    assert (lm_ids >= lm.audio_offset).all()


def test_no_prompt_forward():
    lm = tiny_lm()
    vocab = GraphemeVocab()
    text_ids = torch.tensor([vocab.encode("a")])
    target = torch.randint(0, 16, (1, 3))
    logits = lm(text_ids, None, target)
    assert logits.ndim == 3
