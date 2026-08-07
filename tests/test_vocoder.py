import torch

from canto.vocoder import Vocoder


def test_upsample_length():
    vocoder = Vocoder(num_tokens=16, embed_dim=8, channels=8, upsample_rates=(4, 2))
    tokens = torch.randint(0, 16, (1, 10))
    wav = vocoder(tokens)
    assert wav.shape == (1, 1, 10 * 8)


def test_output_bounded():
    vocoder = Vocoder(num_tokens=16, embed_dim=8, channels=8, upsample_rates=(2, 2))
    tokens = torch.randint(0, 16, (2, 6))
    wav = vocoder(tokens)
    assert wav.abs().max() <= 1.0


def test_upsample_factor_property():
    vocoder = Vocoder(num_tokens=8, upsample_rates=(5, 4))
    assert vocoder.upsample_factor == 20


def test_batch_independence():
    torch.manual_seed(0)
    vocoder = Vocoder(num_tokens=16, embed_dim=8, channels=8, upsample_rates=(4,))
    vocoder.eval()
    a = torch.randint(0, 16, (1, 5))
    b = torch.randint(0, 16, (1, 5))
    batched = torch.cat([a, b], dim=0)
    with torch.no_grad():
        out = vocoder(batched)
        single_a = vocoder(a)
    torch.testing.assert_close(out[0], single_a[0])
