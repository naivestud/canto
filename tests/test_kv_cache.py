import torch

from canto.config import LMConfig
from canto.lm.kv_cache import KVCache
from canto.lm.model import AcousticTokenLM


def tiny_lm() -> AcousticTokenLM:
    config = LMConfig(dim=32, num_layers=2, num_heads=2, context=64, codebook_size=16)
    torch.manual_seed(0)
    return AcousticTokenLM(config)


def test_append_grows_and_tracks_length():
    cache = KVCache(num_layers=2)
    assert cache.length == 0
    keys = torch.randn(1, 2, 3, 4)
    values = torch.randn(1, 2, 3, 4)
    full_keys, full_values = cache.append(0, keys, values)
    assert full_keys.shape == (1, 2, 3, 4)
    assert full_values.shape == (1, 2, 3, 4)
    assert cache.length == 3

    step = torch.randn(1, 2, 1, 4)
    full_keys, _ = cache.append(0, step, step)
    assert full_keys.shape[2] == 4


def test_layers_are_independent():
    cache = KVCache(num_layers=2)
    cache.append(0, torch.randn(1, 1, 5, 2), torch.randn(1, 1, 5, 2))
    cache.append(1, torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2))
    full_keys_0, _ = cache.append(0, torch.zeros(1, 1, 0, 2), torch.zeros(1, 1, 0, 2))
    full_keys_1, _ = cache.append(1, torch.zeros(1, 1, 0, 2), torch.zeros(1, 1, 0, 2))
    assert full_keys_0.shape[2] == 5
    assert full_keys_1.shape[2] == 2


def test_crop_keeps_most_recent_positions():
    cache = KVCache(num_layers=1)
    keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1)
    cache.append(0, keys, keys)
    cache.crop(3)
    assert cache.length == 3
    cropped, _ = cache.append(0, torch.zeros(1, 1, 0, 1), torch.zeros(1, 1, 0, 1))
    assert cropped.flatten().tolist() == [5.0, 6.0, 7.0]


def test_crop_below_length_is_noop():
    cache = KVCache(num_layers=1)
    keys = torch.randn(1, 1, 4, 2)
    cache.append(0, keys, keys)
    cache.crop(10)
    assert cache.length == 4


def test_reset_clears_all_layers():
    cache = KVCache(num_layers=2)
    cache.append(0, torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2))
    cache.append(1, torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2))
    cache.reset()
    assert cache.length == 0


def test_cached_generation_matches_uncached():
    lm = tiny_lm()
    text_ids = torch.tensor([[4, 8, 5, 12]])
    prompt = torch.randint(0, 16, (1, 6))
    cached = lm.generate(text_ids, prompt, max_new_tokens=12, use_cache=True)
    uncached = lm.generate(text_ids, prompt, max_new_tokens=12, use_cache=False)
    assert cached.shape == uncached.shape
    assert torch.equal(cached, uncached)


def test_cached_generation_without_prompt_matches():
    lm = tiny_lm()
    text_ids = torch.tensor([[4, 8, 5, 12]])
    cached = lm.generate(text_ids, None, max_new_tokens=10, use_cache=True)
    uncached = lm.generate(text_ids, None, max_new_tokens=10, use_cache=False)
    assert torch.equal(cached, uncached)


def test_cached_generation_is_deterministic():
    lm = tiny_lm()
    text_ids = torch.tensor([[4, 8, 5, 12]])
    a = lm.generate(text_ids, None, max_new_tokens=8)
    b = lm.generate(text_ids, None, max_new_tokens=8)
    assert torch.equal(a, b)
