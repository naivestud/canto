import torch

from canto.lm.sampling import greedy_sample, top_k_filter, top_k_sample


def test_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 0.9, 0.2], [3.0, 0.0, 1.0]])
    assert greedy_sample(logits).tolist() == [1, 0]


def test_top_k_filter_zeros_outside_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    filtered = top_k_filter(logits, 2)
    # Only the two largest (5.0 and 3.0) survive.
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 3] == float("-inf")
    assert filtered[0, 1] == 5.0
    assert filtered[0, 2] == 3.0


def test_top_k_sample_deterministic_with_seed():
    logits = torch.randn(4, 50)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = top_k_sample(logits, 10, generator=g1)
    b = top_k_sample(logits, 10, generator=g2)
    assert a.tolist() == b.tolist()


def test_top_k_respects_k():
    # With k=1 sampling reduces to greedy.
    logits = torch.randn(3, 20)
    g = torch.Generator().manual_seed(7)
    sampled = top_k_sample(logits, 1, generator=g)
    assert sampled.tolist() == greedy_sample(logits).tolist()
