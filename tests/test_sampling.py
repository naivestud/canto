import pytest
import torch

from canto.lm.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    greedy_sample,
    min_p_filter,
    sample,
    top_k_filter,
    top_k_sample,
    top_p_filter,
)


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


def test_top_p_filter_keeps_nucleus():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    filtered = top_p_filter(logits, 0.9)
    # The dominant token must survive; at least one token is always kept.
    assert filtered[0, 1] == 5.0
    assert (filtered > float("-inf")).any()


def test_sample_with_temperature_is_valid():
    logits = torch.randn(4, 30)
    g = torch.Generator().manual_seed(5)
    ids = sample(logits, temperature=0.8, top_k=10, generator=g)
    assert ids.shape == (4,)
    assert ids.min() >= 0
    assert ids.max() < 30


def test_sample_top_p_path_deterministic():
    logits = torch.randn(2, 25)
    g1 = torch.Generator().manual_seed(9)
    g2 = torch.Generator().manual_seed(9)
    a = sample(logits, top_p=0.9, generator=g1)
    b = sample(logits, top_p=0.9, generator=g2)
    assert a.tolist() == b.tolist()


def test_min_p_filter_keeps_confident_tokens():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    filtered = min_p_filter(logits, 0.1)
    # Softmax gives roughly [0.01, 0.87, 0.12, 0.04]; the top token's
    # probability is ~0.87, so the 0.1 cutoff is ~0.087.
    assert filtered[0, 1] == 5.0
    assert filtered[0, 2] == 3.0
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 3] == float("-inf")


def test_min_p_filter_always_keeps_top():
    logits = torch.randn(3, 20)
    filtered = min_p_filter(logits, 0.99)
    top = greedy_sample(logits)
    for row in range(3):
        assert filtered[row, top[row]] > float("-inf")


def test_min_p_filter_disabled_by_zero():
    logits = torch.randn(2, 10)
    assert torch.equal(min_p_filter(logits, 0.0), logits)


def test_min_p_adapts_to_flat_distribution():
    # A uniform distribution keeps every token at any p <= 1.
    logits = torch.zeros(1, 8)
    filtered = min_p_filter(logits, 1.0)
    assert (filtered > float("-inf")).all()


def test_repetition_penalty_downweights_previous_tokens():
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    previous = torch.tensor([[0, 2]])
    penalised = apply_repetition_penalty(logits, previous, penalty=2.0)
    # Positive logits are divided, negative ones multiplied, others untouched.
    assert penalised[0, 0] == pytest.approx(0.5)
    assert penalised[0, 1] == pytest.approx(2.0)
    assert penalised[0, 2] == pytest.approx(-2.0)


def test_repetition_penalty_one_is_noop():
    logits = torch.randn(2, 10)
    previous = torch.tensor([[1, 2], [3, 4]])
    assert torch.equal(apply_repetition_penalty(logits, previous, 1.0), logits)


def test_repetition_penalty_empty_history_is_noop():
    logits = torch.randn(2, 10)
    empty = torch.zeros(2, 0, dtype=torch.long)
    assert torch.equal(apply_repetition_penalty(logits, empty, 2.0), logits)


def test_temperature_sharpens_and_flattens():
    logits = torch.tensor([[0.0, 2.0]])
    sharp = apply_temperature(logits, 0.5)
    flat = apply_temperature(logits, 2.0)
    # Sharpening widens the gap between logits, flattening narrows it.
    assert (sharp[0, 1] - sharp[0, 0]) == pytest.approx(4.0)
    assert (flat[0, 1] - flat[0, 0]) == pytest.approx(1.0)


def test_temperature_must_be_positive():
    logits = torch.randn(1, 5)
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, -1.0)
