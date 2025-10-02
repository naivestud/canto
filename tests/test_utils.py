import numpy as np
import torch

from canto.utils import seed_everything


def test_seed_everything_makes_torch_deterministic():
    seed_everything(42)
    a = torch.randn(5)
    seed_everything(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_seed_everything_makes_numpy_deterministic():
    seed_everything(7)
    a = np.random.rand(5)
    seed_everything(7)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    seed_everything(1)
    a = torch.randn(8)
    seed_everything(2)
    b = torch.randn(8)
    assert not torch.equal(a, b)
