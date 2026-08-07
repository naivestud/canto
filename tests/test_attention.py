import torch

from canto.lm.attention import CausalSelfAttention


def test_output_shape():
    attn = CausalSelfAttention(dim=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert out.shape == x.shape


def test_causality_future_does_not_affect_past():
    torch.manual_seed(0)
    attn = CausalSelfAttention(dim=16, num_heads=2)
    attn.eval()

    x = torch.randn(1, 8, 16)
    y = x.clone()
    y[:, 5:, :] = torch.randn(1, 3, 16)  # perturb only the future

    with torch.no_grad():
        out_x = attn(x)
        out_y = attn(y)

    # Positions before the perturbation must be identical.
    torch.testing.assert_close(out_x[:, :5, :], out_y[:, :5, :])
    # The perturbed positions themselves should generally differ.
    assert not torch.allclose(out_x[:, 5:, :], out_y[:, 5:, :])


def test_invalid_heads_raise():
    try:
        CausalSelfAttention(dim=10, num_heads=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible heads")
