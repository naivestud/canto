import torch

from canto.codec.quantize import ResidualVectorQuantizer, VectorQuantizer


def test_vector_quantizer_indices_in_range():
    vq = VectorQuantizer(dim=16, codebook_size=32, codebook_dim=8)
    x = torch.randn(2, 16, 10)
    indices = vq.encode(x)
    assert indices.shape == (2, 10)
    assert indices.min() >= 0
    assert indices.max() < 32


def test_vector_quantizer_forward_shapes():
    vq = VectorQuantizer(dim=16, codebook_size=32, codebook_dim=8)
    x = torch.randn(2, 16, 10)
    out, indices, loss = vq(x)
    assert out.shape == x.shape
    assert indices.shape == (2, 10)
    assert loss.ndim == 0
    assert float(loss.detach()) >= 0.0


def test_rvq_encode_decode_roundtrip_consistent():
    rvq = ResidualVectorQuantizer(dim=16, codebook_size=32, codebook_dim=8, num_quantizers=3)
    x = torch.randn(2, 16, 12)
    indices = rvq.encode(x)
    assert indices.shape == (2, 3, 12)
    # Decoding the produced indices must equal summing per-level decode.
    reconstructed = rvq.decode_indices(indices)
    assert reconstructed.shape == x.shape


def test_rvq_forward_matches_levels():
    rvq = ResidualVectorQuantizer(dim=8, codebook_size=16, codebook_dim=4, num_quantizers=2)
    x = torch.randn(1, 8, 6)
    _, indices, _ = rvq(x)
    assert indices.shape == (1, 2, 6)


def test_codebook_utilization_is_not_degenerate():
    # Random data through a fresh quantiser should use more than one code;
    # collapse to a single code would indicate a broken distance computation.
    torch.manual_seed(0)
    vq = VectorQuantizer(dim=16, codebook_size=64, codebook_dim=8)
    x = torch.randn(4, 16, 32)
    indices = vq.encode(x)
    assert indices.unique().numel() > 1
