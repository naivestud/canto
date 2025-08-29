import torch

from canto.codec import AudioCodec
from canto.config import CodecConfig


def tiny_codec() -> AudioCodec:
    config = CodecConfig(
        dim=8,
        enc_strides=(2, 4),
        dec_strides=(4, 2),
        num_quantizers=1,
        codebook_size=16,
        codebook_dim=4,
    )
    return AudioCodec(config)


def test_encode_shape():
    codec = tiny_codec()
    wav = torch.randn(1, 1, 8 * 20)
    indices = codec.encode(wav)
    # One quantiser, and 160 samples at hop 8 gives 20 frames.
    assert indices.shape == (1, 1, 20)
    assert indices.min() >= 0
    assert indices.max() < 16


def test_decode_shape_roundtrip():
    codec = tiny_codec()
    wav = torch.randn(1, 1, 8 * 20)
    indices = codec.encode(wav)
    recon = codec.decode_indices(indices)
    assert recon.shape[1] == 1
    # Transposed-conv arithmetic reconstructs the clean multiple exactly here.
    assert abs(recon.shape[-1] - 160) <= 2


def test_forward_returns_reconstruction_indices_loss():
    codec = tiny_codec()
    wav = torch.randn(2, 1, 8 * 16)
    recon, indices, loss = codec(wav)
    assert recon.shape[0] == 2
    assert indices.shape[0] == 2
    assert loss.ndim == 0


def test_output_is_bounded():
    codec = tiny_codec()
    wav = torch.randn(1, 1, 8 * 20)
    recon = codec.decode_indices(codec.encode(wav))
    assert recon.abs().max() <= 1.0
