import numpy as np

from canto.codec import AudioCodec
from canto.config import CodecConfig
from canto.tokens import AudioTokenizer


def tiny_tokenizer() -> AudioTokenizer:
    config = CodecConfig(
        dim=8,
        enc_strides=(2, 4),
        dec_strides=(4, 2),
        num_quantizers=1,
        codebook_size=16,
        codebook_dim=4,
    )
    return AudioTokenizer(AudioCodec(config))


def test_encode_produces_codes():
    tokenizer = tiny_tokenizer()
    wav = np.random.default_rng(0).standard_normal(8 * 24).astype(np.float32)
    codes = tokenizer.encode(wav)
    assert codes.ndim == 2
    assert codes.shape[0] == 1
    assert codes.dtype.kind == "i"
    assert codes.min() >= 0
    assert codes.max() < 16


def test_roundtrip_lengths():
    tokenizer = tiny_tokenizer()
    wav = np.random.default_rng(1).standard_normal(8 * 24).astype(np.float32)
    codes = tokenizer.encode(wav)
    recon = tokenizer.decode(codes)
    # Hop is 8, so 192 samples -> 24 frames -> 192 samples.
    assert abs(len(recon) - len(wav)) <= 2


def test_1d_codes_accepted_by_decode():
    tokenizer = tiny_tokenizer()
    wav = np.random.default_rng(2).standard_normal(8 * 16).astype(np.float32)
    codes = tokenizer.encode(wav)[0]  # drop quantiser axis
    recon = tokenizer.decode(codes)
    assert recon.ndim == 1


def test_frame_rate_property():
    tokenizer = tiny_tokenizer()
    assert tokenizer.frame_rate == tokenizer.sample_rate / 8
