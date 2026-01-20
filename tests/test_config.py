import pytest

from canto.config import CantoConfig, CodecConfig, LMConfig, VocoderConfig
from canto.errors import ConfigError


def test_tiny_preset_is_valid():
    config = CantoConfig.tiny()
    config.validate()  # must not raise
    assert config.lm.codebook_size == config.codec.codebook_size
    assert config.vocoder.upsample_factor == config.codec.hop_length


def test_default_config_is_valid():
    CantoConfig().validate()


def test_hop_length_is_stride_product():
    codec = CodecConfig(enc_strides=(2, 4, 5, 8), dec_strides=(8, 5, 4, 2))
    assert codec.hop_length == 320
    assert codec.frame_rate == pytest.approx(24_000 / 320)


def test_codec_rejects_nonpositive_dim():
    with pytest.raises(ConfigError):
        CodecConfig(dim=0).validate()


def test_codec_rejects_mismatched_strides():
    with pytest.raises(ConfigError):
        CodecConfig(enc_strides=(2, 4), dec_strides=(4, 4)).validate()


def test_codec_rejects_bad_quantizer_counts():
    with pytest.raises(ConfigError):
        CodecConfig(num_quantizers=0).validate()
    with pytest.raises(ConfigError):
        CodecConfig(codebook_size=1).validate()


def test_lm_rejects_indivisible_heads():
    with pytest.raises(ConfigError):
        LMConfig(dim=10, num_heads=3).validate()


def test_lm_rejects_small_depth_or_context():
    with pytest.raises(ConfigError):
        LMConfig(num_layers=0).validate()
    with pytest.raises(ConfigError):
        LMConfig(context=1).validate()


def test_vocoder_rejects_nonpositive_channels():
    with pytest.raises(ConfigError):
        VocoderConfig(channels=0).validate()
    with pytest.raises(ConfigError):
        VocoderConfig(embed_dim=0).validate()


def test_top_level_rejects_codebook_mismatch():
    config = CantoConfig.tiny()
    config.lm.codebook_size = config.codec.codebook_size + 1
    with pytest.raises(ConfigError):
        config.validate()


def test_top_level_rejects_upsample_mismatch():
    config = CantoConfig.tiny()
    config.vocoder.upsample_rates = (2, 2)
    with pytest.raises(ConfigError):
        config.validate()


def test_dict_roundtrip_preserves_everything():
    config = CantoConfig.tiny()
    restored = CantoConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.to_dict() == config.to_dict()


def test_dict_roundtrip_restores_tuples():
    restored = CantoConfig.from_dict(CantoConfig.tiny().to_dict())
    assert isinstance(restored.codec.enc_strides, tuple)
    assert isinstance(restored.codec.dec_strides, tuple)
    assert isinstance(restored.vocoder.upsample_rates, tuple)
    assert isinstance(restored.vocoder.refinements, tuple)


def test_from_dict_defaults_sample_rate():
    data = CantoConfig.tiny().to_dict()
    del data["sample_rate"]
    restored = CantoConfig.from_dict(data)
    assert restored.sample_rate == 24_000


def test_from_dict_rejects_malformed_sections():
    with pytest.raises(ConfigError):
        CantoConfig.from_dict({})
    with pytest.raises(ConfigError):
        CantoConfig.from_dict({"codec": {}, "lm": {}, "vocoder": {}})
