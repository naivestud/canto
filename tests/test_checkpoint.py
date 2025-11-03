import torch

from canto.checkpoint import load_checkpoint, save_checkpoint
from canto.codec import AudioCodec
from canto.config import CantoConfig


def test_checkpoint_roundtrip(tmp_path):
    config = CantoConfig.tiny()
    codec = AudioCodec(config.codec)
    directory = tmp_path / "ckpt"

    save_checkpoint(directory, config, {"codec": codec})
    loaded_config, state = load_checkpoint(directory)

    assert loaded_config.codec.codebook_size == config.codec.codebook_size
    assert loaded_config.lm.codebook_size == config.lm.codebook_size
    assert loaded_config.vocoder.upsample_factor == config.codec.hop_length
    assert set(state.keys()) == {"codec"}
    # State dict tensors survive the round trip.
    for key, tensor in codec.state_dict().items():
        assert torch.equal(state["codec"][key], tensor)


def test_tuple_fields_restored(tmp_path):
    config = CantoConfig.tiny()
    codec = AudioCodec(config.codec)
    save_checkpoint(tmp_path / "c", config, {"codec": codec})
    loaded_config, _ = load_checkpoint(tmp_path / "c")
    assert isinstance(loaded_config.codec.enc_strides, tuple)
    assert isinstance(loaded_config.vocoder.upsample_rates, tuple)


def test_missing_directory_raises(tmp_path):
    from canto.errors import CheckpointError

    try:
        load_checkpoint(tmp_path / "nope")
    except CheckpointError:
        return
    raise AssertionError("expected CheckpointError")


def test_missing_config_key_raises(tmp_path):
    import json

    from canto.errors import CheckpointError

    directory = tmp_path / "bad"
    directory.mkdir()
    # Valid format marker but a config section that is missing required keys.
    (directory / "config.json").write_text(json.dumps({"format": "canto/v1", "config": {}}))
    (directory / "state.pt").write_bytes(b"placeholder")
    try:
        load_checkpoint(directory)
    except CheckpointError:
        return
    raise AssertionError("expected CheckpointError for missing config keys")
