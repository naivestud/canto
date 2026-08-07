"""Save and load model checkpoints as self-describing directories.

A checkpoint directory holds a ``config.json`` describing the architecture and
a ``state.pt`` with the state dicts of every component, so a pipeline can be
rebuilt from disk without extra metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import CantoConfig, CodecConfig, LMConfig, VocoderConfig
from .errors import CheckpointError

CONFIG_NAME = "config.json"
STATE_NAME = "state.pt"


def save_checkpoint(
    directory: str | Path,
    config: CantoConfig,
    models: dict[str, torch.nn.Module],
) -> Path:
    """Write ``config`` and the ``models`` state dicts into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / CONFIG_NAME).write_text(json.dumps(asdict(config), indent=2))
    state = {name: module.state_dict() for name, module in models.items()}
    torch.save(state, directory / STATE_NAME)
    return directory


def _config_from_dict(data: dict) -> CantoConfig:
    codec = CodecConfig(
        **{
            **data["codec"],
            "enc_strides": tuple(data["codec"]["enc_strides"]),
            "dec_strides": tuple(data["codec"]["dec_strides"]),
        }
    )
    lm = LMConfig(**data["lm"])
    vocoder = VocoderConfig(
        **{
            **data["vocoder"],
            "upsample_rates": tuple(data["vocoder"]["upsample_rates"]),
            "refinements": tuple(data["vocoder"]["refinements"]),
        }
    )
    return CantoConfig(sample_rate=data.get("sample_rate", 24_000), codec=codec, lm=lm, vocoder=vocoder)


def load_checkpoint(directory: str | Path) -> tuple[CantoConfig, dict[str, torch.Tensor]]:
    """Read a checkpoint directory, returning ``(config, state_dicts)``."""
    directory = Path(directory)
    config_path = directory / CONFIG_NAME
    state_path = directory / STATE_NAME
    if not config_path.exists() or not state_path.exists():
        raise CheckpointError(f"not a checkpoint directory: {directory}")

    data = json.loads(config_path.read_text())
    try:
        config = _config_from_dict(data)
    except (KeyError, TypeError) as exc:
        raise CheckpointError(f"malformed config in {config_path}") from exc

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    return config, state
