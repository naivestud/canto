"""Small shared helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed the RNGs used during sampling so synthesis is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
