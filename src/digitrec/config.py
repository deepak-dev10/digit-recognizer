"""Central configuration: which model backend to use, where weights live, and which device to run on.

Everything that might differ between machines or phases lives here, so the rest of the
code never hard-codes a backend name, a path, or a device.

Example::

    from digitrec.config import Config

    cfg = Config()                          # pretrained backend, best available device
    cfg = Config(backend="scratch_cnn")     # Phase 2: our own CNN
    cfg = Config(device="cpu")              # force CPU, e.g. for reproducible tests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# Repository root (…/digit_recognizer). This file is at src/digitrec/config.py, so the root is
# three levels up. The path only makes sense in a source checkout, which is how the project is used.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory for Phase 2 trained weights (gitignored).
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Phase 1 uses a pretrained model; Phase 2 switches this to "scratch_cnn".
DEFAULT_BACKEND = "pretrained"

# Default checkpoint written by training/train.py and read by the scratch_cnn backend.
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "cnn_best.pt"

# Device types we know how to pick, in order of preference.
_DEVICE_PREFERENCE = ("cuda", "mps", "cpu")


def _is_available(device_type: str) -> bool:
    """Return True if PyTorch can run on ``device_type`` on this machine."""
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "mps":  # Apple Silicon GPU
        return torch.backends.mps.is_available()
    return device_type == "cpu"


def get_device(preferred: str | None = None) -> torch.device:
    """Choose the torch device to run models on.

    Args:
        preferred: ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None``. ``None`` means pick
            the best one available (cuda, then mps, then cpu).

    Returns:
        The selected :class:`torch.device`.

    Raises:
        ValueError: If ``preferred`` is not a device type we know.
        RuntimeError: If ``preferred`` is a known device type that this machine can't use.
            We raise instead of quietly falling back to CPU, so an explicit request
            is never silently ignored.
    """
    if preferred is None:
        return torch.device(next(d for d in _DEVICE_PREFERENCE if _is_available(d)))

    preferred = preferred.lower()
    if preferred not in _DEVICE_PREFERENCE:
        raise ValueError(f"Unknown device {preferred!r}; expected one of {_DEVICE_PREFERENCE}")
    if not _is_available(preferred):
        raise RuntimeError(f"Device {preferred!r} was requested but is not available on this machine")
    return torch.device(preferred)


@dataclass(frozen=True)
class Config:
    """Settings for one pipeline run. Immutable, so a single config can be shared safely.

    Attributes:
        backend: Model backend name, looked up in ``models.registry``
            (``"pretrained"`` in Phase 1, ``"scratch_cnn"`` in Phase 2).
        checkpoint_path: Weights file for backends that load local weights (``scratch_cnn``).
            The pretrained backend ignores it and downloads from Hugging Face Hub.
        device: The resolved :class:`torch.device`. If you pass a string such as ``"cpu"``
            (or ``None`` for auto-detect), it is turned into a device when the config is created.
    """

    backend: str = DEFAULT_BACKEND
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    device: torch.device = field(default=None)  # type: ignore[assignment]  # set in __post_init__

    def __post_init__(self) -> None:
        # Turn ``device`` (None, a str, or a torch.device) into a torch.device.
        # The dataclass is frozen, so we have to assign with object.__setattr__.
        device = self.device
        if not isinstance(device, torch.device):
            device = get_device(device)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
