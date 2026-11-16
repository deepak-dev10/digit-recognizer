"""Tests for digitrec.config: device selection and Config defaults."""

from pathlib import Path

import pytest
import torch

from digitrec import config
from digitrec.config import Config, get_device


def test_auto_device_is_available():
    """Auto-detect must return a device this machine can use."""
    device = get_device()
    assert device.type in ("cuda", "mps", "cpu")
    torch.zeros(1, device=device)  # fails if the device can't be used


def test_cpu_always_available():
    assert get_device("cpu") == torch.device("cpu")
    assert get_device("CPU") == torch.device("cpu")  # case-insensitive


def test_auto_prefers_mps_over_cpu(monkeypatch):
    """With no CUDA but MPS available, auto-detect picks MPS."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device().type == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device().type == "cpu"


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        get_device("tpu")


def test_unavailable_device_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        get_device("cuda")


def test_config_defaults():
    cfg = Config(device="cpu")
    assert cfg.backend == "pretrained"
    assert cfg.checkpoint_path == config.CHECKPOINT_DIR / "cnn_best.pt"
    assert cfg.device == torch.device("cpu")


def test_config_normalizes_types():
    """A str checkpoint path and a str device are converted to Path / torch.device."""
    cfg = Config(checkpoint_path="weights.pt", device="cpu")
    assert isinstance(cfg.checkpoint_path, Path)
    assert isinstance(cfg.device, torch.device)


def test_config_is_immutable():
    cfg = Config(device="cpu")
    with pytest.raises(AttributeError):
        cfg.backend = "scratch_cnn"  # type: ignore[misc]
