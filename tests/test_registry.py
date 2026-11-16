"""Tests for digitrec.models.registry: lookup, lazy loading errors and caching."""

import pytest
import torch

from digitrec.config import Config
from digitrec.models import DigitPrediction, probs_to_predictions
from digitrec.models import registry


class _FakePredictor:
    name = "fake"

    def __init__(self, config: Config):
        self.config = config

    def predict(self, crops: torch.Tensor) -> list[DigitPrediction]:
        return probs_to_predictions(torch.full((crops.shape[0], 10), 0.1))


@pytest.fixture
def fake_backend(monkeypatch):
    """Register a 'fake' backend that counts how often it is loaded; undone after each test."""
    monkeypatch.setattr(registry, "_BACKENDS", dict(registry._BACKENDS))
    calls = []

    def factory(config):
        calls.append(config)
        return _FakePredictor(config)

    registry.register_backend("fake", factory)
    yield calls
    registry.clear_cache()


def test_builtin_backends_listed():
    assert {"pretrained", "scratch_cnn"} <= set(registry.available_backends())


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        registry.get_predictor(Config(backend="nope", device="cpu"))


def test_get_predictor_loads_backend(fake_backend):
    cfg = Config(backend="fake", device="cpu")
    predictor = registry.get_predictor(cfg)
    assert isinstance(predictor, _FakePredictor)
    assert predictor.config is cfg


def test_predictor_is_cached_per_config(fake_backend):
    cfg = Config(backend="fake", device="cpu")
    first = registry.get_predictor(cfg)
    second = registry.get_predictor(Config(backend="fake", device="cpu"))  # equal, new object
    assert first is second
    assert len(fake_backend) == 1  # factory ran only once


def test_different_config_loads_again(fake_backend):
    registry.get_predictor(Config(backend="fake", device="cpu"))
    registry.get_predictor(Config(backend="fake", device="cpu", checkpoint_path="other.pt"))
    assert len(fake_backend) == 2


def test_missing_backend_module_is_reported(monkeypatch):
    monkeypatch.setattr(registry, "_BACKENDS", {"ghost": "digitrec.models.does_not_exist:load"})
    registry.clear_cache()
    with pytest.raises(NotImplementedError, match="does not exist yet"):
        registry.get_predictor(Config(backend="ghost", device="cpu"))
