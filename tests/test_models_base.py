"""Tests for digitrec.models.base: DigitPrediction validation and the Predictor contract."""

import pytest
import torch

from digitrec.models import DigitPrediction, Predictor, probs_to_predictions


def test_probs_to_predictions_picks_argmax():
    probs = torch.zeros(2, 10)
    probs[0, 7] = 1.0                 # certain it's a 7
    probs[1, 3], probs[1, 8] = 0.6, 0.4  # leaning towards 3
    preds = probs_to_predictions(probs)

    assert [p.label for p in preds] == [7, 3]
    assert preds[0].confidence == pytest.approx(1.0)
    assert preds[1].confidence == pytest.approx(0.6)
    assert len(preds[1].probs) == 10


def test_probs_to_predictions_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        probs_to_predictions(torch.zeros(10))  # missing batch dimension


def test_digit_prediction_validates_label():
    with pytest.raises(ValueError, match="label"):
        DigitPrediction(label=10, confidence=0.9, probs=(0.1,) * 10)


def test_digit_prediction_validates_probs_length():
    with pytest.raises(ValueError, match="probs"):
        DigitPrediction(label=1, confidence=0.9, probs=(0.5, 0.5))


class _AlwaysSeven:
    """Minimal fake backend: no inheritance, it just matches the Protocol."""

    name = "always_seven"

    def predict(self, crops: torch.Tensor) -> list[DigitPrediction]:
        probs = torch.zeros(crops.shape[0], 10)
        probs[:, 7] = 1.0
        return probs_to_predictions(probs)


def test_duck_typed_backend_satisfies_protocol():
    backend = _AlwaysSeven()
    assert isinstance(backend, Predictor)
    preds = backend.predict(torch.rand(3, 1, 28, 28))
    assert [p.label for p in preds] == [7, 7, 7]
