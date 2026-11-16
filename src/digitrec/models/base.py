"""The contract every model backend follows.

The pipeline only ever talks to a :class:`Predictor`. It hands over a batch of digit crops
and gets back one :class:`DigitPrediction` per crop. Swapping the Phase 1 pretrained model
for the Phase 2 CNN therefore needs no change outside ``models/``.

Input contract for ``Predictor.predict``
----------------------------------------
``crops`` is a float tensor of shape ``[N, 1, 28, 28]`` where:

* ``N`` is the number of digits (1 for single-digit images, more after segmentation).
* values are in ``[0, 1]``, **digit white (1.0) on black (0.0)**, the MNIST convention.
* the tensor is **not normalised**. Each backend applies the normalisation it was trained
  with (MNIST mean/std for our CNN, the Hugging Face image processor's stats for the
  pretrained model) inside ``predict``. Preprocessing stays the same for every backend.
* the tensor may be on any device; the predictor moves it to its own device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

# Digits 0–9.
NUM_CLASSES = 10


@dataclass(frozen=True)
class DigitPrediction:
    """The classifier's answer for one digit crop.

    Attributes:
        label: The predicted digit, 0–9 (the class with the highest probability).
        confidence: Probability of ``label``, in ``[0, 1]``. The pipeline uses it to
            flag uncertain predictions.
        probs: Probability for each digit; ``probs[d]`` is P(digit == d). The 10 values
            sum to ~1. Shown as a bar chart in the UI.
    """

    label: int
    confidence: float
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.label < NUM_CLASSES:
            raise ValueError(f"label must be in 0..{NUM_CLASSES - 1}, got {self.label}")
        if len(self.probs) != NUM_CLASSES:
            raise ValueError(f"probs must have {NUM_CLASSES} entries, got {len(self.probs)}")


@runtime_checkable
class Predictor(Protocol):
    """Interface for a digit classifier backend.

    A Protocol means backends don't need to inherit from anything. Any class with a
    ``name`` attribute and a matching ``predict`` method counts as a Predictor
    (structural typing, i.e. "duck typing" that type-checkers understand).
    """

    #: Registry name of the backend, e.g. ``"pretrained"`` or ``"scratch_cnn"``.
    name: str

    def predict(self, crops: torch.Tensor) -> list[DigitPrediction]:
        """Classify a batch of digit crops.

        Args:
            crops: ``[N, 1, 28, 28]`` float tensor following the input contract in the
                module docstring.

        Returns:
            ``N`` predictions, in the same order as ``crops``.
        """
        ...


def probs_to_predictions(probs: torch.Tensor) -> list[DigitPrediction]:
    """Turn a ``[N, 10]`` probability tensor into ``N`` :class:`DigitPrediction` objects.

    Shared helper so every backend builds its results the same way: run the model,
    apply ``softmax`` to the logits, then call this.

    Args:
        probs: Rows of class probabilities (each row sums to ~1), any device.

    Returns:
        One prediction per row; ``label`` is the row's argmax.
    """
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise ValueError(f"expected probs of shape [N, {NUM_CLASSES}], got {list(probs.shape)}")

    probs = probs.detach().float().cpu()  # plain Python floats need CPU memory
    confidences, labels = probs.max(dim=1)
    return [
        DigitPrediction(label=int(label), confidence=float(conf), probs=tuple(row.tolist()))
        for label, conf, row in zip(labels, confidences, probs)
    ]
